import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import nvdiffrast.torch as dr
import mcubes

from utils.base_utils import az_el_to_points, sample_sphere
from utils.raw_utils import linear_to_srgb
from utils.ref_utils import generate_ide_fn


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 sdf_activation='none',
                 layer_activation='softplus'):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if layer_activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        elif layer_activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def sdf_normal(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return y[..., :1].detach(), gradients.detach()
    





class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation='exp'):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        if self.act == 'exp':
            return torch.ones([*x.shape[:-1], 1]) * torch.exp(self.variance * 10.0)
        elif self.act == 'linear':
            return torch.ones([*x.shape[:-1], 1]) * self.variance * 10.0
        elif self.act == 'square':
            return torch.ones([*x.shape[:-1], 1]) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        return torch.ones([*x.shape[:-1], 1]) * inv_s


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRFNetwork(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRFNetwork, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views, is_feature = False):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            if is_feature: return alpha,rgb, feature
            return alpha, rgb
        else:
            assert False

    def density(self, input_pts):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha


class IdentityActivation(nn.Module):
    def forward(self, x): return x


class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light = max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))

def make_predictor1(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid',
                   exp_max=0.0) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation == 'none':
        activation = IdentityActivation()
    elif activation == 'relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256
    if weight_norm:
        module = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module = nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module

def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid',
                   exp_max=0.0) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation == 'none':
        activation = IdentityActivation()
    elif activation == 'relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256
    if weight_norm:
        module = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module = nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module


def get_camera_plane_intersection(pts, dirs, poses):
    """
    compute the intersection between the rays and the camera XoY plane
    :param pts:      pn,3
    :param dirs:     pn,3
    :param poses:    pn,3,4
    :return:
    """
    R, t = poses[:, :, :3], poses[:, :, 3:]

    # transfer into human coordinate
    pts_ = (R @ pts[:, :, None] + t)[..., 0]  # pn,3
    dirs_ = (R @ dirs[:, :, None])[..., 0]  # pn,3

    hits = torch.abs(dirs_[..., 2]) > 1e-4
    dirs_z = dirs_[:, 2]
    dirs_z[~hits] = 1e-4
    dist = -pts_[:, 2] / dirs_z
    inter = pts_ + dist.unsqueeze(-1) * dirs_
    return inter, dist, hits


def expected_sin(mean, var):
    """Compute the mean of sin(x), x ~ N(mean, var)."""
    return torch.exp(-0.5 * var) * torch.sin(mean)  # large var -> small value.


def IPE(mean, var, min_deg, max_deg):
    scales = 2 ** torch.arange(min_deg, max_deg)
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
    scaled_var = torch.reshape(var[..., None, :] * scales[:, None] ** 2, shape)
    return expected_sin(torch.concat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1),
                        torch.concat([scaled_var] * 2, dim=-1))


def offset_points_to_sphere(points):
    points_norm = torch.norm(points, dim=-1)
    mask = points_norm > 0.999
    if torch.sum(mask) > 0:
        points = torch.clone(points)
        points[mask] /= points_norm[mask].unsqueeze(-1)
        points[mask] *= 0.999
        # points[points_norm>0.999] = 0
    return points


def get_sphere_intersection(pts, dirs):
    dtx = torch.sum(pts * dirs, dim=-1, keepdim=True)  # rn,1
    xtx = torch.sum(pts ** 2, dim=-1, keepdim=True)  # rn,1
    dist = dtx ** 2 - xtx + 1
    assert torch.sum(dist < 0) == 0
    dist = -dtx + torch.sqrt(dist + 1e-6)  # rn,1
    return dist


# this function is borrowed from NeuS
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_weights(sdf_fun, inv_fun, z_vals, origins, dirs):
    points = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2)  # pn,sn,3
    inv_s = inv_fun(points[:, :-1, :])[..., 0]  # pn,sn-1
    sdf = sdf_fun(points)[..., 0]  # pn,sn

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # pn,sn-1
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # pn,sn-1
    surface_mask = (cos_val < 0)  # pn,sn-1
    cos_val = torch.clamp(cos_val, max=0)

    dist = next_z_vals - prev_z_vals  # pn,sn-1
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # pn, sn-1
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) * surface_mask.float()
    weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    mid_sdf[~surface_mask] = -1.0
    return weights, mid_sdf


def get_intersection(sdf_fun, inv_fun, pts, dirs, sn0=128, sn1=9):
    """
    :param sdf_fun:
    :param inv_fun:
    :param pts:    pn,3
    :param dirs:   pn,3
    :param sn0:
    :param sn1:
    :return:
    """
    inside_mask = torch.norm(pts, dim=-1) < 0.999  # left some margin
    pn, _ = pts.shape
    hit_z_vals = torch.zeros([pn, sn1 - 1])
    hit_weights = torch.zeros([pn, sn1 - 1])
    hit_sdf = -torch.ones([pn, sn1 - 1])
    if torch.sum(inside_mask) > 0:
        pts = pts[inside_mask]
        dirs = dirs[inside_mask]
        max_dist = get_sphere_intersection(pts, dirs)  # pn,1
        with torch.no_grad():
            z_vals = torch.linspace(0, 1, sn0)  # sn0
            z_vals = max_dist * z_vals.unsqueeze(0)  # pn,sn0
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, pts, dirs)  # pn,sn0-1
            z_vals_new = sample_pdf(z_vals, weights, sn1, True)  # pn,sn1
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, pts, dirs)  # pn,sn1-1
            z_vals_mid = (z_vals_new[:, 1:] + z_vals_new[:, :-1]) * 0.5

        hit_z_vals[inside_mask] = z_vals_mid
        hit_weights[inside_mask] = weights
        hit_sdf[inside_mask] = mid_sdf
    return hit_z_vals, hit_weights, hit_sdf


class AppShadingNetwork(nn.Module):
    default_cfg = {
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 6,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 3.0,
        'refrac_freq': 6
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        feats_dim = 256

        # material MLPs
        self.metallic_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['metallic_init'] != 0:
            nn.init.constant_(self.metallic_predictor[-2].bias, self.cfg['metallic_init'])
        self.roughness_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['roughness_init'] != 0:
            nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
        self.albedo_predictor = make_predictor(feats_dim + 3, 3)

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)

        self.dir_enc_refrac, dir_dim_refrac = get_embedder(self.cfg['refrac_freq'], 3)
        self.pos_enc_refrac, pos_dim_refrac = get_embedder(self.cfg['refrac_freq'], 3)
        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        if self.cfg['sphere_direction']:
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=exp_max)
        else:
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        # inner lights are indirect lights
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='none')
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        self.transmisstion_weight = make_predictor(feats_dim + 3, 1)
       # nn.init.constant_(self.reflec_weight[-2].bias, self.cfg['inner_init'])
        self.iors = make_predictor(feats_dim + 3, 1)

        self.refrac_light = make_predictor(pos_dim_refrac + dir_dim_refrac , 3,activation='exp', exp_max=-0.5)
        nn.init.constant_(self.refrac_light[-2].bias, np.log(0.5))

        # human lights are the lights reflected from the photo capturer
        if self.cfg['human_light']:
            self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))

    def predict_human_light(self, points, reflective, human_poses, roughness):
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_specular_lights(self, points, feature_vectors, reflective, roughness, human_poses, step):
        human_light, human_weight = 0, 0
        ref_roughness_0 = self.sph_enc(reflective, torch.zeros_like(roughness,device='cuda:0'))
        ref_roughness = self.sph_enc(reflective, roughness)
        pts = self.pos_enc(points)
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1))
            direct_light_0 = self.outer_light(torch.cat([ref_roughness_0, sph_points], -1))
        else:
            direct_light = self.outer_light(ref_roughness)
            direct_light_0 = self.outer_light(ref_roughness_0)
      #  direct_light = self.bkgr(reflective)

        if self.cfg['human_light']:
            human_light, human_weight = self.predict_human_light(points, reflective, human_poses, roughness)

        indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
        indirect_light_0 = self.inner_light(torch.cat([pts, ref_roughness_0], -1))
        ref_ = self.dir_enc(reflective)
        occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1))  # this is occlusion prob
        occ_prob = occ_prob * 0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

        light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
                    1 - occ_prob_)
        light_0 = indirect_light_0 * occ_prob_ + (human_light * human_weight + direct_light_0 * (1 - human_weight)) * (
                    1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, light_0, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, points, feature_vectors, normals):
        roughness = torch.ones([normals.shape[0], 1])
        #print(normals.shape)
        #print(roughness.shape)
      #  exit(1)
        ref = self.sph_enc(normals, roughness)  # von Mises-Fisher distribution
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            light = self.outer_light(torch.cat([ref, sph_points], -1))
        else:
            light = self.outer_light(ref)
        return light

    def forward(self, points, normals, view_dirs, feature_vectors, human_poses, inter_results=False, step=None):
       # print(points.shape)
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)
        
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1)) 
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))

        transmission_weight = self.transmisstion_weight(torch.cat([feature_vectors, points], -1))

        # diffuse light
        diffuse_albedo = (1 - metallic) * albedo #/ torch.pi
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        diffuse_color = diffuse_albedo * diffuse_light
        
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo


        specular_light, specular_light_0, occ_prob, indirect_light, human_light = self.predict_specular_lights(points, feature_vectors,
                                                                                             reflective, roughness,
                                                                                             human_poses, step)
        temp_nov = torch.clamp(1 - NoV, min=0.0, max=1.0)


        schlick = 0.04 + (1 - 0.04) * temp_nov * temp_nov * temp_nov * temp_nov * temp_nov
        reflection_weight = torch.clamp(schlick, min=0, max=1)

      #  ref1_ = self.dir_enc1(view_dirs)
       # pts1 = self.pos_enc1(points)
        #refraction_light = self.refrac_light1(torch.cat([pts1.detach(), ref1_.detach(),feature_vectors.detach()], -1))
        refraction_light = self.refrac_light(torch.cat([self.pos_enc_refrac(points), self.dir_enc_refrac(view_dirs)], -1)) 

        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = points.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
                               boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        specular_color = specular_ref * specular_light 

        # fg_uv_0 = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0) * 0], -1)
        # pn, bn = points.shape[0], 1
        # fg_lookup_0 = dr.texture(self.FG_LUT, fg_uv_0.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
        #                        boundary_mode='clamp').reshape(pn, 2)
        # specular_ref_0 = (0.04 * fg_lookup_0[:, 0:1] + fg_lookup_0[:, 1:2])
        # specular_color_0 = specular_ref_0 * specular_light_0

        # integrated together
        color =(diffuse_color + specular_color) * (1 - transmission_weight) + \
        (reflection_weight * specular_light_0  + (1 - reflection_weight) * refraction_light  )* transmission_weight

        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(specular_color)
        color = linear_to_srgb(color) 
     #   color = torch.clamp(color, min=0.0, max=1.0)

        occ_info = {
            'reflective': reflective,
            'occ_prob': occ_prob,
            'transmission_weight': transmission_weight,
            'metallic': metallic,
        }
        
        if inter_results:
            intermediate_results = {
                'specular_albedo': specular_albedo,
                'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
                'specular_light': torch.clamp(linear_to_srgb(specular_light_0), min=0.0, max=1.0),
                'specular_color': torch.clamp(specular_color * (1 - transmission_weight) + reflection_weight * specular_light_0 * transmission_weight, min=0.0, max=1.0) ,
#torch.clamp(specular_color, min=0.0, max=1.0),

                'diffuse_albedo': diffuse_albedo,
                'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),

                'metallic': metallic,
                'transmission_weight': transmission_weight,
                'roughness': roughness,

                'occ_prob': torch.clamp(occ_prob, max=1.0, min=0.0),
                'indirect_light': indirect_light,
                'refraction_light': torch.clamp(linear_to_srgb((1 - reflection_weight) * refraction_light * transmission_weight ), min=0.0, max=1.0),
                'reflection_weight': reflection_weight,
            }
           #print(intermediate_results['refraction_light'])
           # exit(1)
            if self.cfg['human_light']:
                intermediate_results['human_light'] = linear_to_srgb(human_light)
            return color, occ_info, intermediate_results
        else:
            return color, occ_info

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        return metallic, roughness, albedo


class AppShadingNetwork_S2(nn.Module):
    default_cfg = {
        'human_light': False,
        'sphere_direction': True,
        'light_pos_freq': 6,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 5.0,
        'refrac_freq': 6,
    }

    def __init__(self, cfg, stage1):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        feats_dim = 256
        self.stage1_network = stage1
        # material MLPs
        self.metallic_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['metallic_init'] != 0:
            nn.init.constant_(self.metallic_predictor[-2].bias, self.cfg['metallic_init'])
        self.roughness_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['roughness_init'] != 0:
            nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
        self.albedo_predictor = make_predictor(feats_dim + 3, 3)

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)

       # self.dir_enc1, dir_dim1 = get_embedder(12, 3)
       # self.pos_enc1, pos_dim1 = get_embedder(10, 3)

        self.dir_enc_refrac, dir_dim_refrac = get_embedder(self.cfg['refrac_freq'], 3)
        self.pos_enc_refrac, pos_dim_refrac = get_embedder(self.cfg['refrac_freq'], 3)

        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        if self.cfg['sphere_direction']:
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=exp_max)
        else:
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        # inner lights are indirect lights
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='sigmoid')
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        self.refrac_weight = make_predictor(feats_dim + 3, 1, activation='sigmoid')
     #   nn.init.constant_(self.refrac_weight[-2].bias, self.cfg['inner_init'])
        self.transmisstion_weight = make_predictor(feats_dim + 3, 1)
        self.refrac_light = make_predictor(pos_dim_refrac + dir_dim_refrac, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.refrac_light[-2].bias, np.log(0.5))

        # human lights are the lights reflected from the photo capturer
        if self.cfg['human_light']:
            self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))

    def predict_human_light(self, points, reflective, human_poses, roughness):
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights


    def predict_specular_lights(self, points, feature_vectors, reflective, roughness, step):
        human_light, human_weight = 0, 0
        ref_roughness_0 = self.sph_enc(reflective, torch.zeros_like(roughness,device='cuda:0'))
        ref_roughness = self.sph_enc(reflective, roughness)
        pts = self.pos_enc(points)
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            direct_light = self.stage1_network.color_network.outer_light(torch.cat([ref_roughness, sph_points], -1))
            direct_light_0 = self.stage1_network.color_network.outer_light(torch.cat([ref_roughness_0, sph_points], -1))
        else:
            direct_light = self.stage1_network.color_network.outer_light(ref_roughness)
            direct_light_0 = self.stage1_network.color_network.outer_light(ref_roughness_0)
      #  direct_light = self.bkgr(reflective)

    

        indirect_light = self.stage1_network.color_network.inner_light(torch.cat([pts, ref_roughness], -1))
        indirect_light_0 = self.stage1_network.color_network.inner_light(torch.cat([pts, ref_roughness_0], -1))
        ref_ = self.dir_enc(reflective)
        occ_prob = self.stage1_network.color_network.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1))  # this is occlusion prob
       # occ_prob = occ_prob * 0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

        light = indirect_light * occ_prob_ + direct_light * (
                    1 - occ_prob_)
        light_0 = indirect_light_0 * occ_prob_ +  direct_light_0 * (
                    1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, light_0, occ_prob, indirect_light
    # def predict_specular_lights(self, points, feature_vectors, reflective, roughness, step):
    #   #  print(points.shape)
    #    # print(feature_vectors.shape)
    #    # print(reflective.shape)
    #    # print(roughness.shape)
    #     human_light, human_weight = 0, 0
    #     ref_roughness = self.sph_enc(reflective, roughness)
    #   #  print(reflective.shape)
    #   #  print(roughness.shape)
    #   #  print(ref_roughness.shape)
    #     pts = self.pos_enc(points)
    #     if self.cfg['sphere_direction']:
    #         sph_points = offset_points_to_sphere(points)
    #         sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)
    #         sph_points = self.sph_enc(sph_points, roughness)
    #         direct_light = self.stage1_network.color_network.outer_light(torch.cat([ref_roughness, sph_points], -1))
    #     else:
    #         direct_light = self.stage1_network.color_network.outer_light(ref_roughness)
    #   #  direct_light = self.stage1_network.color_network.bkgr(reflective)

    #     #if self.cfg['human_light']:
    #     #    human_light, human_weight = self.predict_human_light(points, reflective, roughness)

    #     indirect_light = self.stage1_network.color_network.inner_light(torch.cat([pts, ref_roughness], -1))
    #     ref_ = self.dir_enc(reflective)
    #     occ_prob = self.stage1_network.color_network.inner_weight(torch.cat([pts, ref_], -1))  # this is occlusion prob
    #     occ_prob = occ_prob * 0.5 + 0.5
    #     occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

    #     light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
    #                 1 - occ_prob_)
    #     indirect_light = indirect_light * occ_prob_
    #     return light, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, points, feature_vectors, normals):
       # print(normals.shape)
        roughness = torch.ones([normals.shape[0],1, 1])
        ref = self.sph_enc(normals, roughness)  # von Mises-Fisher distribution
      #  print(normals.shape)
       # print(roughness.shape)
       # print(ref.shape)
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            light = self.stage1_network.color_network.outer_light(torch.cat([ref, sph_points], -1))
        else:
            light = self.stage1_network.color_network.outer_light(ref)
        return light

    def forward(self, points, normals, view_dirs, feature_vectors, ior_ratios, is_internal,next_dir, inter_results=False, step=None):
        #print(points.shape)
        #print(normals.shape)
        #print(view_dirs.shape)
       # print(ior_ratios.shape)
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)

        metallic = self.stage1_network.color_network.metallic_predictor(torch.cat([feature_vectors, points], -1)) 
        roughness = self.stage1_network.color_network.roughness_predictor(torch.cat([feature_vectors, points], -1)) 
        albedo = self.stage1_network.color_network.albedo_predictor(torch.cat([feature_vectors, points], -1))

        transmission_weight = self.stage1_network.color_network.transmisstion_weight(torch.cat([feature_vectors, points], -1))

      #  ior = self.iors(torch.cat([feature_vectors, points], -1))
        #print(ior.min())
        #exit(1)
        # diffuse light
        diffuse_albedo = (1 - metallic) * albedo #/ torch.pi
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        diffuse_color = diffuse_albedo * diffuse_light

        # specular light
        #transmission_specular = (1-1.45) * (1-1.45) / (1+1.45) / (1+1.45)
     #   transmission_specular = transmission_specular * transmission_specular
        
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo


        specular_light, specular_light_0, occ_prob, indirect_light = self.predict_specular_lights(points, feature_vectors,
                                                                                             reflective, roughness,
                                                                                             step)
    
        temp_nov = torch.clamp(1 - NoV, min=0.0, max=1.0)


        schlick = 0.04 + (1 - 0.04) * temp_nov * temp_nov * temp_nov * temp_nov * temp_nov
        reflection_weight = torch.clamp(schlick, min=0, max=1)

        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = points.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
                               boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, None, 0:1] + fg_lookup[:, None, 1:2])
        specular_color = specular_ref * specular_light

        # fg_uv_0 = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0) * 0], -1)
        # pn, bn = points.shape[0], 1
        # fg_lookup_0 = dr.texture(self.FG_LUT, fg_uv_0.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
        #                        boundary_mode='clamp').reshape(pn, 2)
        # specular_ref_0 = (0.04 * fg_lookup_0[:, 0:1] + fg_lookup_0[:, 1:2])
        # specular_color_0 = specular_ref_0 * specular_light_0

        # integrated together
        color =  (diffuse_color + specular_color) * (1 - transmission_weight) + \
        (reflection_weight * specular_light_0  )* transmission_weight
        #print(specular_albedo.shape)
        #print(fg_lookup.shape)
        if is_internal: color = color * 0
     
        #exit(1)
        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(specular_color)
        color = linear_to_srgb(color)

        occ_info = {
            'reflective': reflective,
            'occ_prob': occ_prob,
            'transmission_weight': transmission_weight,
            'refraction_coefficient': (1 - reflection_weight) * transmission_weight
        }
        
        if inter_results:
            intermediate_results = {
                'specular_albedo': specular_albedo,
                'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
                'specular_light': torch.clamp(linear_to_srgb(specular_light_0), min=0.0, max=1.0),
                'specular_color': torch.clamp(specular_color * (1 - transmission_weight) + reflection_weight * specular_light_0 * transmission_weight, min=0.0, max=1.0) ,
#torch.clamp(specular_color, min=0.0, max=1.0),

                'diffuse_albedo': diffuse_albedo,
                'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),

                'transmission_weight': transmission_weight,
                'roughness': roughness,

                'occ_prob': torch.clamp(occ_prob, max=1.0, min=0.0),
                'indirect_light': indirect_light,
               # 'refraction_light': torch.clamp(linear_to_srgb((1 - reflection_weight) * refraction_light * transmission_weight ), min=0.0, max=1.0),
                'reflection_weight': reflection_weight,
            }
           #print(intermediate_results['refraction_light'])
           # exit(1)
            if self.cfg['human_light']:
                intermediate_results['human_light'] = linear_to_srgb(human_light)
            return color, occ_info, intermediate_results
        else:
            return color, occ_info

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        return metallic, roughness, albedo



class InfOutNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.dir_enc, dir_dim = get_embedder(10, 3)
        run_dim = 256
        self.module0 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(dir_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, 3)),
            nn.ReLU(),
        )
      

    def forward(self, x):
        x = self.dir_enc(x)
        x = self.module0(x)
        return x
    


class IoRNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc, pos_dim = get_embedder(6, 3)
        run_dim = 256
        self.module0 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(pos_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.utils.weight_norm(nn.Linear(run_dim, 1)),
            nn.Sigmoid(),
        )
      

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.module0(x)
        return x
    

class ThicknessNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc, pos_dim = get_embedder(6, 3)
        run_dim = 256
        self.module0 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(pos_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.utils.weight_norm(nn.Linear(run_dim, 1)),
            nn.Sigmoid(),
        )
      

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.module0(x)
        return x
    
class MaterialFeatsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc, input_dim = get_embedder(8, 3)
        run_dim = 256
        self.module0 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
        )
        self.module1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim + run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        input = x
        x = self.module0(x)
        return self.module1(torch.cat([x, input], -1))


def saturate_dot(v0, v1):
    return torch.clamp(torch.sum(v0 * v1, dim=-1, keepdim=True), min=0.0, max=1.0)




class AppShadingNetwork_DiffuseInner(nn.Module):
    default_cfg = {
        'human_light': False,
        'sphere_direction': True,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 3.0,
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        feats_dim = 256

        # material MLPs
        self.metallic_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['metallic_init'] != 0:
            nn.init.constant_(self.metallic_predictor[-2].bias, self.cfg['metallic_init'])
        self.roughness_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['roughness_init'] != 0:
            nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
        self.albedo_predictor = make_predictor(feats_dim + 3, 3)

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)
        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        if self.cfg['sphere_direction']:
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=exp_max)
        else:
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        # inner lights are indirect lights
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='sigmoid')
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        self.transmisstion_weight = make_predictor(feats_dim + 3, 1)
       # nn.init.constant_(self.reflec_weight[-2].bias, self.cfg['inner_init'])
        self.iors = make_predictor(feats_dim + 3, 1)

        self.refrac_light = make_predictor(pos_dim + feats_dim + dir_dim , 3,activation='exp', exp_max=exp_max)
        nn.init.constant_(self.refrac_light[-2].bias, np.log(0.5))

        # human lights are the lights reflected from the photo capturer
        if self.cfg['human_light']:
            self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))

    def predict_human_light(self, points, reflective, human_poses, roughness):
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_specular_lights(self, points, feature_vectors, reflective, roughness, human_poses, step):
        human_light, human_weight = 0, 0
        ref_roughness = self.sph_enc(reflective, torch.zeros_like(roughness,device='cuda:0'))
        pts = self.pos_enc(points)
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1))
        else:
            direct_light = self.outer_light(ref_roughness)
      #  direct_light = self.bkgr(reflective)

        if self.cfg['human_light']:
            human_light, human_weight = self.predict_human_light(points, reflective, human_poses, roughness)

        indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
        ref_ = self.dir_enc(reflective)
        occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1))  # this is occlusion prob
        #occ_prob = occ_prob * 0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

        light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
                    1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, points, feature_vectors, normals):
        roughness = torch.ones([normals.shape[0], 1])
        #print(normals.shape)
        #print(roughness.shape)
      #  exit(1)
        ref = self.sph_enc(normals, roughness)  # von Mises-Fisher distribution
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            light = self.outer_light(torch.cat([ref, sph_points], -1))
        else:
            light = self.outer_light(ref)
        return light

    def forward(self, points, normals, view_dirs, feature_vectors, human_poses, inter_results=False, step=None):
       # print(points.shape)
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)

        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1)) * 0
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))

        transmission_weight = self.transmisstion_weight(torch.cat([feature_vectors, points], -1)) * 0

      #  ior = self.iors(torch.cat([feature_vectors, points], -1))
        #print(ior.min())
        #exit(1)
        # diffuse light
        diffuse_albedo =  albedo 
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        diffuse_color = diffuse_albedo * diffuse_light

     
        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(diffuse_color)
        color = linear_to_srgb(diffuse_color)
     #   color = torch.clamp(color, min=0.0, max=1.0)

        occ_info = {
            'reflective': reflective,
        }
        

        return color, occ_info

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        return metallic, roughness, albedo


def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                 #   print(xx)
                  #  exit(1)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).detach()
                  # print(val.shape)
                    outside_mask = torch.norm(pts, dim=-1) >= 1.0
                    val[outside_mask] = outside_val
                    val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, outside_val=1.0):
    u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class AppShadingNetwork_SpecInner(nn.Module):
    default_cfg = {
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 10.0,
        'refrac_freq': 6
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        feats_dim = 256

        # material MLPs
        self.metallic_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['metallic_init'] != 0:
            nn.init.constant_(self.metallic_predictor[-2].bias, self.cfg['metallic_init'])
        self.roughness_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['roughness_init'] != 0:
            nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
        self.albedo_predictor = make_predictor(feats_dim + 3, 3)

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)

        self.dir_enc_refrac, dir_dim_refrac = get_embedder(self.cfg['refrac_freq'], 3)
        self.pos_enc_refrac, pos_dim_refrac = get_embedder(self.cfg['refrac_freq'], 3)
        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        if self.cfg['sphere_direction']:
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=exp_max)
        else:
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        # inner lights are indirect lights
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='none')
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        self.transmisstion_weight = make_predictor(feats_dim + 3, 1)
       # nn.init.constant_(self.reflec_weight[-2].bias, self.cfg['inner_init'])
        self.iors = make_predictor(feats_dim + 3, 1)

        self.refrac_light = make_predictor(pos_dim_refrac + dir_dim_refrac , 3,activation='exp', exp_max=exp_max)
        nn.init.constant_(self.refrac_light[-2].bias, np.log(0.5))

        # human lights are the lights reflected from the photo capturer
        if self.cfg['human_light']:
            self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))

    def predict_human_light(self, points, reflective, human_poses, roughness):
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_specular_lights(self, points, feature_vectors, reflective, roughness, human_poses, step):
        human_light, human_weight = 0, 0
        ref_roughness_0 = self.sph_enc(reflective, torch.zeros_like(roughness,device='cuda:0'))
        ref_roughness = self.sph_enc(reflective, roughness)
        pts = self.pos_enc(points)
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1))
            direct_light_0 = self.outer_light(torch.cat([ref_roughness_0, sph_points], -1))
        else:
            direct_light = self.outer_light(ref_roughness)
            direct_light_0 = self.outer_light(ref_roughness_0)
      #  direct_light = self.bkgr(reflective)

        if self.cfg['human_light']:
            human_light, human_weight = self.predict_human_light(points, reflective, human_poses, roughness)

        indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
        indirect_light_0 = self.inner_light(torch.cat([pts, ref_roughness_0], -1))
        ref_ = self.dir_enc(reflective)
        occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1))  # this is occlusion prob
       # occ_prob = occ_prob * 0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

        light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
                    1 - occ_prob_)
        light_0 = indirect_light_0 * occ_prob_ + (human_light * human_weight + direct_light_0 * (1 - human_weight)) * (
                    1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, light_0, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, points, feature_vectors, normals):
        roughness = torch.ones([normals.shape[0], 1])
        #print(normals.shape)
        #print(roughness.shape)
      #  exit(1)
        ref = self.sph_enc(normals, roughness)  # von Mises-Fisher distribution
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1)
            sph_points = self.sph_enc(sph_points, roughness)
            light = self.outer_light(torch.cat([ref, sph_points], -1))
        else:
            light = self.outer_light(ref)
        return light

    def forward(self, points, normals, view_dirs, feature_vectors, human_poses, inter_results=False, step=None):
       # print(points.shape)
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)

        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1)) * 0
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1)) 
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))

        transmission_weight = self.transmisstion_weight(torch.cat([feature_vectors, points], -1)) 

      #  ior = self.iors(torch.cat([feature_vectors, points], -1))
        #print(ior.min())
        #exit(1)
        # diffuse light
        diffuse_albedo = (1 - metallic) * albedo #/ torch.pi
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        diffuse_color = diffuse_albedo * diffuse_light

        # specular light
        #transmission_specular = (1-1.45) * (1-1.45) / (1+1.45) / (1+1.45)
     #   transmission_specular = transmission_specular * transmission_specular
        
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo


        specular_light, specular_light_0, occ_prob, indirect_light, human_light = self.predict_specular_lights(points, feature_vectors,
                                                                                             reflective, roughness,
                                                                                             human_poses, step)
        

        # cosThetaI = torch.sum(normals * view_dirs, dim=-1, keepdim=True)
        # sin2ThetaI = (1 - cosThetaI * cosThetaI).clamp(min = 0)
        # sin2ThetaT = ior * ior * sin2ThetaI
        # totalInerR = (sin2ThetaT >= 1).view(-1)
        # cosThetaT = torch.sqrt(1 - sin2ThetaI.clamp(max = 1))
        # wt = ior * -view_dirs + (ior * cosThetaI - cosThetaT) * normals

        # wt should be already unit length, Numerical error?
        # wt = wt / wt.norm(p=2, dim=1, keepdim=True).detach()
      #  wt = wt / wt.norm(p=2, dim=1, keepdim=True)
       # print(view_dirs.shape)
       # print(wt.shape)
       # exit(1)
    #     ref_ = self.dir_enc(view_dirs)
    #    # refr_ = self.dir_enc(wt)
    #     pts = self.pos_enc(points)
       # reflection_weight = self.reflec_weight(torch.cat([pts.detach(), ref_.detach()], -1))
       # reflection_weight = reflection_weight * 0.5 + 0.5
        temp_nov = torch.clamp(1 - NoV, min=0.0, max=1.0)


        schlick = 0.04 + (1 - 0.04) * temp_nov * temp_nov * temp_nov * temp_nov * temp_nov
        reflection_weight = torch.clamp(schlick, min=0, max=1)

      #  ref1_ = self.dir_enc1(view_dirs)
       # pts1 = self.pos_enc1(points)
        #refraction_light = self.refrac_light1(torch.cat([pts1.detach(), ref1_.detach(),feature_vectors.detach()], -1))
        refraction_light = self.refrac_light(torch.cat([self.pos_enc_refrac(points), self.dir_enc_refrac(view_dirs)], -1)) 

        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = points.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
                               boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        specular_color = specular_ref * specular_light 

        # fg_uv_0 = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0) * 0], -1)
        # pn, bn = points.shape[0], 1
        # fg_lookup_0 = dr.texture(self.FG_LUT, fg_uv_0.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
        #                        boundary_mode='clamp').reshape(pn, 2)
        # specular_ref_0 = (0.04 * fg_lookup_0[:, 0:1] + fg_lookup_0[:, 1:2])
        # specular_color_0 = specular_ref_0 * specular_light_0

        # integrated together
        color =(diffuse_color + specular_color) * (1 - transmission_weight) + \
        (reflection_weight * specular_light_0  + (1 - reflection_weight) * refraction_light  )* transmission_weight 

        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(specular_color)
        color = linear_to_srgb(color)
     #   color = torch.clamp(color, min=0.0, max=1.0)

        occ_info = {
            'reflective': reflective,
            'occ_prob': occ_prob,
            'transmission_weight': transmission_weight,
        }
        
        if inter_results:
            intermediate_results = {
                'specular_albedo': specular_albedo,
                'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
                'specular_light': torch.clamp(linear_to_srgb(specular_light_0), min=0.0, max=1.0),
                'specular_color': torch.clamp(color, min=0.0, max=1.0) * 0,
#torch.clamp(specular_color, min=0.0, max=1.0),

                'diffuse_albedo': diffuse_albedo,
                'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),

                'transmission_weight': transmission_weight,
                'roughness': roughness,

                'occ_prob': torch.clamp(occ_prob, max=1.0, min=0.0),
                'indirect_light': indirect_light,
                'refraction_light': torch.clamp(linear_to_srgb((1 - reflection_weight) * refraction_light * transmission_weight ), min=0.0, max=1.0),
                'reflection_weight': reflection_weight,
            }
           #print(intermediate_results['refraction_light'])
           # exit(1)
            if self.cfg['human_light']:
                intermediate_results['human_light'] = linear_to_srgb(human_light)
            return color, occ_info, intermediate_results
        else:
            return color, occ_info

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        return metallic, roughness, albedo