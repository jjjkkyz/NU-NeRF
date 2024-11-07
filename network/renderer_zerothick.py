import cv2

#import raytracing
import open3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import network.tracing
import trimesh, trimesh.exchange.export
from dataset.database import parse_database_name, get_database_split, BaseDatabase
from network.field import SDFNetwork, SingleVarianceNetwork, NeRFNetwork, AppShadingNetwork, get_intersection, \
    extract_geometry, sample_pdf, AppShadingNetwork_S2, InfOutNetwork, IoRNetwork, ThicknessNetwork, AppShadingNetwork_DiffuseInner, AppShadingNetwork_SpecInner
from utils.base_utils import color_map_forward, downsample_gaussian_blur
from utils.raw_utils import linear_to_srgb,srgb_to_linear
from .DiffRender import Ray,Intersection,Scene
from tqdm import trange
from utils.base_utils import load_cfg

def build_imgs_info(database: BaseDatabase, img_ids, is_nerf=False):
    images = [database.get_image(img_id) for img_id in img_ids]
    poses = [database.get_pose(img_id) for img_id in img_ids]
    Ks = [database.get_K(img_id) for img_id in img_ids]

    images = np.stack(images, 0)
    if is_nerf:
        images = color_map_forward(images).astype(np.float32)
        masks = [database.get_depth(img_id)[1] for img_id in img_ids]
        masks = np.stack(masks, 0)
    else:
        images = color_map_forward(images).astype(np.float32)
    Ks = np.stack(Ks, 0).astype(np.float32)
    poses = np.stack(poses, 0).astype(np.float32)

    imgs_info = {
        'imgs': images, 
        'Ks': Ks, 
        'poses': poses,
    }

    if is_nerf:
        imgs_info['masks'] = masks
    
    return imgs_info


def imgs_info_to_torch(imgs_info, device='cpu'):
    for k, v in imgs_info.items():
        if k.startswith('imgs'):
            v = torch.from_numpy((v).astype(np.float32))
        else:
            v = torch.from_numpy(v)
        if k.startswith('imgs'): v = v.permute(0, 3, 1, 2)
        imgs_info[k] = v.to(device)
    return imgs_info


def imgs_info_slice(imgs_info, idxs):
    new_imgs_info = {}
    for k, v in imgs_info.items():
        new_imgs_info[k] = v[idxs]
    return new_imgs_info


def imgs_info_to_cuda(imgs_info):
    for k, v in imgs_info.items():
        imgs_info[k] = v.cuda()
    return imgs_info


def imgs_info_downsample(imgs_info, ratio):
    b, _, h, w = imgs_info['imgs'].shape
    dh, dw = int(ratio * h), int(ratio * w)
    imgs_info_copy = {k: v for k, v in imgs_info.items()}
    imgs_info_copy['imgs'], imgs_info_copy['Ks'] = [], []
    for bi in range(b):
        img = imgs_info['imgs'][bi].cpu().numpy().transpose([1, 2, 0])
        img = downsample_gaussian_blur(img, ratio)
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)
        imgs_info_copy['imgs'].append(torch.from_numpy(img).permute(2, 0, 1))
        K = torch.from_numpy(np.diag([dw / w, dh / h, 1]).astype(np.float32)) @ imgs_info['Ks'][bi]
        imgs_info_copy['Ks'].append(K)

    imgs_info_copy['imgs'] = torch.stack(imgs_info_copy['imgs'], 0)
    imgs_info_copy['Ks'] = torch.stack(imgs_info_copy['Ks'], 0)
    return imgs_info_copy


class NeROShapeRenderer(nn.Module):
    default_cfg = {
        # standard deviation for opacity density
        'std_net': 'default',
        'std_act': 'exp',
        'inv_s_init': 0.3,
        'freeze_inv_s_step': None,

        # geometry network
        'sdf_net': 'default',
        'sdf_activation': 'none',
        'sdf_bias': 0.5,
        'sdf_n_layers': 8,
        'sdf_freq': 6,
        'sdf_d_out': 257,
        'geometry_init': True,

        # shader network
        'shader_config': {},

        # sampling strategy
        'n_samples': 64,
        'n_bg_samples': 32,
        'inf_far': 1000.0,
        'n_importance': 64,
        'up_sample_steps': 4,  # 1 for simple coarse-to-fine sampling
        'perturb': 1.0,
        'anneal_end': 50000,
        'train_ray_num': 512,
        'test_ray_num': 1024,
        'clip_sample_variance': True,
        'is_nerf': False,
        # dataset
        'database_name': 'nerf_synthetic/lego/black_800',

        # validation
        'test_downsample_ratio': True,
        'downsample_ratio': 0.5,
        'val_geometry': False,

        # losses
        'rgb_loss': 'charbonier',
        'apply_occ_loss': True,
        'occ_loss_step': 20000,
        'occ_loss_max_pn': 2048,
        'occ_sdf_thresh': 0.01,

        "fixed_camera": False,
    }

    def __init__(self, cfg, training=True):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        self.is_nerf = self.cfg['is_nerf']

        self.sdf_network = SDFNetwork(d_out=self.cfg['sdf_d_out'], d_in=3, d_hidden=256,
                                      n_layers=self.cfg['sdf_n_layers'],
                                      skip_in=[self.cfg['sdf_n_layers'] // 2], multires=self.cfg['sdf_freq'],
                                      bias=self.cfg['sdf_bias'], scale=1.0,
                                      geometric_init=self.cfg['geometry_init'],
                                      weight_norm=True, sdf_activation=self.cfg['sdf_activation'])

        self.deviation_network = SingleVarianceNetwork(init_val=self.cfg['inv_s_init'], activation=self.cfg['std_act'])

        # background nerf is a nerf++ model (this is outside the unit bounding sphere, so we call it outer nerf)
        self.outer_nerf = NeRFNetwork(D=8, d_in=4, d_in_view=3, W=256, multires=10, multires_view=4, output_ch=4,
                                      skips=[4], use_viewdirs=True)
        nn.init.constant_(self.outer_nerf.rgb_linear.bias, np.log(0.5))

        self.color_network = AppShadingNetwork(self.cfg['shader_config'])
        
        
        self.sdf_inter_fun = lambda x: self.sdf_network.sdf(x)
        self.infinity_far_bkgr = InfOutNetwork()
        
        if training:
            self._init_dataset()

    def _init_dataset(self):
        # train/test split
        self.database = parse_database_name(self.cfg['database_name'], self.cfg['dataset_dir'])
        self.train_ids, self.test_ids = get_database_split(self.database)
        self.train_ids = np.asarray(self.train_ids)

        self.train_imgs_info = build_imgs_info(self.database, self.train_ids, self.is_nerf)
        self.train_imgs_info = imgs_info_to_torch(self.train_imgs_info, 'cpu')
        b, _, h, w = self.train_imgs_info['imgs'].shape
        print(f'training size {h} {w} ...')
        self.train_num = len(self.train_ids)

        self.test_imgs_info = build_imgs_info(self.database, self.test_ids, self.is_nerf)
        self.test_imgs_info = imgs_info_to_torch(self.test_imgs_info, 'cpu')
        self.test_num = len(self.test_ids)

        # clean the data if we already have
        if hasattr(self, 'train_batch'):
            del self.train_batch

        self.train_batch, self.train_poses, self.tbn, _, _ = self._construct_nerf_ray_batch(
            self.train_imgs_info) if self.is_nerf else self._construct_ray_batch(self.train_imgs_info)
        self.train_poses = self.train_poses.float().cuda()

        self._shuffle_train_batch()

    def _shuffle_train_batch(self):
        self.train_batch_i = 0
        shuffle_idxs = torch.randperm(self.tbn, device='cpu')  # shuffle
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[shuffle_idxs]

    def _construct_ray_batch(self, imgs_info, device='cpu'):
        imn, _, h, w = imgs_info['imgs'].shape
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.to(device)
        coords = coords.float()[None, :, :, :].repeat(imn, 1, 1, 1)  # imn,h,w,2
        coords = coords.reshape(imn, h * w, 2)
        coords = torch.cat([coords + 0.5, torch.ones(imn, h * w, 1, dtype=torch.float32, device=device)],
                           2)  # imn,h*w,3

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        dirs = coords @ torch.inverse(imgs_info['Ks']).permute(0, 2, 1)
        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn,3,4

        rn = imn * h * w
        ray_batch = {
            'dirs': dirs.float().reshape(rn, 3).to(device),
            'rgbs': imgs.float().reshape(rn, 3).to(device),
            'idxs': idxs.long().reshape(rn, 1).to(device),
        }
        return ray_batch, poses, rn, h, w

    def _construct_nerf_ray_batch(self, imgs_info, device='cpu', is_train=True):
        imn, _, h, w = imgs_info['imgs'].shape

        i, j = torch.meshgrid(torch.linspace(0, w - 1, w),
                              torch.linspace(0, h - 1, h))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()

        K = imgs_info['Ks'][0]
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn,3,4
        if is_train:
            masks = imgs_info['masks'].reshape(imn, h * w)

        rays_d = [torch.sum(dirs[..., None, :].cpu() * poses[i, :3, :3], -1) for i in range(imn)]
        rays_d = torch.stack(rays_d, 0).reshape(imn, h * w, 3)
        rays_o = [poses[i, :3, -1].expand(rays_d[0].shape) for i in range(imn)]
       
        rays_o = torch.stack(rays_o, 0).reshape(imn, h * w, 3)
        rn = imn * h * w
        ray_batch = {
            # 'dirs': dirs.float().reshape(rn, 3).to(device),
            'rgbs': imgs.float().reshape(rn, 3).to(device),
            'idxs': idxs.long().reshape(rn, 1).to(device),
            'rays_o': rays_o.float().reshape(rn, 3).to(device),
            'rays_d': rays_d.float().reshape(rn, 3).to(device),
        }
        if is_train:
            ray_batch['masks'] = masks.float().reshape(rn).to(device)
        return ray_batch, poses, rn, h, w

    # def _construct_render_batch(self, imgs_info, device='cpu'):
    #     imn, _, h, w = imgs_info['imgs'].shape
    #     coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
    #     coords = coords.to(device)
    #     coords = coords.float()
    #     coords = coords.reshape(h * w, 2)
    #     coords = torch.cat([coords + 0.5, torch.ones(h * w, 1, dtype=torch.float32, device=device)], 1)  # imn,h*w,3

    #     # h*w,3 @ 3,3 => imn,h*w,3
    #     dirs = coords @ torch.inverse(imgs_info['Ks'][0]).permute(1, 0)
    #     poses = self._construct_render_poses().float().to(device)
    #     pose_n = poses.shape[0]
    #     idxs = torch.arange(pose_n, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1

    #     rn = pose_n * h * w
    #     dirs = dirs.unsqueeze(0).repeat(pose_n, 1, 1)
    #     ray_batch = {
    #         'dirs': dirs.float().to(device),
    #         'idxs': idxs.long().to(device),
    #     }
    #     return ray_batch, poses, rn, h, w, pose_n

    def nvs(self, pose, K, h, w):
        device = 'cuda'
        K = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)
        pose = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(device)

        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.to(device)
        coords = coords.float()[None, :, :, :].repeat(1, 1, 1, 1)  # 1,h,w,2
        coords = coords.reshape(1, h * w, 2)
        coords = torch.cat([coords + 0.5, torch.ones(1, h * w, 1, dtype=torch.float32, device=device)], 2)  # 1,h*w,3

        # 1,h*w,3 @ imn,3,3 => 1,h*w,3
        dirs = coords @ torch.inverse(K).permute(0, 2, 1)
        idxs = torch.arange(1, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # 1,h*w,1

        rn = h * w
        ray_batch = {
            'dirs': dirs.float().reshape(rn, 3).to(device),
            'idxs': idxs.long().reshape(rn, 1).to(device),
        }

        trn = 1024
        output_color = []
        for ri in range(0, rn, trn):
            cur_ray_batch = {}
            for k, v in ray_batch.items(): cur_ray_batch[k] = v[ri:ri + trn]

            with torch.no_grad():
                rays_o, rays_d, near, far, human_poses = self._process_ray_batch(cur_ray_batch, pose)
                cur_outputs = self.render(rays_o, rays_d, near, far, human_poses, 0, 0, is_train=False, step=300000)
                output_color.append(cur_outputs['ray_rgb'].detach().cpu().numpy())

        output_color = np.reshape(np.concatenate(output_color, 0), [h, w, 3])
        return output_color

    def get_anneal_val(self, step):
        if self.cfg['anneal_end'] < 0:
            return 1.0
        else:
            return np.min([1.0, step / self.cfg['anneal_end']])

    @staticmethod
    def near_far_from_sphere(rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        near = torch.clamp(near, min=1e-3)
        return near, far

    def get_human_coordinate_poses(self, poses):
        pn = poses.shape[0]
        cam_cen = (-poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:])[..., 0]  # pn,3
        if self.cfg['fixed_camera']:
            pass
        else:
            cam_cen[..., 2] = 0

        Y = torch.zeros([1, 3]).expand(pn, 3)
        Y[:, 2] = -1.0
        Z = torch.clone(poses[:, 2, :3])  # pn, 3
        Z[:, 2] = 0
        Z = F.normalize(Z, dim=-1)
        X = torch.cross(Y, Z)  # pn, 3
        R = torch.stack([X, Y, Z], 1)  # pn,3,3
        t = -R @ cam_cen[:, :, None]  # pn,3,1
        return torch.cat([R, t], -1)

    def _process_ray_batch(self, ray_batch, poses):
        rays_d = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn

        rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]  # trn,3,1
        rays_o = rays_o[idxs, :, 0]  # rn,3
        rays_d = poses[idxs, :, :3].permute(0, 2, 1) @ rays_d.unsqueeze(-1)
        rays_d = rays_d[..., 0]  # rn,3

        rays_o = rays_o
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = self.near_far_from_sphere(rays_o, rays_d)

        human_poses = self.get_human_coordinate_poses(poses)
        return rays_o, rays_d, near, far, human_poses[idxs]  # rn, 3, 4

    def _process_nerf_ray_batch(self, ray_batch, poses):
        # dirs = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn
        rays_d = ray_batch['rays_d']
        rays_o = ray_batch['rays_o']
        # poses = poses[idxs, :3, :]
        # rays_d = torch.sum(dirs[..., None, :] * poses[..., :3], -1)
        # rays_o = poses[:, :, -1].expand(rays_d.shape)
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = torch.full((rays_o.shape[0], 1), 0.8), torch.full((rays_o.shape[0], 1), 4.5)

        return rays_o, rays_d, near, far, poses[idxs]  # rn, 3, 4

    # def _process_render_ray_batch(self, ray_batch, poses):
    #     dirs = ray_batch['dirs']  # rn,3

    #     render_n = poses.shape[0]
    #     rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]
    #     rays_o = rays_o[:, :, 0]

    #     rays_d = []
    #     for i in range(render_n):
    #         rays_d_i = poses[i, :, :3][None, ...].permute(0, 2, 1) @ dirs.unsqueeze(-1)
    #         rays_d_i = rays_d_i[..., 0]  # rn,3
    #         rays_d.append(rays_d_i[None, ...])

    #     rays_d = torch.cat(rays_d, dim=0)

    #     rays_d = F.normalize(rays_d, dim=-1)
    #     near, far = self.near_far_from_sphere(rays_o, rays_d)

    #     human_poses = self.get_human_coordinate_poses(poses)
    #     return rays_o, rays_d, near, far, human_poses  # rn, 3, 4

    def test_step(self, index, step, ):
        target_imgs_info, target_img_ids = self.test_imgs_info, self.test_ids
        imgs_info = imgs_info_slice(target_imgs_info, torch.from_numpy(np.asarray([index], np.int64)))
        gt_depth, gt_mask = self.database.get_depth(target_img_ids[index])  # used in evaluation
        is_nerf = self.is_nerf
        if self.cfg['test_downsample_ratio']:
            imgs_info = imgs_info_downsample(imgs_info, self.cfg['downsample_ratio'])
            h, w = gt_depth.shape
            dh, dw = int(self.cfg['downsample_ratio'] * h), int(self.cfg['downsample_ratio'] * w)
            gt_depth, gt_mask = cv2.resize(gt_depth, (dw, dh), interpolation=cv2.INTER_NEAREST), \
                cv2.resize(gt_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
        gt_depth, gt_mask = torch.from_numpy(gt_depth), torch.from_numpy(gt_mask.astype(np.int32))
        ray_batch, input_poses, rn, h, w = self._construct_nerf_ray_batch(imgs_info, is_train=False) \
            if is_nerf else self._construct_ray_batch(imgs_info)

        input_poses = input_poses.float().cuda()
        for k, v in ray_batch.items(): ray_batch[k] = v.cuda()

        trn = self.cfg['test_ray_num']
        outputs_keys = ['ray_rgb', 'gradient_error', 'normal', 'depth']
        outputs_keys += [
            'diffuse_albedo', 'diffuse_light', 'diffuse_color','refraction_light',
            'specular_albedo', 'specular_light', 'specular_color', 'specular_ref',
            'transmission_weight', 'roughness', 'occ_prob', 'indirect_light', 'occ_prob_gt',
        ]
        if self.color_network.cfg['human_light']:
            outputs_keys += ['human_light']

        outputs = {k: [] for k in outputs_keys}
        for ri in range(0, rn, trn):
            cur_ray_batch = {k: v[ri:ri + trn] for k, v in ray_batch.items()}
            rays_o, rays_d, near, far, human_poses = self._process_nerf_ray_batch(cur_ray_batch, input_poses) \
                if is_nerf else self._process_ray_batch(cur_ray_batch, input_poses)
            cur_outputs = self.render(rays_o, rays_d, near, far, human_poses, 0, 0, is_train=False, step=step,
                                      is_nerf=is_nerf)
            for k in outputs_keys: outputs[k].append(cur_outputs[k].detach())

        for k in outputs_keys: outputs[k] = torch.cat(outputs[k], 0)
        outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], ray_batch['rgbs'])

        outputs['gt_rgb'] = ray_batch['rgbs'].reshape(h, w, 3)
        outputs['ray_rgb'] = outputs['ray_rgb'].reshape(h, w, 3)

        # used in evaluation
        outputs['gt_depth'] = gt_depth.unsqueeze(-1)
        outputs['gt_mask'] = gt_mask.unsqueeze(-1)

        self.zero_grad()
        return outputs

    def train_step(self, step):
        rn = self.cfg['train_ray_num']
        is_nerf = self.is_nerf
        # fetch to gpu
        train_ray_batch = {k: v[self.train_batch_i:self.train_batch_i + rn].cuda() for k, v in self.train_batch.items()}
        self.train_batch_i += rn
        if self.train_batch_i + rn >= self.tbn: self._shuffle_train_batch()
        train_poses = self.train_poses.cuda()
        rays_o, rays_d, near, far, human_poses = self._process_nerf_ray_batch(train_ray_batch, train_poses) \
            if is_nerf else self._process_ray_batch(train_ray_batch, train_poses)

        outputs = self.render(rays_o, rays_d, near, far, human_poses, -1, self.get_anneal_val(step), is_train=True,
                              step=step, is_nerf=is_nerf)
        outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], train_ray_batch['rgbs'])  # ray_loss
        # np.savetxt('ray.txt',outputs['ray_rgb'].detach().cpu().numpy())
        # np.savetxt('rgbs.txt',train_ray_batch['rgbs'].detach().cpu().numpy())
        # exit(1)
        # if is_nerf:  # only nerf dataset add loss_mask
        #     outputs['loss_mask'] = F.l1_loss(train_ray_batch['masks'], outputs['acc'], reduction='mean')
        return outputs

    # def render_step(self, step):
    #     self.eval()

    #     render_batch, render_poses, rn, h, w, pose_n = self._construct_render_batch(self.train_imgs_info)
    #     render_poses = render_poses.float().cuda()
    #     for k, v in render_batch.items(): render_batch[k] = v.cuda()
    #     trn = self.cfg['test_ray_num']

    #     outputs_keys = ['ray_rgb', 'gradient_error', 'normal', 'depth']
    #     outputs_keys += [
    #         'diffuse_albedo', 'diffuse_light', 'diffuse_color',
    #         'specular_albedo', 'specular_light', 'specular_color', 'specular_ref',
    #         'metallic', 'roughness', 'occ_prob', 'indirect_light', 'occ_prob_gt',
    #     ]

    #     final_outputs = []
    #     for pose in range(pose_n):
    #         outputs = {k: [] for k in outputs_keys}
    #         for ri in trange(0, h * w, trn):
    #             cur_ray_batch = {k: v[pose, ri:ri + trn, :] for k, v in render_batch.items()}
    #             rays_o, rays_d, near, far, human_poses = self._process_ray_batch(cur_ray_batch, render_poses)
    #             cur_outputs = self.render(rays_o, rays_d, near, far, human_poses, 0, 0, is_train=False, step=step)
    #             for k in outputs_keys: outputs[k].append(cur_outputs[k].detach())

    #         for k in outputs_keys: outputs[k] = torch.cat(outputs[k], 0)
    #         outputs['ray_rgb'] = outputs['ray_rgb'].reshape(h, w, 3)
    #         import imageio
    #         imageio.imwrite("test_{}.png".format(pose), outputs['ray_rgb'].cpu())
    #         final_outputs.append(outputs)

    #     self.train()
    #     return outputs

    def compute_rgb_loss(self, rgb_pr, rgb_gt):
        if self.cfg['rgb_loss'] == 'l2':
            rgb_loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)
        elif self.cfg['rgb_loss'] == 'l1':
            rgb_loss = torch.sum(F.l1_loss(rgb_pr, rgb_gt, reduction='none'), -1)
        elif self.cfg['rgb_loss'] == 'smooth_l1':
            rgb_loss = torch.sum(F.smooth_l1_loss(rgb_pr, rgb_gt, reduction='none', beta=0.25), -1)
        elif self.cfg['rgb_loss'] == 'charbonier':
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        else:
            raise NotImplementedError
        return rgb_loss

    def density_activation(self, density, dists):
        return 1.0 - torch.exp(-F.softplus(density) * dists)

    def compute_density(self, points):
        points_norm = torch.norm(points, dim=-1, keepdim=True)
        points_norm = torch.clamp(points_norm, min=1e-3)
        sigma = self.outer_nerf.density(torch.cat([points / points_norm, 1.0 / points_norm], -1))[..., 0]
        return sigma

    @staticmethod
    def upsample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def sample_ray(self, rays_o, rays_d, near, far, perturb):
        n_samples = self.cfg['n_samples']
        n_bg_samples = self.cfg['n_bg_samples']
        n_importance = self.cfg['n_importance']
        up_sample_steps = self.cfg['up_sample_steps']

        # sample points
        batch_size = len(rays_o)
        z_vals = torch.linspace(0.0, 1.0, n_samples)  # sn
        z_vals = near + (far - near) * z_vals[None, :]  # rn,sn
        z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_bg_samples + 1.0), n_bg_samples)

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples

            mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
            upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
            lower = torch.cat([z_vals_outside[..., :1], mids], -1)
            t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
            z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_bg_samples

        # Up sample
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            sdf = self.sdf_network.sdf(pts).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                rn, sn = z_vals.shape
                if self.cfg['clip_sample_variance']:
                    inv_s = self.deviation_network(torch.empty([1, 3])).expand(rn, sn - 1)
                    inv_s = torch.clamp(inv_s, max=64 * 2 ** i)  # prevent too large inv_s
                else:
                    inv_s = torch.ones(rn, sn - 1) * 64 * 2 ** i
                new_z_vals = self.upsample(rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s)
                z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

        z_vals = torch.cat([z_vals, z_vals_outside], -1)
        return z_vals

    def render(self, rays_o, rays_d, near, far, human_poses, perturb_overwrite=-1, cos_anneal_ratio=0.0, is_train=True,
               step=None, is_nerf=False):
        """
        :param rays_o: rn,3
        :param rays_d: rn,3
        :param near:   rn,1
        :param far:    rn,1
        :param human_poses:     rn,3,4
        :param perturb_overwrite: set 0 for inference
        :param cos_anneal_ratio:
        :param is_train:
        :param step:
        :return:
        """
        perturb = self.cfg['perturb']
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        z_vals = self.sample_ray(rays_o, rays_d, near, far, perturb)
        ret = self.render_core(rays_o, rays_d, z_vals, human_poses, cos_anneal_ratio=cos_anneal_ratio, step=step,
                               is_train=is_train, is_nerf=is_nerf)
        return ret

    def compute_validation_info(self, z_vals, rays_o, rays_d, weights, human_poses, step):
        depth = torch.sum(weights * z_vals, -1, keepdim=True)  # rn,
        points = depth * rays_d + rays_o  # rn,3
        gradients = self.sdf_network.gradient(points)  # rn,3
        inner_mask = torch.norm(points, dim=-1, keepdim=True) <= 1.0
        outputs = {
            'depth': depth,  # rn,1
            'normal': ((F.normalize(gradients, dim=-1) + 1.0) * 0.5) * inner_mask,  # rn,3
        }

        feature_vector = self.sdf_network(points)[..., 1:]  # rn,f
        _, occ_info, inter_results = self.color_network(points, gradients, -F.normalize(rays_d, dim=-1), feature_vector,
                                                        human_poses, inter_results=True, step=step)
        _, occ_prob, _ = get_intersection(self.sdf_inter_fun, self.deviation_network, points, occ_info['reflective'],
                                          sn0=128, sn1=9)  # pn,sn-1
        occ_prob_gt = torch.sum(occ_prob, dim=-1, keepdim=True)
        outputs['occ_prob_gt'] = occ_prob_gt
        for k, v in inter_results.items(): inter_results[k] = v * inner_mask
        outputs.update(inter_results)
        return outputs

    def compute_sdf_alpha(self, points, dists, dirs, cos_anneal_ratio, step):
        # points [...,3] dists [...] dirs[...,3]
        sdf_nn_output = self.sdf_network(points)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]

        gradients = self.sdf_network.gradient(points)  # ...,3
        inv_s = self.deviation_network(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        if self.cfg['freeze_inv_s_step'] is not None and step < self.cfg['freeze_inv_s_step']:
            inv_s = inv_s.detach()

        true_cos = (dirs * gradients).sum(-1)  # [...]
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # [...]
        return alpha, gradients, feature_vector, inv_s, sdf

    def compute_density_alpha(self, points, dists, dirs, nerf):
        norm = torch.norm(points, dim=-1, keepdim=True)
        points = torch.cat([points / norm, 1.0 / norm], -1)
        density, color = nerf(points, dirs)  # [...,1] [...,3]
        alpha = self.density_activation(density[..., 0], dists)
        color = linear_to_srgb(torch.exp(torch.clamp(color, max=5.0))) 
        return alpha, color

    def compute_occ_loss(self, occ_info, points, sdf, gradients, dirs, step):
        if step < self.cfg['occ_loss_step']: return torch.zeros(1)

        occ_prob = occ_info['occ_prob']
        reflective = occ_info['reflective']

        # select a subset for occ loss
        # note we only apply occ loss on the surface
        inner_mask = torch.norm(points, dim=-1) < 0.999  # pn
        sdf_mask = torch.abs(sdf) < self.cfg['occ_sdf_thresh']
        normal_mask = torch.sum(gradients * dirs, -1) < 0  # pn
        mask = (inner_mask & normal_mask & sdf_mask)

        if torch.sum(mask) > self.cfg['occ_loss_max_pn']:
            indices = torch.nonzero(mask)[:, 0]  # npn
            idx = torch.randperm(indices.shape[0], device='cuda')  # npn
            indices = indices[idx[:self.cfg['occ_loss_max_pn']]]  # max_pn
            mask_new = torch.zeros_like(mask)
            mask_new[indices] = 1
            mask = mask_new

        if torch.sum(mask) > 0:
            inter_dist, inter_prob, inter_sdf = get_intersection(self.sdf_inter_fun, self.deviation_network,
                                                                 points[mask], reflective[mask], sn0=64,
                                                                 sn1=16)  # pn,sn-1
            occ_prob_gt = torch.sum(inter_prob, -1, keepdim=True)
            return F.l1_loss(occ_prob[mask], occ_prob_gt)
        else:
            return torch.zeros(1)

    def render_core(self, rays_o, rays_d, z_vals, human_poses, cos_anneal_ratio=0.0, step=None, is_train=True,
                    is_nerf=False):
        batch_size, n_samples = z_vals.shape

        # section length in original space
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # rn,sn-1
        dists = torch.cat([dists, dists[..., -1:]], -1)  # rn,sn
        mid_z_vals = z_vals + dists * 0.5

        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * mid_z_vals.unsqueeze(-1)
        inner_mask = torch.norm(points, dim=-1) <= 1.0
        outer_mask = ~inner_mask

        dirs = rays_d.unsqueeze(-2).expand(batch_size, n_samples, 3)
        human_poses_pt = human_poses.unsqueeze(-3).expand(batch_size, n_samples, 3, 4)
        dirs = F.normalize(dirs, dim=-1)
        alpha, sampled_color = torch.zeros(batch_size, n_samples), torch.zeros(batch_size, n_samples, 3)
        #bkgr_color = self.infinity_far_bkgr(dirs[:,0:1,:])
        if torch.sum(outer_mask) > 0:
            # if is_nerf:
            #     alpha[outer_mask] = torch.zeros_like(alpha[outer_mask])
            #     sampled_color[outer_mask] = torch.zeros_like(sampled_color[outer_mask])
            # else:
            alpha[outer_mask], sampled_color[outer_mask] = self.compute_density_alpha(points[outer_mask],
                                                                                       dists[outer_mask],
                                                                                       -dirs[outer_mask],
                                                                                      self.outer_nerf)
        
       # print(inner_mask[:,-1])
       # exit(1)
        #alpha[:,-1] = 1.
        #sampled_color[:,-1:,:] = linear_to_srgb(bkgr_color)
        alpha_bkgr, color_bkgr = alpha.clone(), sampled_color.clone()
#
        if torch.sum(inner_mask) > 0:
            alpha[inner_mask], gradients, feature_vector, inv_s, sdf = self.compute_sdf_alpha(points[inner_mask],
                                                                                              dists[inner_mask],
                                                                                              dirs[inner_mask],
                                                                                              cos_anneal_ratio, step)

            sampled_color[inner_mask], occ_info = self.color_network(points[inner_mask], gradients, -dirs[inner_mask],
                                                                     feature_vector, human_poses_pt[inner_mask],
                                                                     step=step)
            # Eikonal loss
            gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2
        else:
            gradient_error = torch.zeros(1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[...,
                          :-1]  # rn,sn
        color = (sampled_color * weights[..., None]).sum(dim=1)

        weights_bkgr = alpha_bkgr * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha_bkgr + 1e-7], -1), -1)[...,
                          :-1]  # rn,sn
        color_bkgr = (color_bkgr * weights_bkgr[..., None]).sum(dim=1)
        dir_enc = self.color_network.sph_enc(dirs[:,0,:], torch.zeros((batch_size,1),device='cuda:0'))
        color_spec = linear_to_srgb(self.color_network.outer_light(dir_enc))

        acc = torch.sum(weights, -1)
        if is_nerf:
            color = color + (1. - acc[..., None])

        outputs = {
            'ray_rgb': torch.clamp(color,min=0.0,max=1.0),  # rn,3
            'gradient_error': gradient_error,  # rn
            'acc': acc,  # rn
            'color_bkgr': color_bkgr,
            'color_spec':color_spec
        }

        if torch.sum(inner_mask) > 0:
            outputs['std'] = torch.mean(1 / inv_s)
        else:
            outputs['std'] = torch.zeros(1)

        if torch.sum(inner_mask) > 0:
            outputs['transmission'] = occ_info['transmission_weight']
            outputs['metallic'] = occ_info['metallic']
            
        if step < 1000:
            mask = torch.norm(points, dim=-1) < 1.2
            outputs['sdf_pts'] = points[mask]
            outputs['sdf_vals'] = self.sdf_network.sdf(points[mask])[..., 0]

        if self.cfg['apply_occ_loss']:
            # occlusion loss
            if torch.sum(inner_mask) > 0:
                outputs['loss_occ'] = self.compute_occ_loss(occ_info, points[inner_mask], sdf, gradients,
                                                            dirs[inner_mask], step)
            else:
                outputs['loss_occ'] = torch.zeros(1)

        if not is_train:
            outputs.update(self.compute_validation_info(z_vals, rays_o, rays_d, weights, human_poses, step))
            #print(list(outputs.keys()))
        return outputs

    def forward(self, data):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        is_train = 'eval' not in data
        # is_render = "render" in data  # novel view render
        step = data['step']

        if is_train:
            outputs = self.train_step(step)
        # elif is_render:
        #     outputs = self.render_step(step)
        else:
            index = data['index']
            outputs = self.test_step(index, step=step)

            if index == 0 and self.cfg['val_geometry']:
                bbox_min = -torch.ones(3)
                bbox_max = torch.ones(3)
                vertices, triangles = extract_geometry(bbox_min, bbox_max, 128, 0, lambda x: self.sdf_network.sdf(x))
                outputs['vertices'] = vertices
                outputs['triangles'] = triangles

        torch.set_default_tensor_type('torch.FloatTensor')
        return outputs

    def predict_materials(self):
        name = self.cfg['name']
        mesh = open3d.io.read_triangle_mesh(f'data/meshes/{name}-300000.ply')
        xyz = np.asarray(mesh.vertices)
        xyz = torch.from_numpy(xyz.astype(np.float32)).cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        metallic, roughness, albedo = [], [], []
        batch_size = 8192
        for vi in range(0, xyz.shape[0], batch_size):
            feature_vectors = self.sdf_network(xyz[vi:vi + batch_size])[:, 1:]
            m, r, a = self.color_network.predict_materials(xyz[vi:vi + batch_size], feature_vectors)
            metallic.append(m.cpu().numpy())
            roughness.append(r.cpu().numpy())
            albedo.append(a.cpu().numpy())

        return {'metallic': np.concatenate(metallic, 0),
                'roughness': np.concatenate(roughness, 0),
                'albedo': np.concatenate(albedo, 0)}



class Stage2Renderer(nn.Module):
    default_cfg = {
        # standard deviation for opacity density
        'std_net': 'default',
        'std_act': 'exp',
        'inv_s_init': 0.3,
        'freeze_inv_s_step': None,

        # geometry network
        'sdf_net': 'default',
        'sdf_activation': 'none',
        'sdf_bias': 0.5,
        'sdf_n_layers': 8,
        'sdf_freq': 6,
        'sdf_d_out': 257,
        'geometry_init': True,

        # shader network
        'shader_config': {},

        # sampling strategy
        'n_samples': 64,
        'n_bg_samples': 32,
        'inf_far': 1000.0,
        'n_importance': 64,
        'up_sample_steps': 4,  # 1 for simple coarse-to-fine sampling
        'perturb': 1.0,
        'anneal_end': 50000,
        'train_ray_num': 1024,
        'test_ray_num': 1024,
        'clip_sample_variance': True,

        # dataset
        'database_name': 'nerf_synthetic/lego/black_800',
        'is_nerf': False,

        # validation
        'test_downsample_ratio': True,
        'downsample_ratio': 0.5,
        'val_geometry': False,

        # losses
        'rgb_loss': 'charbonier',
        'apply_occ_loss': True,
        'occ_loss_step': 20000,
        'occ_loss_max_pn': 2048,
        'occ_sdf_thresh': 0.01,

        "fixed_camera": False,
    }

    def __init__(self, cfg, training=True):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        self.is_nerf = self.cfg['is_nerf']

      
        self.nerf_network = NeRFNetwork(D=8, d_in=4, d_in_view=3, W=256, multires=10, multires_view=4, output_ch=4,
                                      skips=[4], use_viewdirs=True)

        iors = torch.zeros(10,device='cuda:0')
        self.IORs = torch.nn.Parameter(iors,requires_grad=True)
        self.register_parameter(name="IORs", param=self.IORs)

        
        checkpoint = torch.load(cfg['stage1_ckpt_dir'])     
        cfg_stage1 = load_cfg(cfg['stage1_cfg_dir'])
        self.stage1_network = NeROShapeRenderer(cfg_stage1,training=False)
       # self.stage1_network.color_network.bkgr = self.stage1_network.infinity_far_bkgr
        self.stage1_network.load_state_dict(checkpoint['network_state_dict'], strict=False)
        self.stage1_network.cuda()
       # self.infinity_far_bkgr = InfOutNetwork()
        #self.infinity_far_bkgr = self.stage1_network.infinity_far_bkgr
        mesh_path = cfg['stage1_mesh_dir']
        # self.mesh = pymesh.meshio.load_mesh(mesh_path, drop_zero_dim=False)
        # self.mesh_separated = pymesh.separate_mesh(self.mesh)
        # print(self.mesh_separated)
        # self.bounding_boxes = []
        # for m in self.mesh_separated:
        #     self.bounding_boxes.append([np.min(m.vertices,axis=0),np.max(m.vertices,axis=0)])

        self.scene = Scene(mesh_path)
        self.IORs_pred = IoRNetwork()
        self.IoRint_pred = IoRNetwork()
        self.thickness_pred = ThicknessNetwork()
       # self.deviation_network = SingleVarianceNetwork(init_val=self.cfg['inv_s_init'], activation=self.cfg['std_act'])

        # background nerf is a nerf++ model (this is outside the unit bounding sphere, so we call it outer nerf)
        self.outer_nerf = NeRFNetwork(D=8, d_in=4, d_in_view=3, W=256, multires=10, multires_view=4, output_ch=4,
                                     skips=[4], use_viewdirs=True)
       # self.outer_nerf = self.stage1_network.outer_nerf
        #nn.init.constant_(self.outer_nerf.rgb_linear.bias, np.log(0.5))
        
    
        
        self.color_network = AppShadingNetwork_S2(self.cfg['shader_config'],self.stage1_network)

        
        self.sdf_network_inner = SDFNetwork(d_out=self.cfg['sdf_d_out'], d_in=3, d_hidden=256,
                                      n_layers=self.cfg['sdf_n_layers'],
                                      skip_in=[self.cfg['sdf_n_layers'] // 2], multires=self.cfg['sdf_freq'],
                                      bias=self.cfg['sdf_bias'], scale=1.0,
                                      geometric_init=self.cfg['geometry_init'],
                                      weight_norm=True, sdf_activation=self.cfg['sdf_activation'])

        self.deviation_network_inner = SingleVarianceNetwork(init_val=self.cfg['inv_s_init'], activation=self.cfg['std_act'])
        self.color_network_inner = AppShadingNetwork(self.cfg['shader_config'])
      #  self.sdf_inter_fun = lambda x: self.sdf_network.sdf(x)

        if training:
            self._init_dataset()

    def _init_dataset(self):
        # train/test split
        self.database = parse_database_name(self.cfg['database_name'], self.cfg['dataset_dir'])
        self.train_ids, self.test_ids = get_database_split(self.database)
        self.train_ids = np.asarray(self.train_ids)

        self.train_imgs_info = build_imgs_info(self.database, self.train_ids, self.is_nerf)
        self.train_imgs_info = imgs_info_to_torch(self.train_imgs_info, 'cpu')
        b, _, h, w = self.train_imgs_info['imgs'].shape
        print(f'training size {h} {w} ...')
        self.train_num = len(self.train_ids)

        self.test_imgs_info = build_imgs_info(self.database, self.test_ids, self.is_nerf)
        self.test_imgs_info = imgs_info_to_torch(self.test_imgs_info, 'cpu')
        self.test_num = len(self.test_ids)

        # clean the data if we already have
        if hasattr(self, 'train_batch'):
            del self.train_batch

        self.train_batch, self.train_poses, self.tbn, _, _ = self._construct_nerf_ray_batch(
            self.train_imgs_info) if self.is_nerf else self._construct_ray_batch(self.train_imgs_info)
        self.train_poses = self.train_poses.float().cuda()

        self._shuffle_train_batch()

    def _shuffle_train_batch(self):
        self.train_batch_i = 0
        shuffle_idxs = torch.randperm(self.tbn, device='cpu')  # shuffle
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[shuffle_idxs]

    def _construct_ray_batch(self, imgs_info, device='cpu'):
        imn, _, h, w = imgs_info['imgs'].shape
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.to(device)
        coords = coords.float()[None, :, :, :].repeat(imn, 1, 1, 1)  # imn,h,w,2
        coords = coords.reshape(imn, h * w, 2)
        coords = torch.cat([coords + 0.5, torch.ones(imn, h * w, 1, dtype=torch.float32, device=device)],
                           2)  # imn,h*w,3

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        dirs = coords @ torch.inverse(imgs_info['Ks']).permute(0, 2, 1)
        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn,3,4

        rn = imn * h * w
        ray_batch = {
            'dirs': dirs.float().reshape(rn, 3).to(device),
            'rgbs': imgs.float().reshape(rn, 3).to(device),
            'idxs': idxs.long().reshape(rn, 1).to(device),
        }
        return ray_batch, poses, rn, h, w

    def _construct_nerf_ray_batch(self, imgs_info, device='cpu', is_train=True):
        imn, _, h, w = imgs_info['imgs'].shape

        i, j = torch.meshgrid(torch.linspace(0, w - 1, w),
                              torch.linspace(0, h - 1, h))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()

        K = imgs_info['Ks'][0]
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn,3,4
        if is_train:
            masks = imgs_info['masks'].reshape(imn, h * w)

        rays_d = [torch.sum(dirs[..., None, :].cpu() * poses[i, :3, :3], -1) for i in range(imn)]
        rays_d = torch.stack(rays_d, 0).reshape(imn, h * w, 3)
        rays_o = [poses[i, :3, -1].expand(rays_d[0].shape) for i in range(imn)]
        rays_o = torch.stack(rays_o, 0).reshape(imn, h * w, 3)
        rn = imn * h * w
        ray_batch = {
            # 'dirs': dirs.float().reshape(rn, 3).to(device),
            'rgbs': imgs.float().reshape(rn, 3).to(device),
            'idxs': idxs.long().reshape(rn, 1).to(device),
            'rays_o': rays_o.float().reshape(rn, 3).to(device),
            'rays_d': rays_d.float().reshape(rn, 3).to(device),
        }
        if is_train:
            ray_batch['masks'] = masks.float().reshape(rn).to(device)
        return ray_batch, poses, rn, h, w

    # def _construct_render_batch(self, imgs_info, device='cpu'):
    #     imn, _, h, w = imgs_info['imgs'].shape
    #     coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
    #     coords = coords.to(device)
    #     coords = coords.float()
    #     coords = coords.reshape(h * w, 2)
    #     coords = torch.cat([coords + 0.5, torch.ones(h * w, 1, dtype=torch.float32, device=device)], 1)  # imn,h*w,3

    #     # h*w,3 @ 3,3 => imn,h*w,3
    #     dirs = coords @ torch.inverse(imgs_info['Ks'][0]).permute(1, 0)
    #     poses = self._construct_render_poses().float().to(device)
    #     pose_n = poses.shape[0]
    #     idxs = torch.arange(pose_n, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1

    #     rn = pose_n * h * w
    #     dirs = dirs.unsqueeze(0).repeat(pose_n, 1, 1)
    #     ray_batch = {
    #         'dirs': dirs.float().to(device),
    #         'idxs': idxs.long().to(device),
    #     }
    #     return ray_batch, poses, rn, h, w, pose_n

    def nvs(self, pose, K, h, w):
        device = 'cuda'
        K = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)
        pose = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(device)

        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.to(device)
        coords = coords.float()[None, :, :, :].repeat(1, 1, 1, 1)  # 1,h,w,2
        coords = coords.reshape(1, h * w, 2)
        coords = torch.cat([coords + 0.5, torch.ones(1, h * w, 1, dtype=torch.float32, device=device)], 2)  # 1,h*w,3

        # 1,h*w,3 @ imn,3,3 => 1,h*w,3
        dirs = coords @ torch.inverse(K).permute(0, 2, 1)
        idxs = torch.arange(1, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # 1,h*w,1

        rn = h * w
        ray_batch = {
            'dirs': dirs.float().reshape(rn, 3).to(device),
            'idxs': idxs.long().reshape(rn, 1).to(device),
        }

        trn = 1024
        output_color = []
        for ri in range(0, rn, trn):
            cur_ray_batch = {}
            for k, v in ray_batch.items(): cur_ray_batch[k] = v[ri:ri + trn]

            with torch.no_grad():
                rays_o, rays_d, near, far, human_poses = self._process_ray_batch(cur_ray_batch, pose)
                cur_outputs = self.render(rays_o, rays_d, near, far, human_poses, 0, 0, is_train=False, step=300000)
                output_color.append(cur_outputs['ray_rgb'].detach().cpu().numpy())

        output_color = np.reshape(np.concatenate(output_color, 0), [h, w, 3])
        return output_color

    def get_anneal_val(self, step):
        if self.cfg['anneal_end'] < 0:
            return 1.0
        else:
            return np.min([1.0, step / self.cfg['anneal_end']])

    @staticmethod
    def near_far_from_sphere(rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        near = torch.clamp(near, min=1e-3)
        return near, far

    def get_human_coordinate_poses(self, poses):
        pn = poses.shape[0]
        cam_cen = (-poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:])[..., 0]  # pn,3
        if self.cfg['fixed_camera']:
            pass
        else:
            cam_cen[..., 2] = 0

        Y = torch.zeros([1, 3]).expand(pn, 3)
        Y[:, 2] = -1.0
        Z = torch.clone(poses[:, 2, :3])  # pn, 3
        Z[:, 2] = 0
        Z = F.normalize(Z, dim=-1)
        X = torch.cross(Y, Z)  # pn, 3
        R = torch.stack([X, Y, Z], 1)  # pn,3,3
        t = -R @ cam_cen[:, :, None]  # pn,3,1
        return torch.cat([R, t], -1)

    def _process_ray_batch(self, ray_batch, poses):
        rays_d = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn

        rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]  # trn,3,1
        rays_o = rays_o[idxs, :, 0]  # rn,3
        rays_d = poses[idxs, :, :3].permute(0, 2, 1) @ rays_d.unsqueeze(-1)
        rays_d = rays_d[..., 0]  # rn,3

        rays_o = rays_o
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = self.near_far_from_sphere(rays_o, rays_d)

        human_poses = self.get_human_coordinate_poses(poses)
        return rays_o, rays_d, near, far, human_poses[idxs]  # rn, 3, 4

    def _process_nerf_ray_batch(self, ray_batch, poses):
        # dirs = ray_batch['dirs']  # rn,3
        idxs = ray_batch['idxs'][..., 0]  # rn
        rays_d = ray_batch['rays_d']
        rays_o = ray_batch['rays_o']
        #poses = poses[idxs, :3, :]
        # rays_d = torch.sum(dirs[..., None, :] * poses[..., :3], -1)
        # rays_o = poses[:, :, -1].expand(rays_d.shape)
        rays_d = F.normalize(rays_d, dim=-1)
        near, far = torch.full((rays_o.shape[0], 1),  0.3), torch.full((rays_o.shape[0], 1), 5.0)

        return rays_o, rays_d, near, far, poses[idxs]  # rn, 3, 4

    # def _process_render_ray_batch(self, ray_batch, poses):
    #     dirs = ray_batch['dirs']  # rn,3

    #     render_n = poses.shape[0]
    #     rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]
    #     rays_o = rays_o[:, :, 0]

    #     rays_d = []
    #     for i in range(render_n):
    #         rays_d_i = poses[i, :, :3][None, ...].permute(0, 2, 1) @ dirs.unsqueeze(-1)
    #         rays_d_i = rays_d_i[..., 0]  # rn,3
    #         rays_d.append(rays_d_i[None, ...])

    #     rays_d = torch.cat(rays_d, dim=0)

    #     rays_d = F.normalize(rays_d, dim=-1)
    #     near, far = self.near_far_from_sphere(rays_o, rays_d)

    #     human_poses = self.get_human_coordinate_poses(poses)
    #     return rays_o, rays_d, near, far, human_poses  # rn, 3, 4

    def test_step(self, index, step, ):
        target_imgs_info, target_img_ids = self.test_imgs_info, self.test_ids
        imgs_info = imgs_info_slice(target_imgs_info, torch.from_numpy(np.asarray([index], np.int64)))
        gt_depth, gt_mask = self.database.get_depth(target_img_ids[index])  # used in evaluation
        is_nerf = self.is_nerf
        if self.cfg['test_downsample_ratio']:
            imgs_info = imgs_info_downsample(imgs_info, self.cfg['downsample_ratio'])
            h, w = gt_depth.shape
            dh, dw = int(self.cfg['downsample_ratio'] * h), int(self.cfg['downsample_ratio'] * w)
            gt_depth, gt_mask = cv2.resize(gt_depth, (dw, dh), interpolation=cv2.INTER_NEAREST), \
                cv2.resize(gt_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
        gt_depth, gt_mask = torch.from_numpy(gt_depth), torch.from_numpy(gt_mask.astype(np.int32))
        ray_batch, input_poses, rn, h, w = self._construct_nerf_ray_batch(imgs_info, is_train=False) \
            if is_nerf else self._construct_ray_batch(imgs_info)

        input_poses = input_poses.float().cuda()
        for k, v in ray_batch.items(): ray_batch[k] = v.cuda()

        trn = self.cfg['test_ray_num']
        outputs_keys = ['ray_rgb', 'gradient_error', 'normal','tir_mask']
        outputs_keys += [ 'specular_light', 'specular_color', 'specular_ref'
            #'diffuse_albedo', 'diffuse_light', 'diffuse_color','refraction_light',
           # 'specular_albedo',
           # 'metallic', 'roughness', 'occ_prob', 'indirect_light', 'occ_prob_gt',
        ]
        if self.color_network.cfg['human_light']:
            outputs_keys += ['human_light']

        outputs = {k: [] for k in outputs_keys}
        for ri in range(0, rn, trn):
            cur_ray_batch = {k: v[ri:ri + trn] for k, v in ray_batch.items()}
            rays_o, rays_d, near, far, human_poses = self._process_nerf_ray_batch(cur_ray_batch, input_poses) \
                if is_nerf else self._process_ray_batch(cur_ray_batch, input_poses)
            cur_outputs = self.render(rays_o, rays_d, near, far, human_poses, 0, 0, is_train=False, step=step,
                                      is_nerf=is_nerf)
            for k in outputs_keys: outputs[k].append(cur_outputs[k].detach())

        for k in outputs_keys: outputs[k] = torch.cat(outputs[k], 0)
        outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb']* outputs['tir_mask'].detach(), ray_batch['rgbs']* outputs['tir_mask'].detach())
        outputs['gt_rgb'] = (ray_batch['rgbs'] *  outputs['tir_mask'].detach()).reshape(h, w, 3)
        outputs['ray_rgb'] = (outputs['ray_rgb'] * outputs['tir_mask'].detach()).reshape(h, w, 3)

        # used in evaluation
        outputs['gt_depth'] = gt_depth.unsqueeze(-1)
        outputs['gt_mask'] = gt_mask.unsqueeze(-1)

        self.zero_grad()
       # exit(1)
        return outputs

    def train_step(self, step):
        rn = self.cfg['train_ray_num']
        is_nerf = self.is_nerf
        # fetch to gpu
        train_ray_batch = {k: v[self.train_batch_i:self.train_batch_i + rn].cuda() for k, v in self.train_batch.items()}
        self.train_batch_i += rn
        if self.train_batch_i + rn >= self.tbn: self._shuffle_train_batch()
        train_poses = self.train_poses.cuda()
        rays_o, rays_d, near, far, human_poses = self._process_nerf_ray_batch(train_ray_batch, train_poses) \
            if is_nerf else self._process_ray_batch(train_ray_batch, train_poses)

        outputs = self.render(rays_o, rays_d, near, far, human_poses, -1, self.get_anneal_val(step), is_train=True,
                              step=step, is_nerf=is_nerf)
        outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'] * outputs['tir_mask'].detach(), train_ray_batch['rgbs']* outputs['tir_mask'].detach()) # ray_loss  # ray_loss
       # if is_nerf:  # only nerf dataset add loss_mask
        #    outputs['loss_mask'] = F.l1_loss(train_ray_batch['masks'], outputs['acc'], reduction='mean')
        return outputs

    # def render_step(self, step):
    #     self.eval()

    #     render_batch, render_poses, rn, h, w, pose_n = self._construct_render_batch(self.train_imgs_info)
    #     render_poses = render_poses.float().cuda()
    #     for k, v in render_batch.items(): render_batch[k] = v.cuda()
    #     trn = self.cfg['test_ray_num']

    #     outputs_keys = ['ray_rgb', 'gradient_error', 'normal', 'depth']
    #     outputs_keys += [
    #         'diffuse_albedo', 'diffuse_light', 'diffuse_color',
    #         'specular_albedo', 'specular_light', 'specular_color', 'specular_ref',
    #         'metallic', 'roughness', 'occ_prob', 'indirect_light', 'occ_prob_gt',
    #     ]

    #     final_outputs = []
    #     for pose in range(pose_n):
    #         outputs = {k: [] for k in outputs_keys}
    #         for ri in trange(0, h * w, trn):
    #             cur_ray_batch = {k: v[pose, ri:ri + trn, :] for k, v in render_batch.items()}
    #             rays_o, rays_d, near, far, human_poses = self._process_ray_batch(cur_ray_batch, render_poses)
    #             cur_outputs = self.render(rays_o, rays_d, near, far, human_poses, 0, 0, is_train=False, step=step)
    #             for k in outputs_keys: outputs[k].append(cur_outputs[k].detach())

    #         for k in outputs_keys: outputs[k] = torch.cat(outputs[k], 0)
    #         outputs['ray_rgb'] = outputs['ray_rgb'].reshape(h, w, 3)
    #         import imageio
    #         imageio.imwrite("test_{}.png".format(pose), outputs['ray_rgb'].cpu())
    #         final_outputs.append(outputs)

    #     self.train()
    #     return outputs

    def compute_rgb_loss(self, rgb_pr, rgb_gt):
        if self.cfg['rgb_loss'] == 'l2':
            rgb_loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)
        elif self.cfg['rgb_loss'] == 'l1':
            rgb_loss = torch.sum(F.l1_loss(rgb_pr, rgb_gt, reduction='none'), -1)
        elif self.cfg['rgb_loss'] == 'smooth_l1':
            rgb_loss = torch.sum(F.smooth_l1_loss(rgb_pr, rgb_gt, reduction='none', beta=0.25), -1)
        elif self.cfg['rgb_loss'] == 'charbonier':
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        else:
            raise NotImplementedError
        return rgb_loss

    def density_activation(self, density, dists):
        return 1.0 - torch.exp(-F.softplus(density) * dists)

    def compute_density(self, points):
        points_norm = torch.norm(points, dim=-1, keepdim=True)
        points_norm = torch.clamp(points_norm, min=1e-3)
        sigma = self.stage1_network.outer_nerf.density(torch.cat([points / points_norm, 1.0 / points_norm], -1))[..., 0]
        return sigma

    @staticmethod
    def upsample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
       
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    @staticmethod
    def upsample_nerf(z_vals, weights, n_importance):
        """
        Up sampling give a fixed inv_s
        """
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network_inner.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
    
    def cat_z_vals_nerf(self,  z_vals, new_z_vals):
    
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

      
        return z_vals
    
    

    def sample_ray(self, rays_o, rays_d, near, far, perturb):
        n_samples = self.cfg['n_samples']
        n_bg_samples = self.cfg['n_bg_samples']
        n_importance = self.cfg['n_importance']
        up_sample_steps = self.cfg['up_sample_steps']

        # sample points
        batch_size = len(rays_o)
        z_vals = torch.linspace(0.0, 1.0, n_samples)  # sn
        z_vals = near + (far - near) * z_vals[None, :]  # rn,sn
        z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_bg_samples + 1.0), n_bg_samples)

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples

            mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
            upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
            lower = torch.cat([z_vals_outside[..., :1], mids], -1)
            t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
            z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / n_bg_samples

        # # Up sample
        # with torch.no_grad():
        #     pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        #     sdf = self.sdf_network.sdf(pts).reshape(batch_size, n_samples)

        #     for i in range(up_sample_steps):
        #         rn, sn = z_vals.shape
        #         if self.cfg['clip_sample_variance']:
        #             inv_s = self.deviation_network(torch.empty([1, 3])).expand(rn, sn - 1)
        #             inv_s = torch.clamp(inv_s, max=64 * 2 ** i)  # prevent too large inv_s
        #         else:
        #             inv_s = torch.ones(rn, sn - 1) * 64 * 2 ** i
        #         new_z_vals = self.upsample(rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s)
        #         z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

        z_vals = torch.cat([z_vals, z_vals_outside], -1)
        return z_vals

    def render(self, rays_o, rays_d, near, far, human_poses, perturb_overwrite=-1, cos_anneal_ratio=0.0, is_train=True,
               step=None, is_nerf=False):
        """
        :param rays_o: rn,3
        :param rays_d: rn,3
        :param near:   rn,1
        :param far:    rn,1
        :param human_poses:     rn,3,4
        :param perturb_overwrite: set 0 for inference
        :param cos_anneal_ratio:
        :param is_train:
        :param step:
        :return:
        """
        
        pathes, converges, directions,ior_ratios,infinity_bkgr,gradient_mesh,tir_mask = self.ray_trace(rays_o,rays_d)
      
        #perturb = self.cfg['perturb']
       # if perturb_overwrite >= 0:
       #     perturb = perturb_overwrite
       # z_vals = self.sample_ray(rays_o, rays_d, near, far, perturb)
        ret = self.render_core(rays_o, rays_d, pathes, converges, directions,infinity_bkgr,gradient_mesh,ior_ratios, human_poses, cos_anneal_ratio=cos_anneal_ratio, step=step,
                               is_train=is_train, is_nerf=is_nerf)
        ret['tir_mask'] = tir_mask
        return ret

    def compute_validation_info(self, rays_o, rays_d, weights, human_poses, step):
        depth = torch.sum(weights * z_vals, -1, keepdim=True)  # rn,
        points = depth * rays_d + rays_o  # rn,3
        gradients = self.nerf_network.gradient(points)  # rn,3
        inner_mask = torch.norm(points, dim=-1, keepdim=True) <= 1.0
        outputs = {
            'depth': depth,  # rn,1
            'normal': ((F.normalize(gradients, dim=-1) + 1.0) * 0.5) * inner_mask,  # rn,3
        }

        feature_vector = self.nerf_network(points)[2]  # rn,f
        _, occ_info, inter_results = self.color_network(points, gradients, -F.normalize(rays_d, dim=-1), feature_vector,
                                                        human_poses, inter_results=True, step=step)
        _, occ_prob, _ = get_intersection(self.sdf_inter_fun, self.deviation_network, points, occ_info['reflective'],
                                          sn0=128, sn1=9)  # pn,sn-1
        occ_prob_gt = torch.sum(occ_prob, dim=-1, keepdim=True)
        outputs['occ_prob_gt'] = occ_prob_gt
        for k, v in inter_results.items(): inter_results[k] = v * inner_mask
        outputs.update(inter_results)
        return outputs

    def compute_sdf_alpha(self, points, dists, dirs, cos_anneal_ratio, step):
        # points [...,3] dists [...] dirs[...,3]
        sdf_nn_output = self.sdf_network_inner(points)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]

        gradients = self.sdf_network_inner.gradient(points)  # ...,3
        inv_s = self.deviation_network_inner(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        if self.cfg['freeze_inv_s_step'] is not None and step < self.cfg['freeze_inv_s_step']:
            inv_s = inv_s.detach()

        true_cos = (dirs * gradients).sum(-1)  # [...]
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # [...]
        return alpha, gradients, feature_vector, inv_s, sdf
    
    def compute_sdf(self, points):
        # points [...,3] dists [...] dirs[...,3]
        sdf_nn_output = self.stage1_network.sdf_network(points)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]

        gradients = self.stage1_network.sdf_network.gradient(points)  # ...,3
        inv_s = self.stage1_network.deviation_network(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        return  gradients, feature_vector, inv_s, sdf

    def compute_density_alpha(self, points, dists, dirs, nerf):
        norm = torch.norm(points, dim=-1, keepdim=True)
        points = torch.cat([points / norm, 1.0 / norm], -1)
        density, color = nerf(points, dirs)  # [...,1] [...,3]
      #  print(F.softplus(density[..., 0]))
        alpha = self.density_activation(density[..., 0], dists)
        color = linear_to_srgb(torch.exp(torch.clamp(color, max=5.0)))
       # color = torch.exp(torch.clamp(color, max=5.0)) 
        return alpha, color

    def compute_occ_loss(self, occ_info, points, sdf, gradients, dirs, step):
        if step < self.cfg['occ_loss_step']: return torch.zeros(1)

        occ_prob = occ_info['occ_prob']
        reflective = occ_info['reflective']

        # select a subset for occ loss
        # note we only apply occ loss on the surface
        inner_mask = torch.norm(points, dim=-1) < 0.999  # pn
        sdf_mask = torch.abs(sdf) < self.cfg['occ_sdf_thresh']
        normal_mask = torch.sum(gradients * dirs, -1) < 0  # pn
        mask = (inner_mask & normal_mask & sdf_mask)

        if torch.sum(mask) > self.cfg['occ_loss_max_pn']:
            indices = torch.nonzero(mask)[:, 0]  # npn
            idx = torch.randperm(indices.shape[0], device='cuda')  # npn
            indices = indices[idx[:self.cfg['occ_loss_max_pn']]]  # max_pn
            mask_new = torch.zeros_like(mask)
            mask_new[indices] = 1
            mask = mask_new

        if torch.sum(mask) > 0:
            inter_dist, inter_prob, inter_sdf = get_intersection(self.sdf_inter_fun, self.deviation_network,
                                                                 points[mask], reflective[mask], sn0=64,
                                                                 sn1=16)  # pn,sn-1
            occ_prob_gt = torch.sum(inter_prob, -1, keepdim=True)
            return F.l1_loss(occ_prob[mask], occ_prob_gt)
        else:
            return torch.zeros(1)

    def ray_trace(self,rays_o,rays_d):
        #print(rays_o.shape)
        #exit(1)
        vert = rays_o
        # vert = rays_o.reshape(-1,1,3) + rays_d.reshape(-1,1,3) * torch.linspace(0, 1, 64).reshape(1,64,1).cuda()
        # pc = trimesh.PointCloud(vertices = vert.reshape(-1,3).detach().cpu().numpy())
        # trimesh.exchange.export.export_mesh(pc,'tt.ply')
        # exit(1)
        is_outside = True
        #outside_tracing_sdf = lambda x: self.stage1_network.sdf_network.sdf(x)
       # inside_tracing_sdf = lambda x: -self.stage1_network.sdf_network.sdf(x)
        tirs = []
        next_start = rays_o
        next_dir = rays_d
        starts = [rays_o]
        directions = [rays_d]
        intersections = []
        converges = []
        infinity_bkgr = []
        ior_ratios = []
        converges_orig = []
        converged = torch.ones((rays_o.shape[0],1)).cuda().bool()
        gradient_mesh = []
        for i in range(3):
           # print(next_start)
           # print(next_dir)
           # next_start = torch.nan_to_num(next_start, 0.0)
          #  next_dir = F.normalize(torch.nan_to_num(next_dir, 1.0),dim=-1)
           
            tir = torch.ones((next_start.shape[0],1)).cuda().bool().detach()
            rays_for_intersection = Ray(next_start,next_dir)
            if is_outside:
                intersection_info, converged = self.scene.Dintersect(rays_for_intersection)
                converged =converged.bool().reshape(-1,1).detach()
                intersection = intersection_info.intersection_point

                normal = F.normalize(intersection_info.n, dim=-1)
            else:
                intersection_info, converged = self.scene.Dintersect(rays_for_intersection)
                converged =converged.bool().reshape(-1,1).detach()
                intersection = intersection_info.intersection_point
                #if i == 1:
                #    print(intersection)
               #     print(converged)
                normal = -F.normalize(intersection_info.n, dim=-1)
            #print(converged.sum())
            # if i == 0:
            #     pc = trimesh.PointCloud(vertices = intersection.reshape(-1,3).detach().cpu().numpy())
            #     trimesh.exchange.export.export_mesh(pc,'int0.ply')
            # if i == 1:
            #     pc = trimesh.PointCloud(vertices = intersection.reshape(-1,3).detach().cpu().numpy())
            #     trimesh.exchange.export.export_mesh(pc,'int1.ply')
            # if i == 2:
            #     pc = trimesh.PointCloud(vertices = intersection.reshape(-1,3).detach().cpu().numpy())
            #     trimesh.exchange.export.export_mesh(pc,'int2.ply')
            gradient_mesh_tmp = normal
            infinity_bkgr.append(~converged)
            mask_converge = converged.clone().flatten()
            #for i in range(converged.shape[0]):
            #    if converged[i]: vert = torch.cat([vert,intersection[i].reshape(-1,3)],dim=0)
            #print(intersection)
            
            cos_thetai = torch.sum(normal * -next_dir[converged.flatten()],dim=-1,keepdim=True)
            # if cos_thetai.shape[0] > 0:
            #     print(cos_thetai.min())
            #     if cos_thetai.min() < 0:
            #         print(i)
            sin_thetai_2 = 1 - (cos_thetai * cos_thetai)

          
        
            ior_ratio = self.IORs_pred(intersection.reshape(-1,3)).reshape(-1,1)
            ior_ratio =  1 / (ior_ratio * 1.0 + 1) 
            #ior_ratio = 1 / (ior_ratio * 0.8 + 1) 



           
         #   ior_ratio = torch.where(bounding_box_index == 9, 1, 1/1.5).reshape(-1,1)
            
          #  print(ior_ratio)
          #  print(ior_ratio * ior_ratio * sin_thetai_2)
            if not is_outside: 
                ior_ratio = 1 / ior_ratio
            
         #   print(ior_ratio * ior_ratio * sin_thetai_2)
           # print(converged[mask_converge])
           # print(converged.sum())
           # converged_out
            converged_out = converged.clone()
            converged_out[mask_converge] = torch.where(ior_ratio * ior_ratio * sin_thetai_2 > 0.999, False, True)
            tir[mask_converge] = torch.where(ior_ratio * ior_ratio * sin_thetai_2 > 0.999, False, True).detach()
            tirs.append(tir.detach())
           # converged[ior_ratio * ior_ratio * sin_thetai_2 > 1] *= 0
           # print(converged[mask_converge])
           # print(converged.sum())
            
            
            ior_ratio = ior_ratio[converged_out[mask_converge].flatten()]
            sin_thetat_2 =sin_thetai_2[converged_out[mask_converge].flatten()] * ior_ratio * ior_ratio

            #print(ior_ratio.shape)
           # print(next_dir.shape)
           # print(cos_thetai.shape)
         #   print(ior_ratio)
         #   print(next_dir)
         #   print(cos_thetai)
          #  print(sin_thetat_2)
         #   print(converged_out.flatten())
            next_dir_tmp = (ior_ratio * next_dir[converged_out.flatten()] + (ior_ratio * cos_thetai[converged_out[mask_converge].flatten()] - torch.sqrt(1 - sin_thetat_2)) * normal[converged_out[mask_converge].flatten()])

            next_start_tmp = intersection[converged_out[mask_converge].flatten()] + next_dir_tmp * 1e-5
          #  print(torch.linalg.norm(next_dir_tmp,dim=-1))
            next_dir_tmp = next_dir_tmp / (torch.linalg.norm(next_dir_tmp,dim=-1,keepdim=True) + 0.0001)
          #  print(next_dir_tmp)
          #  next_dir_tmp[~converged.flatten()] = 1
        #    print(next_dir)
          #  print(next_dir_tmp)
           # exit(1)
          #  next_dir_tmp = next_dir_tmp[converged_out[mask_converge].flatten()]
          #  next_start_tmp = next_start_tmp[converged_out[mask_converge].flatten()]
          #  ior_ratio = ior_ratio[converged_out[mask_converge].flatten()]
            gradient_mesh_tmp = gradient_mesh_tmp[converged_out[mask_converge].flatten()]
            intersection = intersection#[converged[mask_converge].flatten()]
            next_dir = next_dir_tmp
            next_start = next_start_tmp
            
            

            directions.append(next_dir)
            starts.append(next_start)
            converges.append(converged_out)
            converges_orig.append(converged)
            intersections.append(intersection)
            if torch.all(~converged_out): break
            gradient_mesh.append(gradient_mesh_tmp)
            
            ior_ratios.append(ior_ratio)
            is_outside = not is_outside
           # print(ior_ratio)
        for i in range(len(tirs) - 1,0,-1):
            #print(i)
            tirs[i - 1][converges[i-1].flatten()] *= tirs[i]
       
        converged_tmp = torch.ones((rays_o.shape[0],1),device='cuda:0').bool()
        
          # rn,sn
        sampled_pathes = []
        for k in range(len(converges)):
            
          #  print(converged_tmp.flatten().shape)
              # sn
            start = starts[k].reshape(-1,3)
            #print(start.shape)
           # print(directions[i].shape)
            #print(intersections[i][converged_tmp.flatten()].shape)
            end = start + directions[k] * 4.5
            if k != 1:
                z_vals = torch.linspace(0, 1, 256).reshape(1,-1)
                sampled_vertices = start.reshape(-1,1,3) + (end.reshape(-1,1,3) - start.reshape(-1,1,3)) * z_vals.reshape(-1,256,1) 
            else:
                z_vals_bkgr = torch.linspace(0, 1, 128).reshape(1,-1)
                z_vals_nobkgr = torch.linspace(0, 1, 64).reshape(1,-1)
                sampled_vertices = start.reshape(-1,1,3) + (end.reshape(-1,1,3) - start.reshape(-1,1,3)) * z_vals_bkgr.reshape(-1,128,1) 
           # print(k)
            if torch.any(~infinity_bkgr[k]):
                end[~infinity_bkgr[k].flatten()] = intersections[k]
                
                if k != 1:
                    sampled_vertices[~infinity_bkgr[k].flatten()] = start[~infinity_bkgr[k].flatten()].reshape(-1,1,3) + (end[~infinity_bkgr[k].flatten()].reshape(-1,1,3) - start[~infinity_bkgr[k].flatten()].reshape(-1,1,3)) * z_vals.reshape(-1,256,1)
                else:
                    upsample_tmp = start[~infinity_bkgr[k].flatten()].reshape(-1,1,3) + (end[~infinity_bkgr[k].flatten()].reshape(-1,1,3) - start[~infinity_bkgr[k].flatten()].reshape(-1,1,3)) * z_vals_nobkgr.reshape(-1,64,1)

                    with torch.no_grad():
                        pts = upsample_tmp
                        z_vals_nobkgr= z_vals_nobkgr.reshape(-1,64).expand(pts.shape[0],64)
                        sdf = self.sdf_network_inner.sdf(pts).reshape(-1,64)
                        rays_o_tmp = start[~infinity_bkgr[k].flatten()].reshape(-1,3)
                        rays_d_tmp = directions[k][~infinity_bkgr[k].flatten()].reshape(-1,3)
                        for i in range(2):
                            rn, sn = z_vals_nobkgr.shape
                            if self.cfg['clip_sample_variance']:
                                inv_s = self.deviation_network_inner(torch.empty([1, 3])).expand(rn, sn - 1)
                                inv_s = torch.clamp(inv_s, max=64 * 2 ** i)  # prevent too large inv_s
                            else:
                                inv_s = torch.ones(rn, sn - 1) * 64 * 2 ** i
                            new_z_vals = self.upsample(rays_o_tmp, rays_d_tmp, z_vals_nobkgr, sdf, 64 // 2, inv_s)
                            z_vals_nobkgr, sdf = self.cat_z_vals(rays_o_tmp, rays_d_tmp, z_vals_nobkgr, new_z_vals, sdf, last=(i + 1 == 2))

                    sampled_vertices[~infinity_bkgr[k].flatten()] = start[~infinity_bkgr[k].flatten()].reshape(-1,1,3) + (end[~infinity_bkgr[k].flatten()].reshape(-1,1,3) - start[~infinity_bkgr[k].flatten()].reshape(-1,1,3)) * z_vals_nobkgr.reshape(-1,64+32+32,1)

            if not torch.all(~infinity_bkgr[k]) and k != 1:
                #z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (64 + 1.0), 64)
                if k == 0:
                    z_vals_outside = torch.linspace(0.1, 64.0, 192)
                else:
                    z_vals_outside = torch.linspace(0.1, 64.0, 192)
                bkgr_size = torch.sum((infinity_bkgr[k]))

                upsample_out_tmp = start[infinity_bkgr[k].flatten()].reshape(-1,1,3) + directions[k][infinity_bkgr[k].flatten()].reshape(-1,1,3) * z_vals_outside.reshape(-1,192,1)
                # t_rand = (torch.rand([bkgr_size, 1]) - 0.5)
                # z_vals = z_vals + t_rand * 2.0 / 64
                # z_vals_outside = z_vals_outside.reshape(1,-1).expand(bkgr_size,64)
                # mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                # upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                # lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                # t_rand = torch.rand([bkgr_size, z_vals_outside.shape[-1]])
                # z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand
#4.5
                #z_vals_outside = 4.5 / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / 64

                #print(z_vals_outside)
                dirs = directions[k][infinity_bkgr[k].flatten()].reshape(-1,1,3)
                with torch.no_grad():
                    pts = upsample_out_tmp
                    z_vals_outside = z_vals_outside.reshape(-1,192).expand(pts.shape[0],192)
                    dists = z_vals_outside[..., 1:] - z_vals_outside[..., :-1]
                    dists = torch.cat([dists, dists[..., -1:]], -1)
                    alpha ,color = self.compute_density_alpha(pts, dists, -dirs.expand(dirs.shape[0],192,3), self.stage1_network.outer_nerf)
                    weights = alpha * torch.cumprod( torch.cat([torch.ones([pts.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
                    #print(weights.shape)
                    #print(z_vals_outside.shape)
                    rays_o_tmp = start[infinity_bkgr[k].flatten()].reshape(-1,3)
                    rays_d_tmp = directions[k][infinity_bkgr[k].flatten()].reshape(-1,3)
                    for i in range(1):
                        new_z_vals = self.upsample_nerf(z_vals_outside, weights[:,:-1], 64)
                        z_vals_outside = self.cat_z_vals_nerf(z_vals_outside, new_z_vals)

                sampled_vertices[infinity_bkgr[k].flatten()] = start[infinity_bkgr[k].flatten()].reshape(-1,1,3) + directions[k][infinity_bkgr[k].flatten()].reshape(-1,1,3) * z_vals_outside.reshape(-1,192+64,1)
            #print(k)
            #if k == 1: 
              # vert = torch.cat([vert,sampled_vertices.reshape(-1,3)],dim=0)
            #    vert = sampled_vertices.reshape(-1,3)
            sampled_pathes.append(sampled_vertices)
            #print(z_vals_tmp[~converges[k][converged_tmp.flatten()].flatten()])
            #print(converges[k])
            #print(z_vals_tmp)
           # print(start.shape)
           # print(end.shape)
           # print(((end - start) * z_vals.reshape(1,64,1)).shape)
            
           # vert = torch.cat([vert,sampled_vertices.reshape(-1,3)],dim=0)
            converged_tmp = converges[k]
            # if i == 1:
            #     print(start)
            #     print(end)
            #     print(intersections[i])
      #  print(len(sampled_pathes)
            
       # pc = trimesh.PointCloud(vertices = vert.detach().cpu().numpy())
       # trimesh.exchange.export.export_mesh(pc,'tst_pc_optix1.ply')
      #  exit(1)
      #  print(sampled_pathes, converges, directions)
       # exit(1)
        #print(self.IORs)
       # print(directions)
       # exit(1)
        return sampled_pathes, converges, directions, ior_ratios, infinity_bkgr,gradient_mesh,tirs[0]






    def render_core(self, rays_o, rays_d, pathes, converges, directions, infinity_bkgr,gradient_mesh,ior_ratios,human_poses, cos_anneal_ratio=0.0, step=None, is_train=True,
                    is_nerf=False):
        

        current_transmission_portion = torch.ones((converges[0].shape[0],3),device='cuda:0').reshape(-1,3)
        normals_output = torch.zeros((converges[0].shape[0],3),device='cuda:0').reshape(-1,3)
        specular_color_output = torch.zeros((converges[0].shape[0],3),device='cuda:0').reshape(-1,3)
        specular_light_output = torch.zeros((converges[0].shape[0],3),device='cuda:0').reshape(-1,3)
        specular_ref_output = torch.zeros((converges[0].shape[0],3),device='cuda:0').reshape(-1,3)
        colors = []
        outputs = {}
        outputs_tmp = {}
       # print(len(pathes))
        for i in range(len(pathes)):
            
         #   print(current_transmission_portion)
            #[N,M,3]
          #  print(i)
            candidate_points = pathes[i]
            color_now = torch.zeros((converges[i].shape[0],3),device='cuda:0')
            bkgr_mask = infinity_bkgr[i]
           # print(candidate_points.shape[0])
            batch_size = candidate_points.shape[0]
            #[N,3]
            candidate_directions = directions[i]
            gradient_error = torch.zeros(1)
            #[N,1]
            candidate_converges = converges[i]
           # print(converges[i])
            actual_intersection_pathes = candidate_points[candidate_converges.flatten()]

            points_for_nerf = candidate_points[:,:-1,:]
            dists = points_for_nerf[:,1:] -points_for_nerf[:,:-1]
            dists = torch.linalg.norm(dists,dim=-1)
            dists = torch.cat([dists, dists[..., -1:]], -1)
            n_samples = points_for_nerf.shape[1]
            points_for_neus = actual_intersection_pathes[:,-1:,:]

            inner_mask = torch.norm(points_for_nerf, dim=-1) <= 1.0
            outer_mask = ~inner_mask
            
            dirs = candidate_directions.reshape(-1,1,3).expand(batch_size, n_samples, 3)
            dir_neus = candidate_directions[candidate_converges.flatten()].reshape(-1,1,3)


            alpha_nerf, sampled_color_nerf = torch.zeros(batch_size, n_samples), torch.zeros(batch_size, n_samples, 3)

            alpha_nerf[outer_mask], sampled_color_nerf[outer_mask] = self.compute_density_alpha(points_for_nerf[outer_mask],
                                                                                        dists[outer_mask],
                                                                                        -dirs[outer_mask],
                                                                                        self.stage1_network.outer_nerf)
            # if i == 0:
            #     pc = trimesh.PointCloud(vertices = points_for_nerf.reshape(-1,3).detach().cpu().numpy())
            #     trimesh.exchange.export.export_mesh(pc,'outer0.ply')

            #     pc = trimesh.PointCloud(vertices = points_for_nerf[:,0].reshape(-1,3).detach().cpu().numpy())
            #     trimesh.exchange.export.export_mesh(pc,'rayo.ply')
            # if i == 2:
            #     pc = trimesh.PointCloud(vertices = points_for_nerf.reshape(-1,3).detach().cpu().numpy())
            #     trimesh.exchange.export.export_mesh(pc,'outer2.ply')

            if i == 1:
                alpha_nerf[inner_mask], gradients_inner, feature_vector, inv_s_inner, sdf = self.compute_sdf_alpha(points_for_nerf[inner_mask],
                                                                                                dists[inner_mask],
                                                                                                dirs[inner_mask],
                                                                                                cos_anneal_ratio, step)
                
                human_poses_pre = torch.zeros((batch_size, n_samples, 3, 4),device='cuda:0')
                # print(points_for_nerf.shape)
                # print(gradients.shape)
                # print(dirs.shape)
                # print(feature_vector.shape)
             #   print(outer_mask.sum())
                # print(points_for_nerf[outer_mask])
                # pc = trimesh.PointCloud(vertices = points_for_nerf.reshape(-1,3).detach().cpu().numpy())
                # trimesh.exchange.export.export_mesh(pc,'outerx.ply')
                #exit(1)

                sampled_color_nerf[inner_mask], occ_info = self.color_network_inner(points_for_nerf[inner_mask], gradients_inner, -dirs[inner_mask],
                                                                        feature_vector,human_poses_pre[inner_mask],
                                                                        step=step)

               # alpha_nerf = alpha_nerf.reshape(batch_size, n_samples)
               # sampled_color_nerf = sampled_color_nerf.reshape(batch_size, n_samples,3)

                outputs_tmp['std'] = torch.mean(1 / inv_s_inner)
                outputs_tmp['gradient_error'] = (torch.linalg.norm(gradients_inner, ord=2, dim=-1) - 1.0) ** 2
            
            if points_for_neus.nelement() > 0:
                # print('111')
                gradients, feature_vector, inv_s, sdf = self.compute_sdf(points_for_neus)
                
                    
            # print(points_for_neus.shape)
            #print(feature_vector.shape)
                if i % 2 != 0: gradients = -gradients
               # print(points_for_neus.shape)
               # print(directions[i+1].shape)
                sampled_color_sdf, occ_info = self.color_network(points_for_neus, gradient_mesh[i].reshape(-1,1,3), -dir_neus,
                                                                    feature_vector,ior_ratios[i].reshape(-1,1,1), i % 2 != 0,directions[i+1].reshape(-1,1,3), #, human_poses_pt,
                                                                    step=step)
                if i == 0 and not is_train:      
                    sampled_color_sdf, occ_info,intermediate_results = self.color_network(points_for_neus, gradient_mesh[i].reshape(-1,1,3), -dir_neus,
                                                                    feature_vector,ior_ratios[i].reshape(-1,1,1), i % 2 != 0,directions[i+1],inter_results=True,
                                                                    step=step)
                    normals_output[converges[i].flatten()] = (F.normalize(gradient_mesh[i].reshape(-1,3), dim=-1) + 1.0) * 0.5
                    specular_color_output[converges[i].flatten()] = intermediate_results['specular_color'].reshape(-1,3)
                    specular_light_output[converges[i].flatten()] = intermediate_results['specular_light'].reshape(-1,3)
                    specular_ref_output[converges[i].flatten()] = intermediate_results['specular_ref'].reshape(-1,3)

           #
           # sampled_color_nerf = srgb_to_linear(sampled_color_nerf)
            weights_nerf = alpha_nerf * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha_nerf + 1e-7], -1), -1)[:, :-1]
            color_nerf = (sampled_color_nerf * weights_nerf[..., None]).sum(dim=1)
            acc_nerf = torch.sum(weights_nerf, -1)
          #  print(converges[i])
          #  print(color_nerf)
            color_now += color_nerf * current_transmission_portion
            transmission = torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha_nerf + 1e-7], -1), -1)[:, -1:]
           # print(transmission.shape)
            current_transmission_portion = current_transmission_portion * transmission
            #color_now[bkgr_mask.flatten()] += bkgr_color[bkgr_mask.flatten()] * current_transmission_portion[bkgr_mask.flatten()]

           
            if points_for_neus.nelement() > 0:
                color_neus = sampled_color_sdf[:,0,:]
               
                color_now[converges[i].flatten()] += color_neus * current_transmission_portion[converges[i].flatten()]
               
                current_transmission_portion = current_transmission_portion[converges[i].flatten()] * occ_info['refraction_coefficient'][:,0,:]
                colors.append(color_now)
            else: 
                colors.append(color_now)
                break
        
        for i in range(len(colors) - 1,0,-1):
            #print(i)
            colors[i - 1][converges[i-1].flatten()] += colors[i]
        ray_rgb = colors[0]
        #ray_rgb = linear_to_srgb(ray_rgb)#torch.clamp(linear_to_srgb(ray_rgb), min=0.0, max=1.0)
       # exit(1)
       # print()
    #    print(normals_output.sum())
        outputs = {
            'ray_rgb': ray_rgb,  # rn,3
            'gradient_error': gradient_error,  # rn
            'acc': torch.ones_like(candidate_converges[0],device='cuda:0'),  # rn
            'normal':normals_output,
            'specular_color':specular_color_output,
            'specular_light':specular_light_output,
            'specular_ref':specular_ref_output,
            
        }

        if  outputs_tmp.__contains__('std'):
            outputs['std'] = outputs_tmp['std']
        else:
            outputs['std'] = torch.zeros(1)

        if  outputs_tmp.__contains__('gradient_error'):
            outputs['gradient_error'] = outputs_tmp['gradient_error']
        else:
            outputs['gradient_error'] = torch.zeros(1)

        # if step < 1000:
        #     mask = torch.norm(points, dim=-1) < 1.2
        #     outputs['sdf_pts'] = points[mask]
        #     outputs['sdf_vals'] = self.sdf_network.sdf(points[mask])[..., 0]

       # if not is_train:
            #outputs.update(self.compute_validation_info(z_vals, #rays_o, rays_d, weights, human_poses, step))
            #print(list(outputs.keys()))
       # print(converges)
        #exit(1)
        return outputs

    def forward(self, data):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        is_train = 'eval' not in data
        # is_render = "render" in data  # novel view render
        step = data['step']

        if is_train:
            outputs = self.train_step(step)
        # elif is_render:
        #     outputs = self.render_step(step)
        else:
            index = data['index']
            outputs = self.test_step(index, step=step)

            if index == 0 and self.cfg['val_geometry']:
                bbox_min = -torch.ones(3)
                bbox_max = torch.ones(3)
                vertices, triangles = extract_geometry(bbox_min, bbox_max, 128, 0, lambda x: self.sdf_network.sdf(x))
                outputs['vertices'] = vertices
                outputs['triangles'] = triangles

        torch.set_default_tensor_type('torch.FloatTensor')
        return outputs

    def predict_materials(self):
        name = self.cfg['name']
        mesh = open3d.io.read_triangle_mesh(f'data/meshes/{name}-300000.ply')
        xyz = np.asarray(mesh.vertices)
        xyz = torch.from_numpy(xyz.astype(np.float32)).cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        metallic, roughness, albedo = [], [], []
        batch_size = 8192
        for vi in range(0, xyz.shape[0], batch_size):
            feature_vectors = self.sdf_network(xyz[vi:vi + batch_size])[:, 1:]
            m, r, a = self.color_network.predict_materials(xyz[vi:vi + batch_size], feature_vectors)
            metallic.append(m.cpu().numpy())
            roughness.append(r.cpu().numpy())
            albedo.append(a.cpu().numpy())

        return {'metallic': np.concatenate(metallic, 0),
                'roughness': np.concatenate(roughness, 0),
                'albedo': np.concatenate(albedo, 0)}

name2renderer = {
    'shape': NeROShapeRenderer,
    'stage2': Stage2Renderer,
}
