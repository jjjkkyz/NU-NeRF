import argparse

from pathlib import Path

import torch
import trimesh
from network.field import extract_geometry

from network.renderer import name2renderer
from utils.base_utils import load_cfg
import numpy as np
import torch.nn.functional as F
import cv2 as cv
import imageio

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


def compute_envmap(H= 250, W = 500):
    phi, theta = np.meshgrid(np.linspace(0., np.pi, H), np.linspace(0.0*np.pi, 2.0*np.pi, W))
    phi = phi.T
    theta = theta.T
    viewdirs = np.stack([np.cos(theta) * np.sin(phi),np.sin(theta) * np.sin(phi),np.cos(phi)],
                           axis=-1)    # [H, W, 3]
    return torch.tensor(viewdirs).cuda().float()

def main():
    H= 250
    W = 500
    cfg = load_cfg(flags.cfg)
    print(cfg)
    network = name2renderer[cfg['network']](cfg, training=False)
    #network.color_network.bkgr = network.infinity_far_bkgr
    ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth')
    step = ckpt['step']
    dict_in = {}
    for k, v in ckpt['network_state_dict'].items():
        if k.find('outer_light') != -1:
            dict_in[k] = v
    network.load_state_dict(dict_in,strict=False)
    network.eval().cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f'successfully load {cfg["name"]} step {step}!')
    directions = compute_envmap(H,W).reshape(-1,3)
    ref_roughness = network.color_network.sph_enc(directions, torch.zeros((directions.shape[0],1),device='cuda:0'))\
    
    points = torch.zeros((directions.shape[0],3),device='cuda:0')
    if (network.color_network.outer_light[0].in_features) == 144:
        sph_points = offset_points_to_sphere(points)
        sph_points = F.normalize(sph_points + directions * get_sphere_intersection(sph_points, directions), dim=-1)
        sph_points = network.color_network.sph_enc(sph_points, torch.zeros((directions.shape[0],1),device='cuda:0'))
        res = network.color_network.outer_light(torch.cat([ref_roughness, sph_points], -1))
    else:
        res = network.color_network.outer_light(ref_roughness)
    # non sphere direction
    res = res.reshape(H,W,3).detach().cpu().numpy()
    imageio.imwrite('/home/sunjiamu/instant-ngp/bottle2/envmap.exr', res)

    print(res.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    #parser.add_argument('--resolution', type=int, default=1024)
    flags = parser.parse_args()
    main()
