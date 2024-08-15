import argparse

from pathlib import Path
import cv2 as cv
import torch
import trimesh
from network.field import extract_geometry
import numpy as np
from network.renderer_zerothick_copy import name2renderer
from utils.base_utils import load_cfg
from trimesh.visual.color import *
from trimesh.exchange import *

def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError
def main():
    cfg = load_cfg(flags.cfg)
    print(cfg)
    with torch.no_grad():
        network = name2renderer[cfg['network']](cfg, training=False)
        #network.color_network.bkgr = network.infinity_far_bkgr
        ckpt = torch.load(f'data/model/{cfg["name"]}/model_best.pth')
        step = ckpt['step']
        network.load_state_dict(ckpt['network_state_dict'])
        network.eval().cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(f'successfully load {cfg["name"]} step {step}!')

        base_color = np.zeros((1024,1024,3),dtype=np.float32)
        roughness = np.zeros((1024,1024,1),dtype=np.float32)
        metallic = np.zeros((1024,1024,1),dtype=np.float32)
        
        uv_mesh = trimesh.load('/home/sunjiamu/instant-ngp/pot_uv_true.obj',process=False, maintain_order=True)
        normal = torch.tensor(uv_mesh.vertex_normals,device='cuda:0').reshape(-1,3).float()
        
        vert = torch.tensor(uv_mesh.vertices,device='cuda:0').reshape(-1,3).float()# - normal * 0.005

        print(vert.max(0))
        print(vert.min(0))
        feature_vectors = network.sdf_network_inner(vert)[..., 1:]
        sdf = network.sdf_network_inner(vert)[..., 0]
        print(sdf.max())
        print(sdf.min())
        net_input = torch.cat([feature_vectors, vert], -1)
        #net_input1 = torch.cat([feature_vectors, network.color_network_inner.pos_enc(vert),network.color_network_inner.dir_enc(vert)], -1)
        #color = network.color_network_inner.refrac_light(net_input1).reshape(-1,3).detach().cpu().numpy()

        color = linear_to_srgb(network.color_network_inner.albedo_predictor(net_input)).reshape(-1,3).detach().cpu().numpy()
        cc = np.clip(color,0,1)
        visual = ColorVisuals(uv_mesh,None,(cc*255).astype(np.uint8))
        uv_mesh.visual = visual
        #uv_mesh.show()
        tex = visual.to_texture()
        #print(tex.material.to_obj()[2].keys())
        obj = trimesh.exchange.obj.export_obj(uv_mesh)
        with open('pot_e.obj','w') as f:
            f.write(obj)
        exit(1)
        metal = network.color_network_inner.metallic_predictor(net_input).reshape(-1,1).detach().cpu().numpy()
        rough = network.color_network_inner.roughness_predictor(net_input).reshape(-1,1).detach().cpu().numpy()
        print(vert.shape)
        print(uv_mesh.visual.uv.shape)
        # print(uv_mesh.visual.uv.max())
        # print(uv_mesh.visual.uv.min())
        for i in range(color.shape[0]):
            uv = uv_mesh.visual.uv[i]
            actual_coordinate = uv*1024
            u = 1024 - int(actual_coordinate[1])
            v = int(actual_coordinate[0])
           # print(color[i])
            #base_color[u,v] = color[i] * 255
            base_color[u,v] = color[i]
            metallic[u,v] = metal[i]
        cv.imwrite('albedo.png',(base_color * 255).astype(np.uint8)[...,[2,1,0]])

        cv.imwrite('metallic.png',(metallic * 255).astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    flags = parser.parse_args()
    main()
