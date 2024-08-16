import argparse

from pathlib import Path

import torch
import trimesh
from network.field import extract_geometry

from network.renderer import name2renderer
from utils.base_utils import load_cfg


def main():
    cfg = load_cfg(flags.cfg)
    print(cfg)
    network = name2renderer[cfg['network']](cfg, training=False)
    #network.color_network.bkgr = network.infinity_far_bkgr
    ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth')
    step = ckpt['step']
    network.load_state_dict(ckpt['network_state_dict'])
    network.eval().cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f'successfully load {cfg["name"]} step {step}!')

    network2 = name2renderer['shape'](cfg,training=False)
    ckpt2 = torch.load(f'anon')
    step2 = ckpt2['step']
    network2.load_state_dict(ckpt2['network_state_dict'])
    network2.eval().cuda()


    bbox_min = -torch.ones(3)
    bbox_max = torch.ones(3)
    with torch.no_grad():
        vertices, triangles = extract_geometry(bbox_min, bbox_max, flags.resolution, 0,
                                               lambda x: torch.where( network2.sdf_network.sdf(x) < 0, network.sdf_network_inner.sdf(x), torch.ones((x.shape[0],1),device='cuda:0')))
                                              # lambda x:  network.sdf_network.sdf(x))

    # output geometry
    mesh = trimesh.Trimesh(vertices, triangles)
    output_dir = Path('data/meshes')
    output_dir.mkdir(exist_ok=True)
    mesh.export(str(output_dir / f'{cfg["name"]}-{step}11.ply'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    flags = parser.parse_args()
    main()
