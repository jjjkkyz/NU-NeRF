import os
import trimesh
import torch
import numpy as np
from dataset.database_formask import *
from network.DiffRender import Ray,Intersection,Scene
import cv2 as cv
database = NeRFSyntheticDatabase('nerf/thickbottle_kirsch_final','anon')


def build_imgs_info(database, img_ids, is_nerf=False):
    images = [database.get_image(img_id) for img_id in img_ids]
    poses = [database.get_pose(img_id) for img_id in img_ids]
    Ks = [database.get_K(img_id) for img_id in img_ids]

    images = np.stack(images, 0)
    masks = [database.get_depth(img_id)[1] for img_id in img_ids]
    masks = np.stack(masks, 0)
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

def imgs_info_to_torch(imgs_info, device='cuda:0'):
    for k, v in imgs_info.items():
        v = torch.from_numpy(v)
        if k.startswith('imgs'): v = v.permute(0, 3, 1, 2)
        imgs_info[k] = v.to(device)
    return imgs_info

train_ids, test_ids = get_database_split(database)
imgs_info = build_imgs_info(database,train_ids)
imgs_info = imgs_info_to_torch(imgs_info, 'cpu')

imn, _, h, w = imgs_info['imgs'].shape

i, j = torch.meshgrid(torch.linspace(0, w - 1, w),
                        torch.linspace(0, h - 1, h))  # pytorch's meshgrid has indexing='ij'
i = i.t()
j = j.t()

K = imgs_info['Ks'][0]
dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
poses = imgs_info['poses']  # imn,3,4
# if is_train:
#     masks = imgs_info['masks'].reshape(imn, h * w)

scene = Scene(anon)
for img_idx in range(imn):
    rays_d = torch.sum(dirs[..., None, :].cpu() * poses[img_idx, :3, :3], -1).cuda()
    rays_o = poses[img_idx, :3, -1].reshape(1,1,3).expand(1024,1024,3).cuda()
    #poses = poses[img_idx].reshape(1,4,4).repeat(h * w, 1, 1)
    print(rays_o.shape)
    print(rays_d.shape)
    rays_for_intersection = Ray(rays_o.reshape(-1,3),rays_d.reshape(-1,3))
    intersection_info, converged = scene.Dintersect(rays_for_intersection)
    converged = converged.reshape(1024,1024).reshape(1024,1024,1).expand(1024,1024,3).detach().cpu().numpy()
    converged = (converged * 255).astype(np.uint8)
    cv.imwrite('anonthickbottle_kirsch_final/mask/lgt0_r_' + str(img_idx) + '.jpg',converged)
   # exit(1)
