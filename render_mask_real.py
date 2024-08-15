import os
import trimesh
import torch
import numpy as np
from dataset.database_formask import *
from network.DiffRender import Ray,Intersection,Scene
import cv2 as cv
database = CustomDatabase('custom/ballstatue/1080','/home/sunjiamu/instant-ngp')
import torch.nn.functional as F

def build_imgs_info(database, img_ids, is_nerf=False):
    images = [database.get_image(img_id) for img_id in img_ids]
    poses = [database.get_pose(img_id) for img_id in img_ids]
    Ks = [database.get_K(img_id) for img_id in img_ids]
    #print(Ks)
   # exit(1)
    names = [database.image_names[img_id] for img_id in img_ids]

    images = np.stack(images, 0)
    masks = [database.get_depth(img_id)[1] for img_id in img_ids]
    masks = np.stack(masks, 0)
    Ks = np.stack(Ks, 0).astype(np.float32)
    poses = np.stack(poses, 0).astype(np.float32)

    imgs_info = {
        'imgs': images, 
        'Ks': Ks, 
        'poses': poses,
        'names':names
    }

    if is_nerf:
        imgs_info['masks'] = masks
    
    return imgs_info

def imgs_info_to_torch(imgs_info, device='cpu'):
    for k, v in imgs_info.items():
        if k == 'names': continue
        v = torch.from_numpy(v)
        if k.startswith('imgs'): v = v.permute(0, 3, 1, 2)
        imgs_info[k] = v.to(device)
    return imgs_info

train_ids, test_ids = get_database_split(database)
imgs_info = build_imgs_info(database,train_ids)
imgs_info = imgs_info_to_torch(imgs_info, 'cpu')
imn, _, h, w = imgs_info['imgs'].shape

coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
#print(coords[504//2,672//2])
#coords = coords.to('cuda:0')
coords = coords.float()[None, :, :, :].repeat(imn, 1, 1, 1)  # imn,h,w,2
coords = coords.reshape(imn, h * w, 2)
coords = torch.cat([coords + 0.5, torch.ones(imn, h * w, 1, dtype=torch.float32)],2)  # imn,h*w,3
#print(coords[0].reshape(504,672,-1)[504//2,672//2])
# print(coords.shape)
# print(imgs_info['Ks'].shape)
# print((coords @ torch.inverse(imgs_info['Ks']).permute(0, 2, 1).cuda()).shape)
rays_d = coords @ torch.inverse(imgs_info['Ks']).permute(0, 2, 1)
#print(rays_d[0].reshape(504,672,-1)[504//2,672//2])

poses = imgs_info['poses']

R, t = poses[:, :, :3], poses[:, :, 3:]
#print(R[0],t[0])
rays_d = rays_d @ R
#print(rays_d[0].reshape(504,672,-1)[504//2,672//2])

rays_d = F.normalize(rays_d, dim=-1)
rays_o = -R.permute(0, 2, 1) @ t  # imn,3,3 @ imn,3,1
rays_o = rays_o.permute(0, 2, 1).repeat(1, h * w, 1)  # imn,h*w,3
#print(rays_o.shape)
#print(rays_d.shape)
#print(rays_o[0])

#print(rays_d[0].reshape(504,672,-1)[504//2,672//2])
#exit(1)
# if is_train:
#     masks = imgs_info['masks'].reshape(imn, h * w)
scene = Scene('/home/sunjiamu/NeRO/data/meshes/ballstatue3-300000_inv.ply')
for img_idx in range(imn):
    
    rays_o_i = rays_o[img_idx].cuda()
    rays_d_i = rays_d[img_idx].cuda()
    print(rays_o_i.shape)
    print(rays_d_i.shape)
    #exit(1)
    rays_for_intersection = Ray(rays_o_i.reshape(-1,3),rays_d_i.reshape(-1,3))
    intersection_info, converged = scene.Dintersect(rays_for_intersection)
    converged = converged.reshape(1920,1080).reshape(1920,1080,1).expand(1920,1080,3).detach().cpu().numpy()
    converged = (converged * 255).astype(np.uint8)
    print(database.image_names)
    cv.imwrite('/home/sunjiamu/instant-ngp/ballstatue/mask/' + imgs_info['names'][img_idx][:-4] +  '.jpg',converged)
   # exit(1)
