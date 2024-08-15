import numpy as np

import trimesh, trimesh.proximity, trimesh.exchange
import os
import open3d as o3d

scene = o3d.t.geometry.RaycastingScene()



mesh_inner_path = '/home/sunjiamu/NeRO/data/meshes/refrac_s2_plastic_single2-100011.ply'
mesh_outer_path = '/home/sunjiamu/NeRO/data/meshes/refrac_withexpref_bottle_nothickness_near13-300000inv.ply'

mesh_outer = trimesh.load(mesh_outer_path)
mesh_inner = trimesh.load(mesh_inner_path)
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_outer.as_open3d))
ret_dict = scene.compute_closest_points(o3d.core.Tensor.from_numpy(mesh_inner.vertices.astype(np.float32)))
distance = np.linalg.norm(mesh_inner.vertices - ret_dict['points'].numpy(),axis=-1)
print(distance)
print(2)
mask = distance > 0.005
face_mask = mask[mesh_inner.faces].all(axis=1)
mesh_inner.update_faces(face_mask)
with open('s2_cleaned.ply','wb') as f:
    f.write(trimesh.exchange.ply.export_ply(mesh_inner))