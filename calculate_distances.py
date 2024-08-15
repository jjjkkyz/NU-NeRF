from chamfer_distance import ChamferDistance
from PyTorchEMD.emd import earth_mover_distance
from scipy.spatial.distance import cdist
import trimesh, trimesh.sample
import torch

torch.set_printoptions(precision=7)
NUMBER = 15000
mesh_gt = trimesh.load('/home/sunjiamu/instant-ngp/ours_numbers/bottle2_gt.ply',force='mesh')
mesh_ours = trimesh.load('/home/sunjiamu/instant-ngp/ours_numbers/bottle2_nemto.ply',force='mesh')
#mesh_ours = trimesh.load('/home/sunjiamu/NeRO/data/meshes/refrac_withexpref_bottle_nothickness_near14_wloss4-376000_inverted.ply',force='mesh')
#mesh_ours = trimesh.load('/home/sunjiamu/li2020/data/plastic_RGB_250/Shape__0/reconstruct_20_view_renderError_loss_chamfer_Subd.ply',force='mesh')
mesh_gt_points = torch.tensor(trimesh.sample.sample_surface(mesh_gt,NUMBER)[0],device='cuda:0').reshape(-1,3).float() 
mesh_ours_points = torch.tensor(trimesh.sample.sample_surface(mesh_ours,NUMBER)[0],device='cuda:0').reshape(-1,3).float() 

print(torch.any(torch.isnan(mesh_gt_points)))
print(torch.any(torch.isnan(mesh_ours_points)))

cd_metric = ChamferDistance()
dist1, idx1, dist2, idx2 = cd_metric(mesh_gt_points.reshape(1,-1,3), mesh_ours_points.reshape(1,-1,3))
cd = (dist1.mean(dim=1) + dist2.mean(dim=1))

emd = earth_mover_distance(mesh_gt_points.reshape(1,-1,3), mesh_ours_points.reshape(1,-1,3), transpose=False)

print(emd / NUMBER)
print(cd)
