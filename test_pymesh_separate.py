import pymesh
import numpy as np
mesh = pymesh.meshio.load_mesh('/home/sunjiamu/NeRO/2.ply', drop_zero_dim=False)
mesh_separated = pymesh.separate_mesh(mesh,'voxel')
print(mesh_separated)

for i in range(len(mesh_separated)):
    pymesh.meshio.save_mesh('m' + str(i) + '.ply', mesh_separated[i])


# bounding_boxes = []

# for m in mesh_separated:
#     bounding_boxes.append([np.max(m.vertices,axis=0),np.min(m.vertices,axis=0)])

# print(bounding_boxes)