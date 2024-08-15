import pymesh
import numpy as np
mesh = pymesh.load_mesh("/home/sunjiamu/NeRO/sphere0.553.ply")

mesh.add_attribute("vertex_gaussian_curvature")
print(mesh.get_attribute("vertex_gaussian_curvature").max())
print(mesh.get_attribute("vertex_gaussian_curvature").min())


gaussian_cur = 1 / np.sqrt(np.abs(np.clip(mesh.get_attribute("vertex_gaussian_curvature"),a_min=-10.0,a_max=10.0)))

np.savetxt('gaussian.txt',gaussian_cur)

print(np.abs(np.ones_like(gaussian_cur) * 0.553 - gaussian_cur).mean())

mesh.add_attribute("vertex_mean_curvature")
mean_cur = 1 / (np.abs(np.clip(mesh.get_attribute("vertex_mean_curvature"),a_min=-10.0,a_max=10.0)))
np.savetxt('mean.txt',mean_cur)
print(np.abs(np.ones_like(gaussian_cur) * 0.553 - mean_cur).mean())
