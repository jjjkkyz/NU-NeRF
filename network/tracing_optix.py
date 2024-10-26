import os
import optix as ox
import cupy as cp
import numpy as np
from PIL import Image, ImageOps
import trimesh
import torch

script_dir = os.path.dirname(__file__)
cuda_src = 'cuda/triangle.cu'

#img_size = (1024, 768)

# use a regular function for logging
def log_callback(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))
    pass


def create_acceleration_structure(ctx, vertices,faces):
    build_input = ox.BuildInputTriangleArray(vertices,faces, flags=[ox.GeometryFlags.NONE])
    gas = ox.AccelerationStructure(ctx, build_input, compact=True, allow_update=True, random_vertex_access=True)
    return gas

def update_accleration_structure(ctx,gas, vertices,faces):
    build_input = ox.BuildInputTriangleArray(vertices,faces, flags=[ox.GeometryFlags.NONE])
    gas.update(build_input)
    return gas

def create_module(ctx, pipeline_opts):
    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.FULL, opt_level=ox.CompileOptimizationLevel.LEVEL_0)
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module


def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_CH="__closesthit__ch")

    return raygen_grp, miss_grp, hit_grp


def create_pipeline(ctx, program_grps, pipeline_options):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                       debug_level=ox.CompileDebugLevel.FULL)

    pipeline = ox.Pipeline(ctx,
                           compile_options=pipeline_options,
                           link_options=link_opts,
                           program_groups=program_grps)

    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 1)  # max_dc_depth

    return pipeline


def create_sbt(program_grps):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp)
    #miss_sbt['rgb'] = [0.3, 0.1, 0.2]

    hit_sbt = ox.SbtRecord(hit_grp)
    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)

    return sbt


def launch_pipeline(pipeline : ox.Pipeline, sbt, gas,rays_o,rays_d):

    hit = np.zeros(rays_o.shape[:2] + (1, ), dtype=np.float32)
    hit[:, :, :] = 0

    triangle_index = np.zeros(rays_o.shape[:2] + (1, ), dtype=np.uint32)
    triangle_index[:, :, :] = [10000]

   # rays_o = cp.asarray(rays_o.detach().cpu().numpy().astype(np.float32))
   # rays_d = cp.asarray(rays_d.detach().cpu().numpy().astype(np.float32))
    rays_o = cp.asarray(rays_o)
    rays_d = cp.asarray(rays_d)

    hit = cp.asarray(hit)
    triangle_index = cp.asarray(triangle_index)
    params_tmp = [
        ( 'u8', 'hit'),
        ( 'u8', 'triangle_index'),
        ( 'u8', 'rays_o'),
        ( 'u8', 'rays_d'),
        ( 'u4', 'image_width'),
        ( 'u4', 'image_height'),
        ( 'u8', 'trav_handle')
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                   formats=[p[0] for p in params_tmp])
   # print(params)
    params['hit'] = hit.data.ptr
    params['triangle_index'] = triangle_index.data.ptr

    params['rays_o'] = rays_o.data.ptr
    params['rays_d'] = rays_d.data.ptr
    params['image_width'] = rays_o.shape[0]
    params['image_height'] = rays_o.shape[1]
    params['trav_handle'] = gas.handle
    #print(rays_o.shape[1])
    stream = cp.cuda.Stream()

    pipeline.launch(sbt, dimensions=rays_o.shape[:2], params=params, stream=stream)

    stream.synchronize()

    return cp.asnumpy(hit), cp.asnumpy(triangle_index)


def optix_tracing(ctx,rays_o,rays_d,gas,pipeline,sbt):
   # ctx = ox.DeviceContext(validation_mode=True, log_callback_function=log_callback, log_callback_level=3)
    #gas = create_acceleration_structure(ctx, vertices)
    
    hit,idx = launch_pipeline(pipeline, sbt, gas,rays_o,rays_d)
    return hit,idx
 #   return img

class optix_mesh:
    def __init__(self):
        self.ctx = ox.DeviceContext(validation_mode=True, log_callback_function=log_callback, log_callback_level=3)
        pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                    num_payload_values=2,
                                                    num_attribute_values=4,
                                                    exception_flags=ox.ExceptionFlags.NONE,
                                                    pipeline_launch_params_variable_name="params")

        self.module = create_module(self.ctx, pipeline_options)
        self.program_grps = create_program_groups(self.ctx, self.module)
        self.pipeline = create_pipeline(self.ctx, self.program_grps, pipeline_options)
        self.sbt = create_sbt(self.program_grps)
        self.gas = None
    def update_mesh(self,_F,_V):
        self.F = _F.detach().cpu().numpy().astype(np.uint32)
        self.V = _V.detach().cpu().numpy().astype(np.float32)
       # if self.gas is None:
        self.gas = create_acceleration_structure(self.ctx, self.V,self.F)
       # else:
        #    self.gas = update_accleration_structure(self.ctx,self.gas,self.V,self.F)

    def update_vert(self,_V):
        self.V = _V.detach().cpu().numpy().astype(np.float32)
        self.gas = update_accleration_structure(self.ctx,self.gas,self.V,self.F)

    def intersect(self,ray):
        rays_o = ray[:,:3].reshape(1,-1,3).detach().cpu().numpy()
        rays_d = ray[:,3:].reshape(1,-1,3).detach().cpu().numpy()
        hit,idx = optix_tracing(self.ctx,rays_o,rays_d,self.gas,self.pipeline,self.sbt)
        return torch.tensor(hit.reshape(-1),device='cuda:0'), torch.tensor(idx.reshape(-1).astype(np.int32),device='cuda:0')

if __name__ == '__main__':
    ctx = ox.DeviceContext(validation_mode=True, log_callback_function=log_callback, log_callback_level=3)
    pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                num_payload_values=2,
                                                num_attribute_values=4,
                                                exception_flags=ox.ExceptionFlags.NONE,
                                                pipeline_launch_params_variable_name="params")

    module = create_module(ctx, pipeline_options)
    program_grps = create_program_groups(ctx, module)
    pipeline = create_pipeline(ctx, program_grps, pipeline_options)
    sbt = create_sbt(program_grps)
    
    mesh_tmp = trimesh.load('anon')
    #mesh_tmp = trimesh.load(anon)
    #self.renderer.bvh = cubvh.cuBVH(mesh_tmp.vertices, mesh_tmp.faces)
    vertices = cp.asarray(mesh_tmp.vertices.astype(np.float32))
    faces = cp.asarray(mesh_tmp.faces.astype(np.uint32))
    print(vertices.shape)
    print(faces.shape)
    gas = create_acceleration_structure(ctx, vertices,faces)

    
    rays_o = np.zeros((100,100,3),dtype=np.float32)
    rays_d = np.zeros((100,100,3),dtype=np.float32)
   
    rays_d[:,:,2] = -1
    rays_o[:,:,-1] = 10
  
    
   # print(rays_d)
    for i in range(100):
        for j in range(100):
            rays_o[i,j,0] = -106 + 204 * i / 100
            rays_o[i,j,1] = -1 + 151 * j / 100
        #   rays_o[i,j,0] = -0.4 + 0.8 * i / 100
         #  rays_o[i,j,1] = -0.4 + 0.8 * j / 100
   # rays_o = rays_o.reshape(-1,3)
   # rays_d = rays_d.reshape(-1,3)
    #print(rays_o)
 #   for i in range(100):
  #     print(rays_d.item(i))
    pc = trimesh.PointCloud(vertices = rays_o.reshape(-1,3))
    trimesh.exchange.export.export_mesh(pc,'tstpc.ply')

    hit,idx = optix_tracing(ctx,rays_o,rays_d,gas,pipeline,sbt)
    print(hit)
    print(idx)
    hit = hit.astype(np.bool_).reshape(100,100)
    pc = trimesh.PointCloud(vertices = rays_o[hit].reshape(-1,3))
    trimesh.exchange.export.export_mesh(pc,'tstpc1.ply')
   # print(hit.shape)
   # print(idx)