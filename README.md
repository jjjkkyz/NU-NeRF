

# NU-NeRF: Neural Reconstruction of Nested Transparent Objects with Uncontrolled Capture Environment
### ACM Transactions on Graphics(SIGGRAPH Asia 2024)


## Jia-Mu Sun, Tong Wu, Ling-Qi Yan, Lin Gao
### Institute of Computing Technology, CAS

### University of Chinese Academy of Sciences

### [KIRI Innovation](https://www.kiriengine.app/)

### University of California, Santa Barbara
## [Project Page](http://geometrylearning.com/NU-NeRF/) | [Paper](https://drive.google.com/drive/folders/1DP_aQ5GRow-Se4LpImYLjX3mah2__PSh?usp=sharing) (Our version, not TOG version)
****

## Update
[24.11] Release the project.

## Get Started
### Setting up environments
#### Step 1. Install python requirements using pip
```bash
pip install -r requirements
```
#### Step 2. Install PyMesh and nvdiffrast and python-optix
In this project, we use PyMesh to calculate the curvature used for non-zero thickness stage 2 reconstruction.
PyMesh currently only support Linux(If you try to install it on Windows, it may get REALLY messy)
Since PyMesh is not actively maintained now, we use part of @bhacha ‘s fork to solve some problems.
PLEASE Make sure your default CXX complier is g++-9, otherwise tbb compiling will produce an error.
```bash
git clone https://github.com/bhacha/PyMesh.git
cd PyMesh
git submodule update --init
export PYMESH_PATH=`pwd`
sudo apt-get install libeigen3-dev libgmp-dev libgmpxx4ldbl libmpfr-dev libboost-dev libboost-thread-dev libtbb-dev python3-dev
pip install -r python/requirements.txt

git submodule deinit third_party/mmg
cd third_party
rm -rf mmg # We need to get a different MMG version to avoid compiling errors
git clone --depth=1 -b v5.4.3 https://github.com/MmgTools/mmg.git

cd ..
./setup.py build
./setup.py install

cd .. 
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .

cd ..
# please install optix 7.6 and cuda from NVIDIA website
# and fill in the path to them in the following lines
export OPTIX_PATH=/path/to/optix
export CUDA_PATH=/path/to/cuda_toolkit
export OPTIX_EMBED_HEADERS=1 # embed the optix headers into the package
git clone https://github.com/78ij/python-optix
cd python-optix
pip install .
```
### Download our Datasets(a part of the scenes shown in the paper)

We have provided  some synthetic and real datasets used in the paper in [Google Drive](https://drive.google.com/drive/folders/1xe4c2io66j1hbLGitXedkOJNCtizZr2V?usp=sharing). 

### -OR- Preparing your own dataset
Please consult [the manual for NeRO for now](https://github.com/liuyuan-pal/NeRO/blob/main/custom_object.md).

### Fill in config files or use the corresponding file for our dataset
Explanations of some essential entrys in the config files. Most of the config entry should document themselves. If you are using our released datasets, you can use the configs in the configs/ folder, only needing to modify the dataset directories.

```yaml
network: shape # set to 'shape' if stage1, 'stage2' if stage2.
dataset_dir: # fill in the dataset directory. if you have a dataset named 'test' in /foo/bar/test, you should fill '/foo/bar' here.
database_name: # fill in the dataset folder name and type. Please consult the example configs for the detailed usage. 
zero_thickness: #whether to use zero thickness configuration. Should be the same in stage1 and stage2.
shader_config:
  sphere_direction: false # Whether to apply the sphere direction formulation. If false, only direction is fed into the light predition(i.e. infinity far light)
  human_light: false # Whether to apply the human light assumption from NeRO. Should be false all the time in our experiments.
```

### Run Stage 1 Reconstruction(Example on Spherepot dataset)
```bash
python run_training.py --cfg configs/shape/nerf/spherepot.yaml 
```

### Extract Outer Geometry mesh for stage1
```bash
python extract_mesh_stage1.py --cfg configs/shape/nerf/spherepot.yaml 
### the script will output original MC mesh and a fixed and simplified version of mesh. Please use the simplified version for Stage2, since original MC mesh may have degenerated normal/surfaces, causing ray tracing to fail, and do not have reasonable curvature)
### the extracted result will be located in data/meshes/EXPNAME-step.ply
### the simplified version will be located in data/meshes/EXPNAME-step_simplified.ply
```

### Render mask and do erosion on the extracted mask(If you are using non-zero thickness configuration)
```bash
python render_mask.py --cfg configs/shape/nerf/spherepot.yaml --mesh_path <exported mesh path>
python mask_erosion.py --cfg configs/shape/nerf/spherepot.yaml --mesh_path <exported mesh path>
```
### Fill in directories of the Stage 2 config file
```yaml
#Please fill in the following 3 directories:
stage1_mesh_dir: <stage1_mesh_dir> # should be the extracted SIMPLIFIED mesh 
stage1_ckpt_dir: <stage1_ckpt_dir> # should be the checkpoint at data/model/EXP_NAME/model_best.pth or model.pth
stage1_cfg_dir: <stage1_cfg_dir> # should be the stage1 config file you used, here it should be ./configs/shape/nerf/spherepot.yaml 
```


### Run Stage 2 Reconstruction(Example on Spherepot dataset)
```bash
python run_training.py --cfg configs/stage2/nerf/spherepot.yaml 
```

### Extract Inner Geometry mesh for stage2 and do postprocessing(OPTIONAL)
```bash
python extract_mesh_stage2.py --cfg configs/stage2/nerf/spherepot.yaml

# please adjust the pathes in inner/outer meshes in postprocess_stage2_mesh.py to your extracted mesh path.
python postprocess_stage2_mesh.py

```

### Acknowledgements
This project is funded by [KIRI Innovation](https://www.kiriengine.app/). A portion of the dataset is also provided by them. 

A large portion of this repo is built upon the code of [NeRO](https://github.com/liuyuan-pal/NeRO). Thank the authors of NeRO for their incredible work!

### Cite
If you find the paper/code/data helpful for you, please cite our work:
```
@article{NU-NeRF,
    author = {Jia-Mu Sun and Tong Wu and Ling-Qi Yan and Lin Gao},
    title = {NU-NeRF: Neural Reconstruction of Nested Transparent Objects with Uncontrolled Capture Environment},
    journal = {ACM Transactions on Graphics(ACM SIGGRAPH Asia 2024)},
    year = {2024}
}
```
