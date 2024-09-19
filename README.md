

# NU-NeRF: Neural Reconstruction of Nested Transparent Objects with Uncontrolled Capture Environment

## Jia-Mu Sun, Tong Wu, Ling-Qi Yan, Lin Gao
### ACM Transactions on Graphics(SIGGRAPH Asia 2024)
## [Project Page](http://geometrylearning.com/NU-NeRF/) | Paper (Coming Soon)
****

## Get Started
### Setting up environments
#### Step 1. Install python requirements using pip
```bash
pip install -r requirements.txt
```
#### Step 2. Install PyMesh
PyMesh currently only support Linux(If you try to install it on Windows, it may get REALLY messy)
Since PyMesh is not actively maintained now, we use part of @bhacha ‘s fork to solve some problems.
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
./setup build
./setup install
```
### Downloading Datasets 

We have provided  some synthetic and real datasets used in the paper in [Google Drive](FIXME). 

### Preparing your own dataset

TODO

### Fill in config files
TODO

### Run Stage 1 Reconstruction(Example on Spherepot dataset)
```bash
python run_training.py --cfg configs/shape/nerf/spherepot.yaml 
```

### Run Stage 2 Reconstruction(Example on Spherepot dataset)
```bash
python run_training.py --cfg configs/stage2/nerf/spherepot.yaml 
```