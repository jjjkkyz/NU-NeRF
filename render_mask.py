import argparse

from utils.render_mask_real import *
from utils.render_mask_synthetic import *
from utils.base_utils import load_cfg
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
parser.add_argument('--mesh_path', type=str)
#parser.add_argument("--zero_thickness", action="store_true")


flags = parser.parse_args()
cfg =load_cfg(flags.cfg)
if cfg['is_nerf']:
    render_mask_synthetic(cfg['dataset_dir'], cfg['database_name'], flags.mesh_path)
else:
    render_mask_real(cfg['dataset_dir'], cfg['database_name'], flags.mesh_path)
