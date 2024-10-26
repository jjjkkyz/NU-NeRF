import argparse

from train.trainer import Trainer
from train.trainer_zero import Trainer_zero
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
#parser.add_argument("--zero_thickness", action="store_true")


flags = parser.parse_args()
cfg_tmp  =load_cfg(flags.cfg)
if cfg_tmp['zero_thickness']:
    Trainer_zero(load_cfg(flags.cfg)).run()
else:
    Trainer(load_cfg(flags.cfg)).run()
