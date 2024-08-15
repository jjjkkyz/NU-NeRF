import argparse

from train.trainer_onlygr import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
flags = parser.parse_args()

Trainer(load_cfg(flags.cfg)).run()
