import torch
from network.field import *
from network.renderer import *
import numpy as np
import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
flags = parser.parse_args()

#Trainer(load_cfg(flags.cfg)).run()

checkpoint = torch.load('/home/sunjiamu/NeRO/data/model/refrac_withexpref/model_best.pth')
network = NeROShapeRenderer(load_cfg(flags.cfg))
network.load_state_dict(checkpoint['network_state_dict'])
network.eval().cuda()
bbox_min = -torch.ones(3).cuda()
bbox_max = torch.ones(3).cuda()
with torch.no_grad():
    f = extract_fields(bbox_min,bbox_max,512,lambda x: network.sdf_network.sdf(x),batch_size=128)
print(np.abs(f).min())
print(np.abs(f).max())