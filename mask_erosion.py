import cv2 as cv
import os
import numpy as np
import argparse

from utils.base_utils import load_cfg
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
flags = parser.parse_args()
cfg =load_cfg(flags.cfg)

if cfg['is_nerf']:
    mask_path = os.path.join(cfg['dataset_dir'], cfg['database_name'].split('/')[-1], 'mask')
    mask_erosion_path = os.path.join(cfg['dataset_dir'], cfg['database_name'].split('/')[-1], 'mask_erosion')
else:
    mask_path = os.path.join(cfg['dataset_dir'], cfg['database_name'].split('/')[-2], 'mask')
    mask_erosion_path = os.path.join(cfg['dataset_dir'], cfg['database_name'].split('/')[-2], 'mask_erosion')
if not os.path.exists(mask_erosion_path):
    os.makedirs(mask_erosion_path)

if not os.path.exists(mask_erosion_path):
    os.makedirs(mask_erosion_path,exist_ok=True)

imgs = os.listdir(mask_path)

for img in imgs:
    print(img)
    img_data_orig = cv.imread(mask_path+ '/' + img)
    img_data_orig = img_data_orig.max() - img_data_orig
    img_data = cv.imread(mask_path+ '/' + img)
    img_data = cv.erode(img_data,np.ones((20, 20), np.uint8) ,cv.BORDER_REFLECT)
    cv.imwrite(mask_erosion_path+'/'+img,img_data + img_data_orig)