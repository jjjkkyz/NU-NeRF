import cv2 as cv
import os
import numpy as np

mask_path = '/home/sunjiamu/instant-ngp/cylinderbottle/mask'
mask_erosion_path = '/home/sunjiamu/instant-ngp/cylinderbottle/mask_erosion'

if not os.path.exists(mask_erosion_path):
    os.makedirs(mask_erosion_path,exist_ok=True)

imgs = os.listdir(mask_path)

for img in imgs:
    print(img)
    img_data = cv.imread(mask_path+ '/' + img)
    img_data = cv.erode(img_data,np.ones((20, 20), np.uint8) ,cv.BORDER_REFLECT)
    cv.imwrite(mask_erosion_path+'/'+img,img_data)