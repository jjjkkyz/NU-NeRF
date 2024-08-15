import cv2 as cv 
import numpy as np
import random
import os
import glob

name = 'glassice'

images = []

for i in range(4):
    images.append(
        cv.resize(cv.imread('/home/sunjiamu/instant-ngp/' + name + '/lgt0_r_' + str(random.randint(0,250)) + '.png'),(512,512))
    )

im_upper = np.concatenate([images[0],images[1]], axis=1)
im_bottom = np.concatenate([images[2],images[3]], axis=1)
im = np.concatenate([im_upper, im_bottom], axis=0)
print(im.shape)
#cv.imshow('a',im)
#cv.waitKey(0)
cv.imwrite('thumb_' + name + '.jpg',im)


# images = []
# paths = glob.glob('/home/sunjiamu/instant-ngp/' + name + '/images/*.JPG')

# for i in range(4):
#     images.append(
#         cv.imread((random.choice(paths)))
#     )
# H,W = images[0].shape[0], images[0].shape[1]

# im_upper = np.concatenate([images[0],images[1]], axis=1)
# im_bottom = np.concatenate([images[2],images[3]], axis=1)
# im = np.concatenate([im_upper, im_bottom], axis=0)
# print(im.shape)
# #cv.imshow('a',im)
# #cv.waitKey(0)
# cv.imwrite('thumb_' + name + '.jpg',im)