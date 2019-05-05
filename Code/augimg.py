import numpy as np
import cv2
import os
from imgaug import augmenters as iaa

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:19:15 2019

@author: bhargavdesai
"""

PATH = '/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 3/vasc/vasc stage 2'
os.chdir('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 3/vasc/vasc sup')
num = 30

'''-------FLIP LATERAL--------'''


images = np.empty((num, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(str(img_id[12:]),"_AL")))
print(images.shape)   
print(len(image_ids))
aug = iaa.Fliplr(1.0)
images_aug = aug.augment_images(images)
for i in range(num):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])


    
'''-------FLIP VERTICAL-------'''    
    

images = np.empty((num, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(str(img_id[12:]),"_AV")))
print(images.shape)   
print(len(image_ids))
aug = iaa.Flipud(1.0)
images_aug = aug.augment_images(images)
for i in range(num):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])
 
    
    
'''-----HISTOGRAM EQUALISATION----'''
  
images = np.empty((num, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(".jpg","_AC")))
print(images.shape)   
print(len(image_ids))
aug = iaa.ContrastNormalization((0.8, 1.3))
images_aug = aug.augment_images(images)
for i in range(num):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])
    



'''------------SCALE---------------'''

images = np.empty((num, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(".jpg","_AS")))
print(images.shape)   
print(len(image_ids))
aug = iaa.Affine(scale=(1.2, 1.5))
images_aug = aug.augment_images(images)
for i in range(num):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])


'''-----------SHARPEN-----------------'''

images = np.empty((num, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(".jpg","_ASH")))
print(images.shape)   
print(len(image_ids))
aug = iaa.Sharpen(alpha = 0.25)
images_aug = aug.augment_images(images)
for i in range(num):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])
    



'''--------------ROTATE-----------------'''

images = np.empty((100, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(".jpg","_AR")))
print(images.shape)   
print(len(image_ids))
aug = iaa.Affine(rotate=(-5, 5))
images_aug = aug.augment_images(images)
for i in range(100):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])


'''----------------SHEAR-----------------'''

images = np.empty((100, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(".jpg","_ASR")))
print(images.shape)   
print(len(image_ids))
aug = iaa.Affine(shear=(-5, 5))
images_aug = aug.augment_images(images)
for i in range(100):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])
    
    
'''-----------ADD--------------'''

images = np.empty((100, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(".jpg","_AA")))
print(images.shape)   
print(len(image_ids))
aug = iaa.Add((-20, 20))
images_aug = aug.augment_images(images)
for i in range(100):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])
    
    
    
'''-------------MULTIPLY-----------'''

images = np.empty((100, 128, 171, 3))
image_ids = []
for idx, img_id in enumerate(os.listdir(PATH)): 
    image = cv2.imread(PATH+'/'+img_id, 1) 
    images[idx, :, :, :] = image 
    image_ids.append((idx, img_id.replace(".jpg","_AM")))
print(images.shape)   
print(len(image_ids))
aug = iaa.Multiply((0.85, 1.1))
images_aug = aug.augment_images(images)
for i in range(100):
    cv2.imwrite(image_ids[i][1]+'.jpg', images_aug[i])
    print("wrote image"+image_ids[i][1])
        