#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:55:11 2019

@author: bhargavdesai
"""

import os 
def rename_train():
    l = ['Contrast Normalisation','Scaled','Vertical']
    for folder in l:
        for img_id in os.listdir('/Users/bhargavdesai/Desktop/Train (Augmented)'+'/'+folder):
            if (img_id=='.DS_Store'):
                print("---------------Found .DS_Store File breaking the loop-------------------------------------")
                break
            img_id_r = img_id.replace(".jpg",("_A"+folder[0])+".jpg")
            path_folder = '/Users/bhargavdesai/Desktop/Train (Augmented)'+'/'+folder
            path_img_id = os.path.join(path_folder, img_id)
            path_img_id_r = os.path.join(path_folder, img_id_r)
            os.rename(path_img_id, path_img_id_r)
  


def rename_test():
    l = ['Contrast Normalisation','Scaled','Vertical']
    for folder in l:
        for img_id in os.listdir('/Users/bhargavdesai/Desktop/Test (Augmented)'+'/'+folder):
            if (img_id=='.DS_Store'):
                print("---------------Found .DS_Store File breaking the loop-------------------------------------")
                break
            img_id_r = img_id.replace(".jpg",("_A"+folder[0])+".jpg")
            path_folder = '/Users/bhargavdesai/Desktop/Test (Augmented)'+'/'+folder
            path_img_id = os.path.join(path_folder, img_id)
            path_img_id_r = os.path.join(path_folder, img_id_r)
            os.rename(path_img_id, path_img_id_r)
     
        
rename_train() 
rename_test()           