#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:19:15 2019

@author: bhargavdesai
"""
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import keras
import shutil


def check_img():
    img = cv2.imread('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Malignant/ISIC_0024345.jpg', -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()    
    print(img.shape)
    

def separate_images_according_to_cancerous_benign():
    df = pd.read_csv("/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
    c = ['mel','bcc','akiec']
    b = ['bkl','df','nv','vasc']
    lc = pd.Series.tolist(df.loc[df['dx'].isin(c), 'image_id'])
    lb = pd.Series.tolist(df.loc[df['dx'].isin(b), 'image_id'])
    for image_id in lb:
        print(image_id)
        shutil.copy('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Images/'+image_id+'.jpg', '/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Benign')
        print("done copying benign images")
    for image_id in lc:
        print(image_id)
        shutil.copy('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Images/'+image_id+'.jpg', '/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Malignant')
        print("done copying cancerous images")    
        
        
def separate_images_according_to_individual_categories():
    df1 = pd.read_csv("/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
    categories = ['mel','bcc','akiec','bkl','df','nv','vasc']
    for category in categories:
        file_name = category
        category =  pd.Series.tolist(df1.loc[df1['dx'] == category, 'image_id'])
        for image_id in category:
            print(image_id)
            shutil.copy('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Images/'+image_id+'.jpg', '/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/'+file_name)
            print("done copying image")
        print("done copying the category")    
        
