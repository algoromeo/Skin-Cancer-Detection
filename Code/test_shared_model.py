#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 00:58:27 2019

@author: bhargavdesai
"""

import os
import pandas as pd 
import cv2
import numpy as np
import keras
from keras.layers import Input
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Model
from keras.utils import plot_model
import sklearn.metrics

os.chdir('/Users/bhargavdesai/Desktop/Test')

def create_model():
    stage1 = load_model('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Models/final_categorical_stage_1_lr=0.000075_trained-on-mac.h5')
    stage1.summary()
    plot_model(stage1, to_file='left branch.png')
    stage2 = load_model('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Models/final_cat_stage_2_lr=95e-6_trained-on-mac.h5')
    stage2.summary()
    plot_model(stage2, to_file='right branch.png')
    master_input = Input(shape=(128,171,3))
    stage_1_out = stage1(master_input)
    stage_2_out = stage2(master_input)
    merged = keras.layers.average([stage_1_out, stage_2_out])
    model = Model(inputs=master_input, outputs=merged)
    model.summary()
    plot_model(model, to_file='shared_final_model.png')
    return model
    

def create_labels_as_list():
    label_dict = dict([('akiec', 0), ('bcc', 1), ('mel', 2), ('bkl', 3), ('df', 4), ('nv', 5), ('vasc', 6)]) 
    x_test = []
    test_labels = []
    df = pd.read_csv("/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
    for img_id in os.listdir('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 3/Full Test'):
        if (img_id=='.DS_Store'):
                print("---------------Found .DS_Store File breaking the loop-------------------------------------")
                break
        x_test.append(img_id)
        img_id_wo_suffix = extract_suffix_if_any(img_id)
        print(img_id_wo_suffix)
        dx_value = pd.Series.tolist(df.loc[df['image_id'] == img_id_wo_suffix, 'dx'])
        dx_value = ''.join(dx_value)
        print(dx_value)
        label = label_dict[dx_value]
        print(label)
        test_labels.append(label)
    return test_labels, x_test



def extract_suffix_if_any(img_id):
        img_id_suffix = img_id[12:]
        for suffix in ['_AC.jpg', '_AS.jpg', '_AV.jpg', '_AL.jpg', '_ASH.jpg', '_ALSH.jpg', '_ALC.jpg', '_ALS.jpg', '_AVSH.jpg', '_AVC.jpg', '_AVS.jpg']:
            if (img_id_suffix == suffix):
                img_id = img_id.replace(suffix, '')
            else:
                continue
        img_id = img_id.replace('.jpg', '')    
        return img_id     
 
    
def one_hot(test_labels):
    test_labels = to_categorical(test_labels)
    return test_labels
    

def load_test_array_with_labels(test_labels, x_test):
    size = len(test_labels)
    x_t = np.empty((size, 128, 171, 3))
    y_t = one_hot(test_labels)
    for idx, img_id in enumerate(x_test):
        if (img_id=='.DS_Store'):
            print("---------------Found .DS_Store File breaking the loop-------------------------------------")
            break
        image = cv2.imread('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 3/Full Test'+'/'+img_id, 1) 
        x_t[idx, :, :, :] = image
    print("done with converting to array")
    return x_t, y_t  


def predictions(model, x_t, test_labels):
    preds = model.predict(x_t)
    pred_label = np.empty((len(test_labels)),dtype=int)
    for i in range(0,preds.shape[0]):
        pred_label[i] = np.argmax(preds[i])
    print(pred_label.shape)    
    correct = 0
    incorrect = 0
    y_arr = np.asarray(test_labels)
    for i in range(preds.shape[0]):
        if pred_label[i] == y_arr[i]:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    acc = (correct/(correct+incorrect))*100   
    print('accuracy=', acc)
    return pred_label, y_arr
    
    

def conf_matrix(y_arr, pred_label):
    conf_mat = sklearn.metrics.confusion_matrix(y_arr, pred_label)
    print("confusion matrix:")
    print(conf_mat)
    
    
    
model = create_model()   
test_labels, x_test = create_labels_as_list()   
x_t, y_t = load_test_array_with_labels(test_labels, x_test)
pred_label, y_arr = predictions(model, x_t, test_labels)
pred_label.astype(np.int32)
y_arr.astype(np.int32)
print("Predicted labels")
print(pred_label)
print('Actual Labels')
print(y_arr)
conf_matrix(y_arr, pred_label)



    
    
