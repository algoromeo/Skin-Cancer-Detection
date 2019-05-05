#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:01:40 2019

@author: bhargavdesai
"""

from time import time
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.backend import tensorflow_backend as K
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import load_model


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline"""
    
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True
                
                
def train_stage_2():
    label_dict = dict([('akiec', 0), ('bcc', 1), ('mel', 2), ('bkl', 3), ('df', 4), ('nv', 5), ('vasc', 6)]) 
    train_stage_2 = []
    df = pd.read_csv("/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
    for img_id in os.listdir('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Train Stage 2'):
        if (img_id=='.DS_Store'):
                print("---------------Found .DS_Store File breaking the loop-------------------------------------")
                break
        img_id_wo_suffix = extract_suffix_if_any(img_id)
        print(img_id_wo_suffix)
        dx_value = pd.Series.tolist(df.loc[df['image_id'] == img_id_wo_suffix, 'dx'])
        dx_value = ''.join(dx_value)
        print(dx_value)
        label = label_dict[dx_value]
        print(label)
        train_stage_2.append((img_id, label))
    return train_stage_2


def extract_suffix_if_any(img_id):
    img_id_suffix = img_id[12:]
    for suffix in ['_AC.jpg', '_AS.jpg', '_AV.jpg', '_AL.jpg', '_ASH.jpg', '_ALSH.jpg', '_ALC.jpg', '_ALS.jpg', '_AVSH.jpg', '_AVC.jpg', '_AVS.jpg']:
        if (img_id_suffix == suffix):
            img_id = img_id.replace(suffix, '')
        else:
            continue
    img_id = img_id.replace('.jpg', '')    
    return img_id   


def train_stage_2_to_df(train_stage_2):
    labels = ['Images', 'Class']
    train_stage_2_df = pd.DataFrame.from_records(train_stage_2, columns = labels)
    return train_stage_2_df  

train_stage_2 = train_stage_2()
train_stage_2_df = train_stage_2_to_df(train_stage_2)
train_stage_2_df.to_csv('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Processed (.csv) Files/train_categorical_stage_2_df.csv', index=False)


def initialise_training_set_with_labels():
    train_stage_1_df = pd.read_csv('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Processed (.csv) Files/train_categorical_stage_2_df.csv')
    print(train_stage_1_df.head())
    x_train_list = pd.Series.tolist(train_stage_1_df['Images'])
    y_train_list = pd.Series.tolist(train_stage_1_df['Class'])
    return x_train_list, y_train_list

def one_hot(y_train_list):
    y_train = to_categorical(y_train_list)
    return y_train

def ndarray_from_list(x_train_list, y_train_list):
    size = len(y_train_list)
    x_train = np.empty((size, 128, 171, 3))
    y_train = one_hot(y_train_list)
    for idx, img_id in enumerate(x_train_list,0):
        if (img_id=='.DS_Store'):
            print("---------------Found .DS_Store File breaking the loop-------------------------------------")
            break
        image = cv2.imread('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Train Stage 2'+'/'+img_id, 1) 
        x_train[idx, :, :, :] = image
    print("done with converting to array")
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train


x_train_list, y_train_list = initialise_training_set_with_labels()
x_train, y_train = ndarray_from_list(x_train_list, y_train_list)


with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    K.set_session(sess) 
    model = load_model('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Models/final_categorical_stage_1_lr=0.000075_trained-on-mac.h5')
    model.summary()
    x = model.layers[1].output
    x = Conv2D(32, kernel_size =(3,3), activation='relu', name = 'model_2_conv2d')(x)
    x = MaxPool2D(pool_size = (2, 2), name = 'model_2_maxpool')(x)
    x = Conv2D(16, kernel_size =(3,3), activation='relu', name = 'model_2_conv2d_2')(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    preds = Dense(7,activation='softmax')(x) 
    model_2 = Model(inputs=model.input, outputs=preds)
    model_2.summary()
    for layer in model_2.layers[:2]:
        layer.trainable=False
    opt = Adam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_2.summary()
    tensorboard = TensorBoard(log_dir='/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Logs/Stage 2'.format(time()))
    model_2.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])    
    model_2.fit(x= x_train,y = y_train, epochs= 200, batch_size = 32, callbacks=[TerminateOnBaseline(monitor='acc', baseline=0.89), tensorboard])   
    model_2.save('final_cat_stage_2_lr=95e-6_trained-on-mac.h5')
    del model_2
    print("-------End of Line-------------")
K.clear_session()      