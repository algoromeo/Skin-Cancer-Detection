#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:23:39 2019

@author: bhargavdesai
"""
from time import time
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.backend import tensorflow_backend as K
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import TensorBoard


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
                
                
def train_stage_1():
    label_dict = dict([('akiec', 0), ('bcc', 1), ('mel', 2), ('bkl', 3), ('df', 4), ('nv', 5), ('vasc', 6)]) 
    train_stage_1 = []
    df = pd.read_csv("/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
    for img_id in os.listdir('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Train Stage 1'):
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
        train_stage_1.append((img_id, label))
    return train_stage_1


def extract_suffix_if_any(img_id):
    img_id_suffix = img_id[12:]
    for suffix in ['_AC.jpg', '_AS.jpg', '_AV.jpg', '_AL.jpg', '_ASH.jpg', '_ALSH.jpg', '_ALC.jpg', '_ALS.jpg', '_AVSH.jpg', '_AVC.jpg', '_AVS.jpg']:
        if (img_id_suffix == suffix):
            img_id = img_id.replace(suffix, '')
        else:
            continue
    img_id = img_id.replace('.jpg', '')    
    return img_id   


def train_stage_1_to_df(train_stage_1):
    labels = ['Images', 'Class']
    train_stage_1_df = pd.DataFrame.from_records(train_stage_1, columns = labels)
    return train_stage_1_df  

train_stage_1 = train_stage_1()
train_stage_1_df = train_stage_1_to_df(train_stage_1)
train_stage_1_df.to_csv('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Processed (.csv) Files/train_categorical_stage_1_df.csv', index=False)

def initialise_training_set_with_labels():
    train_stage_1_df = pd.read_csv('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Processed (.csv) Files/train_categorical_stage_1_df.csv')
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
        image = cv2.imread('/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Train Stage 1'+'/'+img_id, 1) 
        x_train[idx, :, :, :] = image
    print("done with converting to array")
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train

def Model():
    input_shape = (128, 171, 3)
    num_classes = 7
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding = 'Same'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',padding = 'Same'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model 

def train_model(x_train, y_train, model):
    opt = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.summary()
    model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])    
    tensorboard = TensorBoard(log_dir='/Users/bhargavdesai/Desktop/Projects/Skin Cancer Detection/Dataset/skin-cancer-mnist-ham10000/Final 2/Logs/Stage 1'.format(time()))
    model.fit(x= x_train,y = y_train, epochs= 500, batch_size = 32, callbacks=[TerminateOnBaseline(monitor='acc', baseline=0.89), tensorboard])   
    model.save('final_categorical_stage_1_lr=0.000075_trained-on-mac.h5')
    del model
    print("-------End of Line-------------")
    
    
x_train_list, y_train_list = initialise_training_set_with_labels()
x_train, y_train = ndarray_from_list(x_train_list, y_train_list)


with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    K.set_session(sess) 
    model = Model()
    train_model(x_train, y_train, model) 


K.clear_session()    
