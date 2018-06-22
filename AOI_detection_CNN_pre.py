#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 00:57:29 2018

@author: root
"""

import numpy as np
import cv2
import skimage.io as io
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

classes=['OK','NG']

numOfTrain=200
numOfTest=160
np.random.seed(3366)
index=np.arange(1000)
np.random.shuffle(index)
img_size1=64
img_size2=32

numOfTest_OK=3000
numOfTest_NG=160

#x_train=np.zeros((numOfTrain*len(classes),img_size1*img_size2))
#y_train=np.zeros(numOfTrain*len(classes),dtype='int8')
#x_test=np.zeros((numOfTest*len(classes),img_size1*img_size2))
#y_test=np.zeros(numOfTest*len(classes),dtype='int8')

x_train=np.zeros((numOfTrain*len(classes),img_size1*img_size2))
y_train=np.zeros(numOfTrain*len(classes),dtype='int8')
x_test=np.zeros((numOfTest_OK+numOfTest_NG,img_size1*img_size2))
y_test=np.zeros((numOfTest_OK+numOfTest_NG),dtype='int8')


for i in range(len(classes)):
    dir_input='AOIDetection_da/'+classes[i]+'/*.png'
    coll=io.ImageCollection(dir_input)
    print (coll.files)
    np.random.shuffle(coll.files)
    if i==0:
        numOfTest=numOfTest_OK
    else:
        numOfTest=numOfTest_NG
    for j in range(numOfTrain+numOfTest):
#    for j in range(1):
        img=cv2.imread(coll.files[j],0)
        img_flatted=img.flatten()
        if(j<numOfTrain):
            x_train[j+numOfTrain*i]=img_flatted
            y_train[j+numOfTrain*i]=i
        else:
            x_test[j-numOfTrain+numOfTest*i]=img_flatted
            y_test[j-numOfTrain+numOfTest*i]=i
print (x_train.shape)
print (y_train.shape)  
print (x_test.shape)  
print (x_test.shape)              
index_train=np.arange(numOfTrain*len(classes))
#print (index_train)
np.random.shuffle(index_train)
#print (index_train)
index_test=np.arange(numOfTest_OK+numOfTest_NG)
np.random.shuffle(index_test)
x_train_shuffled=np.zeros((numOfTrain*len(classes),img_size1*img_size2))
x_test_shuffled=np.zeros((numOfTest_OK+numOfTest_NG,img_size1*img_size2))
y_train_oneHot=np.zeros((numOfTrain*len(classes),len(classes)))
y_test_oneHot=np.zeros((numOfTest_OK+numOfTest_NG,len(classes)))