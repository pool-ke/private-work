#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 06:56:41 2018

@author: root
"""

import numpy as np
import cv2
import skimage.io as io
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

np.random.seed(44)
index_train=np.arange(50)
print (index_train)
#print (index_train)
np.random.shuffle(index_train)
print (index_train)

np.random.seed(48)
index_train1=np.arange(50)
print (index_train1)
#print (index_train)
np.random.shuffle(index_train1)
print (index_train1)
#rate_OK=0.2
#rate_NG=0.8
#
#classes=['OK','NG']
#
#for i in range(len(classes)):
#    if i ==0:
#        rate=rate_OK
#    else:
#        rate=rate_NG
#    for x in range(4):
#        for y in range(5):
##        dir_input='AOIDetection_da/'+classes[i]+'/0_0*.png'
#            dir_input='AOIDetection_da/'+classes[i]+'/'+str(x)+'_'+str(y)+'*.png'
#            print (dir_input)
#            dir_output1='AOIDetection_da2/train_set/'+classes[i]+'/'
#            dir_output2='AOIDetection_da2/test_set/'+classes[i]+'/'
#            coll=io.ImageCollection(dir_input)
#            for j in range(len(coll.files)):
#                img=cv2.imread(coll.files[j])
#                if j<rate*(len(coll.files)):
#                    file_save=dir_output1+coll.files[j].split('/')[-1]
#                    cv2.imwrite(file_save,img)
#                else:
#                    file_save=dir_output2+coll.files[j].split('/')[-1]
#                    cv2.imwrite(file_save,img)
    