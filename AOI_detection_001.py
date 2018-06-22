#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 07:05:33 2018

@author: root
"""

import CNN_ModelRead as CNN
import cv2


file_path="AOIDetection_testimage/3_1_SC0402-BGA_261.png"
img=cv2.imread(file_path,0)
label=CNN.test_model(img)
print (label)