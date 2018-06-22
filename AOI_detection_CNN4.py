#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:33:18 2018

@author: root
"""

import os
import skimage
import numpy as np
import cv2
import copy
import CNN_ModelRead2 as CNN_MR

#share_path='/home/huawei/myfile/code_python/KE/CV_test/AOIDetection0620/'
share_path='/home/huawei/myfile/code_python/KE/CV_test/'
#share_path='/home/huawei/myfile/code_python/KE/CV_test/AOIDetection_pro/'
#folder_name='test_NG/'
#folder_name='NG_SC0402-BGA0/'
#folder_name='OK_SC0402-BGA/'
folder_name='AOI_detection_brightness2/'
#folder_name='OK/'

save_path='/home/huawei/myfile/code_python/KE/CV_test/AOIDetection00621/'
save_path1='/home/huawei/myfile/code_python/KE/CV_test/OKDetection0621/'
save_path2='/home/huawei/myfile/code_python/KE/CV_test/NGDetection0621/'

img_ori_size1=70
img_ori_size2=45

img_size1=64
img_size2=32

n_classes=2
batch_size=30

res_dict={}


#SC0402_BGA_Files=[]
#image_path=share_path+folder_name
#AllFiles=os.listdir(image_path)
#
#for i in range(len(AllFiles)):
#    if(AllFiles[i].split('_')[0]=='SC0402-BGA'):
#        SC0402_BGA_Files.append(AllFiles[i])
        
image_path=share_path+folder_name
SC0402_BGA_Files=os.listdir(image_path)
    

X_test=np.zeros((len(SC0402_BGA_Files),img_size1*img_size2))
Y_test=np.zeros((len(SC0402_BGA_Files),n_classes))


for i in range(len(SC0402_BGA_Files)):
    path_read=image_path+SC0402_BGA_Files[i]
    print(path_read)
    path_write=save_path+SC0402_BGA_Files[i]
    img_temp=cv2.imread(path_read,0)
    H=img_temp.shape[0]
    if (H==img_ori_size1):
        img_res=img_temp
    elif(H==img_ori_size2):
        img_res=np.rot90(img_temp)

#    img_dect=copy.copy(img_res[3:67,6:38])
    img_dect=copy.copy(img_res[int(int(img_ori_size1/2)-img_size1/2):int(int(img_ori_size1/2)+img_size1/2),int(int(img_ori_size2/2)-img_size2/2):int(int(img_ori_size2/2)+img_size2/2)])
    X_test[i]=img_dect.flatten()
    cv2.imwrite(path_write,img_dect)

index=0
sum_count=len(SC0402_BGA_Files)
count=len(SC0402_BGA_Files)
while (count>0):
    if (count>=batch_size):
        Y_test_temp=CNN_MR.test_model(X_test[batch_size*index:batch_size*index+batch_size])
        print (Y_test_temp.shape)
        Y_test[batch_size*index:batch_size*index+batch_size]=Y_test_temp
        count=count-batch_size
        index=index+1
    else:
        Y_test_temp=CNN_MR.test_model(X_test[sum_count-count:sum_count])
        print (Y_test_temp.shape)
        Y_test[sum_count-count:sum_count]=Y_test_temp
        count=0

print(Y_test)
True_Label=np.argmax(Y_test,axis=1)

for i in range(len(SC0402_BGA_Files)):
    res_dict[SC0402_BGA_Files[i]]=True_Label[i]
    
NG_dict={k: v for k,v in res_dict.items() if v==1}
print (NG_dict)
print (len(NG_dict))

for k,v in res_dict.items():
    img_path=save_path+k
    if v==0:
        copy_path=save_path1+k
    else:
        copy_path=save_path2+k
    
    str_exec="cp "+img_path+" "+copy_path
    os.system(str_exec)
print("................................")        
for i in range(len(SC0402_BGA_Files)):
    if(True_Label[i]==1):
        print(Y_test[i])

print("................................")        
for i in range(len(SC0402_BGA_Files)):
    if(True_Label[i]==0):
        print(Y_test[i])
    
    
