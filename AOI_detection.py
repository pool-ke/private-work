#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 02:57:43 2018

@author: root
"""

import os
import skimage
import numpy as np
import cv2
import copy


file_path_OK_pro='AOIDetection_pro/OK/'
file_path_NG_pro='AOIDetection_pro/NG/'
file_path_template='AOIDetection_pro/OK/SC0402-BGA_37.png'
file_path_test='AOIDetection_pro/NG/SC0402-BGA_284.png'

if __name__=='__main__':
    img_test=cv2.imread(file_path_test)
    img_ori=cv2.imread(file_path_template)
    img_template=copy.copy(img_ori[8:45,20:38])
    ROI_H=37
    ROI_W=18
#    cv2.imshow("img_template",img_template)
    
    methods=['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
    meth=methods[0]
    method=eval(meth)
#    res=cv2.matchTemplate(img_ori,img_template,method)
    res=cv2.matchTemplate(img_test,img_template,method)
    print (res.shape)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        print (1)
        top_left=min_loc
        target_val=min_val
    else:
        print (2)
        top_left=max_loc
        target_val=max_val
    print (top_left)
    print (target_val)
    img_res=copy.copy(img_test[top_left[1]:top_left[1]+ROI_H,top_left[0]:top_left[0]+ROI_W])
    cv2.imshow("img_res",img_res)
#    cv2.imshow("img_pro",img_pro)
    cv2.waitKey(0)
    cv2.destroyAllWindows()