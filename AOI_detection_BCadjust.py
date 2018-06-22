#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 06:51:52 2018

@author: root
"""

import numpy as np
import cv2
import os
from skimage import filters,io, measure, color
import skimage.morphology as sm
import random
from PIL import ImageEnhance
from PIL import Image
import matplotlib.pyplot as plt


def fun_CONTRAST(img):
    en_2 = ImageEnhance.Contrast(img)
    contrast = 1.5
    img_temp = en_2.enhance(contrast)
    print(22)
    return img_temp

def PIL2CV(img):
#    img1=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img1=np.asarray(img)
    return img1

def CV2PIL(img):
#    img1=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    img1=Image.fromarray(img)
    return img1

def brightness_enhance(img,para_brightness):
    enh_bri=ImageEnhance.Brightness(img)
    brightness=para_brightness
    image_brightened=enh_bri.enhance(brightness)
    return image_brightened
    
def color_enhance(img,para_color):
    enh_col=ImageEnhance.Color(img)
    color1=para_color
    image_colored=enh_col.enhance(color1)
    return image_colored
    
def contrast_enhance(img,para_contrast):
    enh_con=ImageEnhance.Contrast(img)
    contrast=para_contrast
    image_contrasted=enh_con.enhance(contrast)
    return image_contrasted
    
def sharpness_enhance(img,para_sharpness):
    enh_sha=ImageEnhance.Sharpness(img)
    sharpness =para_sharpness
    image_sharped=enh_sha.enhance(sharpness)
    return image_sharped

    
def calcAndDrawHist(image,color):
    hist=cv2.calcHist([image],[0],None,[256],[0.0,255.0])
    minVal,maxVal,minLoc,maxLoc=cv2.minMaxLoc(hist)
    histImg=np.zeros([256,256,3],np.uint8)
    hpt=int(0.9*256)
    for h in range(256):
        intensity=int(hist[h]*hpt/maxVal)
        cv2.line(histImg,(h,256),(h,256-intensity),color)
        
    return histImg

def Histequalize(img):
#    img=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    eq=cv2.equalizeHist(img)
    img2=np.hstack([img,eq])
    return img2

def LaplaceImageEnhancement(img):
    kernel=np.array([[-1,-1,-1],
                     [-1,8,-1],
                     [-1,-1,-1]])
#    kernel=np.array([[0,-1,0],
#                     [-1,4,-1],
#                     [0,-1,0]])
#    kernel=np.array([[0,1,0],
#                     [1,-4,1],
#                     [0,1,0]])
#    kernel=np.array([[1,1,1],
#                     [1,-8,1],
#                     [1,1,1]])
    img_temp=cv2.filter2D(img_ori,-1,kernel)
    img_res=img_ori+img_temp
    Hist_img_res=calcAndDrawHist(img_res,[255,255,255])
    return img_res,Hist_img_res

def HistequlizeImageEnhancement(img):
    img_res=Histequalize(img_ori)
    print (img_ori.shape)
    Hist_img_res=calcAndDrawHist(img_res,[255,255,255])
    return img_res,Hist_img_res

def ContrastImageEnhancement(img):
    image1=CV2PIL(img_ori)
    image2=contrast_enhance(image1,2)
    img_res=PIL2CV(image2)
    Hist_img_res=calcAndDrawHist(img_res,[255,255,255])
    return img_res,Hist_img_res

def HighExposuretimePro(img):
    img_temp=255-img
    img_res=np.zeros([img.shape[0],img.shape[1] ],np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]>img_temp[i,j]:
                img_res[i,j]=img_temp[i,j]
            else:
                img_res[i,j]=img[i,j]
    Hist_img_res=calcAndDrawHist(img_res,[255,255,255])
    return img_res,Hist_img_res

def HighContrastReserve(img):
    para=3
    img_temp=cv2.GaussianBlur(img,(5,5),0)
    img_res=img+para*(img-img_temp)
    Hist_img_res=calcAndDrawHist(img_res,[255,255,255])
    return img_res,Hist_img_res

def SobelImageEnhancement(img):
    x_temp=cv2.Sobel(img,cv2.CV_16S,1,0)
    y_temp=cv2.Sobel(img,cv2.CV_16S,0,1)
    img_temg_x=cv2.convertScaleAbs(x_temp)
    img_temg_y=cv2.convertScaleAbs(y_temp)
    img_temp=cv2.addWeighted(img_temg_x,0.5,img_temg_y,0.5,0)
    img_res=img+img_temp
    Hist_img_res=calcAndDrawHist(img_res,[255,255,255])
    return img_res,Hist_img_res


    
    
    
'''whether show the image'''
showimage=True
'''whether save the image'''
saveimage=True

file_path="/home/huawei/myfile/code_python/KE/CV_test/image_daguang/"
file_path2="/home/huawei/myfile/code_python/KE/CV_test/AOI_detection_brightness2/"
file_name="020QYR4MBC007627_SC0402-BGA_64.bmp"
if __name__=='__main__':
#    image1=Image.open(file_path)
#    print (image1.shape)
#    image1.show()
    read_path=file_path+file_name
    img_ori=cv2.imread(read_path,0)
    cv2.imshow("img_ori",img_ori)
#    Hist_img_ori=calcAndDrawHist(img_ori,[255,255,255])
#    cv2.imshow("Hist_img",Hist_img_ori)
    
    
##    CalcHist(img_ori)
#    #EqulizeHist Transform
#    img_res=Histequalize(img_ori)
#    cv2.imshow("img_res",img_res)
#    Hist_img_res=calcAndDrawHist(img_res,[255,255,255])
#    img_res,Hist_img_res=HistequlizeImageEnhancement(img_ori)
##    CalcHist(img_res)    
#    #Image Enhance class
    for i in np.arange(0.1,2,0.1):
        print (i)
        save_path=file_path2+'B'+str(i)+"_"+file_name
        image1=CV2PIL(img_ori)
        image2=brightness_enhance(image1,i)
#        image2=contrast_enhance(image1,i)
        img_res=PIL2CV(image2)
#        cv2.imshow("img_res",img_res)
        cv2.imwrite(save_path,img_res)
    #Laplace image enhancement
#    img_res,Hist_img_res=ContrastImageEnhancement(img_ori)
#    img_res,Hist_img_res=HighExposuretimePro(img_ori)
#    img_res,Hist_img_res=HighContrastReserve(img_ori)
#    img_res,Hist_img_res=SobelImageEnhancement(img_ori)
    
#    
#    mix2=np.zeros([img_ori.shape[0],img_ori.shape[1]*2],np.uint8)
#    mix2[0:img_ori.shape[0],0:img_ori.shape[1]]=img_ori
#    mix2[0:img_ori.shape[0],img_ori.shape[1]:28*img_ori.shape[1]]=img_res
##    cv2.imwrite('result20180530/mix2.bmp',mix2)
#    cv2.imshow("img_mix",mix2)
#
#    mixHist=np.zeros([Hist_img_ori.shape[0],Hist_img_ori.shape[1]*2,3],np.uint8)
#    mixHist[0:Hist_img_ori.shape[0],0:Hist_img_ori.shape[1]]=Hist_img_ori
#    mixHist[0:Hist_img_ori.shape[0],Hist_img_ori.shape[1]:2*Hist_img_ori.shape[1]]=Hist_img_res
##    cv2.imwrite('result20180530/Hist_img.bmp',mixHist)
#    cv2.imshow("img_mix_Hist",mixHist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#    img=cv2.imread(file_path)
#    img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    img2=fun_CONTRAST(img)
#    
#    if showimage:
#        cv2.imshow("img_ori",img)
#        cv2.imshow("img_pro",img2)
#        
#    if showimage:
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()