#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:41:01 2017

@author: wujialin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:37:41 2017

@author: root
"""

import numpy as np 
import random
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import os
import shutil # delect a file

class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        self.X = np.zeros((self.pN,self.dim)) # all the coordination and vertical of partical
        self.V = np.zeros((self.pN,self.dim))
        self.pbest = np.zeros((self.pN,self.dim)) #best coordination for individual and global
        self.gbest = np.zeros((1,self.dim))
        self.p_fit = np.zeros(self.pN)
        self.fit = 1e10
        
        # target functon
    def Brenna_img(self,img_arr):
        img_arr=img_arr
        m=0
        w,h=img_arr.shape
        for i in range(w):
            for j in range(h-2):
                m=m+np.square(img_arr[i,j+2]-img_arr[i,j])
        return m
            
    def Reblur_image(self,img1, img2, img3):
        Gradient1 = self.Brenna_img(img1)
        Gradient2 = self.Brenna_img(img2)#filter image
        Gradient3 = self.Brenna_img(img3)#filter image
        
        alpha =(Gradient2 -  Gradient3) / (Gradient1 - Gradient2)
        return alpha
            
#    def function(self, x):
#        sum_f = 0
#        length = len(x)
#        x = x**2
#        for i in range(length):
#            sum_f += x[i]
#        return sum_f
    
    ### initial the set
    def init_Population(self):
        class_path = "/home/huawei/DOE_DataSets/Logo-Victoria-0927-RAW/Victoria-AL00C-Lan/OK/H/"
        for i in range(self.pN):
            
            for j in range(self.dim):
                
                self.X[i][j] = random.uniform(0,1)
                self.V[i][j] = random.uniform(0,1)
            self.pbest[i] = self.X[i]
            # take photoes by using the parameters
            img = Image.open(img_name).convert('L')
            img_arr=np.array(img)
            img2=img.filter(ImageFilter.GaussianBlur(radius=2))
            img3=img2.filter(ImageFilter.GaussianBlur(radius=2))
            img2_arr=np.array(img2)
            img3_arr=np.array(img3)
            
            tmp = self.Reblur_image(img_arr, img2_arr, img3_arr)
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]
            
    ## update the position
    
    def iterator(self):
        fitness = []
        
        for t in range(self.max_iter):
            # slow down the time
            class_path = "/home/huawei/DOE_DataSets/Logo-Victoria-0927-RAW/Victoria-AL00C-Lan/OK/H/"
            for i in range(self.pN):
                #take photoes by using X[i]
                img = Image.open(img_name).convert('L')
                img_arr=np.array(img)
                img2=img.filter(ImageFilter.GaussianBlur(radius=2))
                img3=img2.filter(ImageFilter.GaussianBlur(radius=2))
                img2_arr=np.array(img2)
                img3_arr=np.array(img3)
                
                temp = self.Reblur_image(img_arr, img2_arr, img3_arr)
                
#                temp = self.function(self.X[i])
                if(temp < self.p_fit[i]):
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]

                    if(self.p_fit[i] < self.fit):
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
                
            for i in range(self.pN):
                self.V[i] = self.w *self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + self.c2 *self.r2*(self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            print(self.fit)
#            shutil.rmtree(class_path)
#            os.mkdir(class_path)
            # choose the camera and take photoes by using the parameters X and producing images in path class_path
            
        return fitness
    
    ## execute the program
#img = Image.open("/home/huawei/DOE_DataSets/Logo-Victoria-0927-RAW/Victoria-AL00C-Lan/OK/H/1_170922085004_OK.bmp").convert('L')
#img_arr=np.array(img)
#
##img2_arr=filters.gaussian_filter(img_arr,13.5)
#img2=img.filter(ImageFilter.GaussianBlur(radius=2))
#img3=img2.filter(ImageFilter.GaussianBlur(radius=2))
#img2_arr=np.array(img2)
#img3_arr=np.array(img3)



my_pso = PSO(pN = 30, dim = 5, max_iter = 100)
my_pso.init_Population()
fitness = my_pso.iterator()
plt.figure(1)
plt.title("Figure1")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size = 14)
t = np.array(fitness)
plt.plot(t,fitness,color = 'b',linewidth = 3)
plt.show()
    