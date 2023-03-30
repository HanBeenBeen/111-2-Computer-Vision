# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:27:53 2023

@author: Young_BoyFriend
"""

import numpy as np
import cv2

sigma_s, sigma_r = 2 ,0.1

sigma_r = sigma_r
sigma_s = sigma_s
wndw_size = 6*sigma_s+1
pad_w = 3*sigma_s
    
img = cv2.imread('./testdata/1.png')
guidance = cv2.imread('./testdata/1.png',0)

BORDER_TYPE = cv2.BORDER_REFLECT
padded_img = cv2.copyMakeBorder(img, pad_w, pad_w, pad_w, pad_w, BORDER_TYPE).astype(np.int32)
padded_guidance = cv2.copyMakeBorder(guidance, pad_w, pad_w, pad_w, pad_w, BORDER_TYPE).astype(np.int32)

padded_guidance = cv2.normalize(padded_guidance, 
                                None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)

padded_img = cv2.normalize(padded_img, None, alpha=0,beta=1,        norm_type=cv2.NORM_MINMAX)


### TODO ###
#Spacial kernel table
#Index by abs(Tp-Tq)
x, y = np.meshgrid(np.arange(2 * pad_w + 1) - pad_w
               , np.arange(2 * pad_w + 1) - pad_w)

Spacial_kernel = np.exp(-(x * x + y * y) 
              / (2 * sigma_s ** 2))

#Range kernel table
#-(Tp - Tq)^2 / (2sigma_r^2)
#Tp: Totle of region ixel value
#Tq: centerixel value
#Index by abs(Tp-Tq)
Range_kernel = np.exp(-(np.arange(0,1+1/255,1/255)**2)/ 
                      (2 * sigma_r ** 2))


#Determine the img format
output = np.zeros_like(img)

if padded_img.ndim == 3 and padded_guidance.ndim == 2 :
    spliding = int(wndw_size)
    center = int((spliding-1) /2)
    
    Tp_box = np.lib.stride_tricks.sliding_window_view(padded_guidance, 
                                             (spliding, spliding))
    
    img_box = np.lib.stride_tricks.sliding_window_view(
    padded_img, (spliding, spliding),(0,1))
    
    #Tp_box = img_sliding
    Tq_boxCenter = Tp_box[:,:,center,center]
    Tq_boxCenter = np.expand_dims(np.expand_dims(Tq_boxCenter, axis=2), axis=3)
    
    wgt = Range_kernel[np.abs(Tp_box - Tq_boxCenter)] * Spacial_kernel
    

    
    output = np.sum(np.array([wgt * img_box[:,:,d] for d in range(3)]), axis = (0,3,4))/np.sum(wgt, axis = (2,3))

else:
    Range_kernel = np.exp(-np.arange(256) * np.arange(256) / (2*sigma_r**2))
    Range_kernel = np.vstack([Range_kernel,
                              Range_kernel,
                              Range_kernel])
    
    spliding = wndw_size
    center = (spliding-1) /2
    
    Tp_box = [np.lib.stride_tricks.sliding_window_view(img, 
                                             (spliding, spliding, 1))]
    #Tq_box = Tp_box[:,:,center,center]
    #Tq_box = np.expand_dims(np.expand_dims(Tq_box, axis=2), axis=3)
            
            
            
            
            
        
        
r = np.clip(output, 0, 255).astype(np.uint8)

r2 = cv2.normalize(output,None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    

    
    