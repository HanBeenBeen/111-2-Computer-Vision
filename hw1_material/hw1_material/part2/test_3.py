# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 03:06:37 2023

@author: Young_BoyFriend
"""

import numpy as np
import cv2

sigma_s, sigma_r = 2 ,0.1

sigma_r = sigma_r
sigma_s = sigma_s
wndw_size = 6*sigma_s+1
pad_w = 3*sigma_s

pa ="BORDER_TYPE"

img = cv2.imread('./testdata/1.png')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

bilateral = cv2.bilateralFilter(img, 15, 75, 75)

