#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test file for rgb and grascale outputs  
Please cite below works if you find it useful:
Akgun, D., A TensorFlow implementation of Local Binary Patterns Transform. MANAS Journal of Engineering, 9(1), 15-21. DOI:10.51354/mjen.822630
Akgun, Devrim. "A PyTorch Operations Based Approach for Computing Local Binary Patterns." U. Porto Journal of Engineering 7.4 (2021): 61-69. https://doi.org/10.24840/2183-6493_007.004_0005
"""

from lib.lbplib import py_lbp,tf_lbp
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import time

# test image
path='images/ILSVRC2012_test_00000689.jpg'

#Gray input--------------------------------------------------------------------
img_gray =image.load_img(path,target_size=(224,224,1),color_mode='grayscale')
img_gray=image.img_to_array(img_gray)
[Rows,Cols,nChannel]=img_gray.shape

lbp_gray = tf_lbp(img_gray.reshape(1,Rows,Cols).astype('uint8')).numpy()


# RGB input -------------------------------------------------------------------
img_org =image.load_img(path,target_size=(224,224,3),color_mode='rgb')
img_org=image.img_to_array(img_org)
[Rows,Cols,nChannel]=img_org.shape

# Allocation for the rgb output 
lbp_rgb=img_org.copy()
lbp_rgb[:,:,0] = tf_lbp(img_org[:,:,0].reshape(1,Rows,Cols).astype('uint8')).numpy()
lbp_rgb[:,:,1] = tf_lbp(img_org[:,:,1].reshape(1,Rows,Cols).astype('uint8')).numpy()
lbp_rgb[:,:,2] = tf_lbp(img_org[:,:,2].reshape(1,Rows,Cols).astype('uint8')).numpy()


# Show input and output images-------------------------------------------------
plt.figure(1)
plt.imshow(img_org.astype('uint8'))
plt.title('Input file')    
plt.figure(2)
plt.imshow(lbp_rgb.astype('uint8') )
plt.title('RGB output')
plt.figure(3)
plt.imshow(lbp_gray.reshape(Rows,Cols).astype('uint8') ,cmap='gray' )
plt.title('Grayscale output')
