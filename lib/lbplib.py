#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBP implementations using Python and Tensorflow
py_lbp : lbp using Pyton 
tf_lbp : lbp using TensorFlow


Please cite below works if you find it useful:
Akgun, D., A TensorFlow implementation of Local Binary Patterns Transform. MANAS Journal of Engineering, 9(1), 15-21. DOI:10.51354/mjen.822630
Akgun, Devrim. "A PyTorch Operations Based Approach for Computing Local Binary Patterns." U. Porto Journal of Engineering 7.4 (2021): 61-69. https://doi.org/10.24840/2183-6493_007.004_0005
"""

import tensorflow as tf
import numpy as np


def py_lbp(Im):
    # Local Binary Patterns 
    rows=Im.shape[0]
    cols=Im.shape[1]
    L=np.zeros((rows,cols),dtype='uint8')
    I=np.zeros((rows+2,cols+2),dtype='uint8')
    
    #Zero padding
    I[1:rows+1,1:cols+1]=Im
    
    #Select center pixel
    for i in range(1,rows+1):
        for j in range(1,cols+1):
            #Compute LBP transform
            L[i-1,j-1]=\
            ( I[i-1,j]  >= I[i,j] )*1+\
            ( I[i-1,j+1]>= I[i,j] )*2+\
            ( I[i,j+1]  >= I[i,j] )*4+\
            ( I[i+1,j+1]>= I[i,j] )*8+\
            ( I[i+1,j]  >= I[i,j] )*16+\
            ( I[i+1,j-1]>= I[i,j] )*32+\
            ( I[i,j-1]  >= I[i,j] )*64+\
            ( I[i-1,j-1]>= I[i,j] )*128;  
    
    return L


def tf_lbp(Im):    
    paddings = tf.constant([[0,0],[1, 1], [1, 1]])
    Im=tf.pad(Im, paddings,"CONSTANT")        
    M=Im.shape [1]
    N=Im.shape [2]      
   
    # Select the pixels of masks in the form of matrices
    y00=Im[:,0:M-2, 0:N-2]
    y01=Im[:,0:M-2, 1:N-1]
    y02=Im[:,0:M-2, 2:N  ]
    #     
    y10=Im[:,1:M-1, 0:N-2]
    y11=Im[:,1:M-1, 1:N-1]
    y12=Im[:,1:M-1, 2:N  ]
    #
    y20=Im[:,2:M, 0:N-2]
    y21=Im[:,2:M, 1:N-1]
    y22=Im[:,2:M, 2:N ]  
    
    
    # y00  y01  y02
    # y10  y11  y12
    # y20  y21  y22
    
    # Comparisons 
    # 1 -----------------------------------------        
    g=tf.greater_equal(y01,y11) 
    z=tf.multiply(tf.cast(g,dtype='uint8'), 
                  tf.constant(1,dtype='uint8') )      
    # 2 -----------------------------------------
    g=tf.greater_equal(y02,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(2,dtype='uint8') )
    z =tf.add(z,tmp)              
    # 3 -----------------------------------------
    g=tf.greater_equal(y12,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(4,dtype='uint8') )
    z =tf.add(z,tmp)
    # 4 -----------------------------------------
    g=tf.greater_equal(y22,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(8,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 5 -----------------------------------------
    g=tf.greater_equal(y21,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(16,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 6 -----------------------------------------
    g=tf.greater_equal(y20,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(32,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 7 -----------------------------------------
    g=tf.greater_equal(y10,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(64,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 8 -----------------------------------------
    g=tf.greater_equal(y00,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(128,dtype='uint8') )
    z =tf.add(z,tmp)  
    #--------------------------------------------
    return tf.cast(z,dtype=tf.uint8)
