#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance evaluation of lbp using TensorFlow

"""
from lib.lbplib import py_lbp,tf_lbp
import numpy as np
import time

  
# Test data with random numbers
Rows=128
Cols=128
img_org=np.random.randint(0,255,(Rows,Cols))

 
[Rows,Cols]=img_org.shape
img_lbp=img_org.reshape(1,Rows,Cols)

# Test Python implementation 
start_time   = time.time()    
ypy  = py_lbp(img_lbp[0,:,:].astype('uint8'))
elapsed_py = time.time() - start_time    
print('python elapsed_time=',elapsed_py)

# Test TensorFlow
start_time   = time.time()
ytf          = tf_lbp(img_lbp.astype('uint8')).numpy()
elapsed_tf = time.time() - start_time
print('tensor flow elapsed_time=',elapsed_tf)

#Check if there is an error between TensorFlow and Python implementations
print('error=',np.sum(ytf[0,:,:]-ypy))

print('python:')
print(ypy)
print('TensorFlow:')
print(ytf[0,:,:])    
