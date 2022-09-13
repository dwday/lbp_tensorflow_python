#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:09:51 2022

@author: ubuntu
"""
import numpy as np
from skimage import feature


A=np.zeros(shape=(5,5),dtype='uint8')

B=np.array([[1,1,1],
          [1,2,5],
          [1,1,1]])

A[1:4,1:4]=B

# A=np.array([[1,1,1],
#           [1,2,5],
#           [1,1,1]])




p=feature.local_binary_pattern(A, 8, 1, method='default')

print(p)