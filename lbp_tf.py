# -*- coding: utf-8 -*-
"""
Program to test TensorFlow implementation of lbp transform

for details:
AKGÜN, D. "A TensorFlow implementation of Local Binary Patterns Transform." 
MANAS Journal of Engineering 9.1: 15-21, 2021, doi.org/10.51354/mjen.822630
https://dergipark.org.tr/en/download/article-file/1384888
"""
import numpy as np
import tensorflow as tf
import time

def lbp_python(Im):
    # Local Binary Patterns 
    sat=len(Im)
    sut=len(Im[0])
    L=np.zeros((sat,sut),dtype='uint8')
    I=np.zeros((sat+2,sut+2),dtype='uint8')
    I[1:sat+1,1:sut+1]=Im
    for i in range(1,sat+1):
        for j in range(1,sut+1):
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


def tf_lbp(x):    
    paddings = tf.constant([[0,0],[1, 1], [1, 1]])
    x=tf.pad(x, paddings,"CONSTANT")        
    b=x.shape 
    M=b[1]
    N=b[2]      
    y=x
    #select the pixels of masks in the form of matrices
    y00=y[:,0:M-2, 0:N-2]
    y01=y[:,0:M-2, 1:N-1]
    y02=y[:,0:M-2, 2:N  ]
    #     
    y10=y[:,1:M-1, 0:N-2]
    y11=y[:,1:M-1, 1:N-1]
    y12=y[:,1:M-1, 2:N  ]
    #
    y20=y[:,2:M, 0:N-2]
    y21=y[:,2:M, 1:N-1]
    y22=y[:,2:M, 2:N ]  

    # Comparisons 
    # 1 -------------------------------        
    g=tf.greater_equal(y01,y11)
    z=tf.multiply(tf.cast(g,dtype='uint8'), 
                  tf.constant(1,dtype='uint8') )      
    # 2 ---------------------------------
    g=tf.greater_equal(y02,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(2,dtype='uint8') )
    z =tf.add(z,tmp)              
    # 3 ---------------------------------
    g=tf.greater_equal(y12,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(4,dtype='uint8') )
    z =tf.add(z,tmp)
    # 4 ---------------------------------
    g=tf.greater_equal(y22,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(8,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 5 ---------------------------------
    g=tf.greater_equal(y21,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(16,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 6 ---------------------------------
    g=tf.greater_equal(y20,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(32,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 7 ---------------------------------
    g=tf.greater_equal(y10,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(64,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 8 ---------------------------------
    g=tf.greater_equal(y00,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(128,dtype='uint8') )
    z =tf.add(z,tmp)  
    #---------------------------------    
    return tf.cast(z,dtype=tf.uint8)



def main():    
    # Test data with random numbers
    Rows=128
    Cols=128
    img_org=np.random.randint(0,255,(Rows,Cols))
    
 
    [Rows,Cols]=img_org.shape
    img_lbp=img_org.reshape(1,Rows,Cols)
    
    # Test Python implementation 
    start_time   = time.time()    
    ypy  =lbp_python(img_lbp[0,:,:].astype('uint8'))
    elapsed_py = time.time() - start_time    
    print('python elapsed_time=',elapsed_py)
    
    # Test TensorFlow
    start_time   = time.time()
    ytf          = tf_lbp(img_lbp.astype('uint8')).numpy()
    elapsed_tf = time.time() - start_time
    print('tensor flow elapsed_time=',elapsed_tf)
    
    #Check if there is an error between TensorFlow and Python implementations
    print('error=',np.sum(ytf[0,:,:]-ypy))
    
    # print('python:')
    # print(ypy)
    # print('TensorFlow:')
    # print(ytf[0,:,:])    

if __name__ == "__main__":
    main()
