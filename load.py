# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 23:55:09 2021

@author: Robin
"""

import numpy as np
import h5py
from scipy import signal
import matplotlib.pyplot as plt
import cv2 as cv

def load_model_params(model_path):
    
    model = h5py.File(model_path, "r")
    
    
    conv1_kernel = np.array(model["conv2d_1"]["conv2d_1"]["kernel:0"])
    conv1_bias = np.array(model["conv2d_1"]["conv2d_1"]["bias:0"])
    
    conv2_kernel = np.array(model["conv2d_2"]["conv2d_2"]["kernel:0"])
    conv2_bias = np.array(model["conv2d_2"]["conv2d_2"]["bias:0"])
    
    dense1_weights = np.array(model["dense_1"]["dense_1"]["kernel:0"])
    dense1_bias = np.array(model["dense_1"]["dense_1"]["bias:0"])
    
    dense2_weights = np.array(model["dense_2"]["dense_2"]["kernel:0"])
    dense2_bias = np.array(model["dense_2"]["dense_2"]["bias:0"])
    model.close()
    return conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, dense1_weights, dense1_bias, dense2_weights, dense2_bias
    
def relu(x):
    return x.clip(0)

def softmax(x):
    """ Returns softmax of logits"""
    exp = np.exp(x)
    return exp/np.sum(exp)

def max_pooling_2d(x, pool_size):
    """ Standard 2D max pooling with valid packing and a stride equal to pool_size"""
    strides = pool_size
    im_size = x.shape[0]   
    new_size = int((im_size-pool_size)/strides+1)
    out = np.empty((new_size, new_size, x.shape[2]))
    for i in range(new_size):
        for j in range(new_size):
            
            out[i,j,:] = np.max(x[i*strides:i*strides+pool_size, j*strides:j*strides+pool_size], axis = (0,1))
            
    return out

def conv_2d(x, kernels, bias):
    """ takes a (im_size ,im_size) or (im_size ,im_size, n_channels) shape array and returns a
    (im_size-kernel_size + 1, im_size-kernel_size + 1, n_filters) shaped array. Only valid packingb is supported """
    
    # Get input and output shape parameters
    im_shape = x.shape
    im_size = im_shape[0]
    kernel_shape = kernels.shape
    kernel_size = kernel_shape[0]
    n_filters = kernel_shape[3]
    n_channels = kernel_shape[2]
    output_size = im_size - kernel_size + 1
    output_shape = (output_size, output_size, n_filters)
    
    
    
    convolved = np.zeros(output_shape)
    for i in range(n_filters):
        for j in range(n_channels):
            convolved[:,:,i] += signal.correlate2d(x[:, :, j], kernels[:,:,j,i], mode = "valid")
    
    return convolved + bias

def dense(x, weights, bias):
    return np.dot(x, weights) + bias

def read_image(image, model_path = "model.h5"):
    conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, dense1_weights, dense1_bias, dense2_weights, dense2_bias = load_model_params(model_path)
    conv1 = relu(conv_2d(image, conv1_kernel, conv1_bias))
    conv2 = relu(conv_2d(conv1, conv2_kernel, conv2_bias))
    pooled = max_pooling_2d(conv2, 2)
    flatten = pooled.flatten()
    dense1 = relu(dense(flatten, dense1_weights, dense1_bias))
    dense2 = dense(dense1, dense2_weights, dense2_bias)
    output = softmax(dense2)
    return output

