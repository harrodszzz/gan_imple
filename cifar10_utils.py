#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:13:24 2017

@author: yucheng
"""

from keras.datasets import cifar10
import numpy as np

# load data
(x_tr,y_tr),(x_te,y_te) = cifar10.load_data()

# concatenate
def load_data():    
    x = np.concatenate((x_tr,x_te),axis=0)
    y = np.ones(y_tr.shape[0]+y_te.shape[0])
    return x,y
