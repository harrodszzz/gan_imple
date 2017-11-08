#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:22:32 2017

@author: yucheng
"""

from keras.models import Model

# generator
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

# discriminator takes input of an image 
# and give the output to tell whether it is a fake image

def discriminator(size=28):
    input_img = Input(shape=(size,size,1))
    conv1 = Conv2D(32,(3,3),padding='same',activation='relu')(input_img)
    conv1 = Conv2D(32,(3,3),padding='same',activation='relu')(conv1)
    max1 = MaxPooling2D()(conv1)
    conv2 = Conv2D(64,(3,3),padding='same',activation='relu')(max1)
    conv2 = Conv2D(64,(3,3),padding='same',activation='relu')(conv2)
    max2 = MaxPooling2D()(conv2)
    confidence = Conv2D(1,(3,3),padding='same',activation='relu')(max2)
    output = GlobalAveragePooling2D()(confidence)
    d = Model(inputs=input_img,outputs=output)
    return d

# generator takes input of some noise 
# and give the output of an image close to the real dataset
