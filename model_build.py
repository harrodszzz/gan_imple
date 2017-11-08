#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:22:32 2017

@author: yucheng
"""

from keras.models import Model, Sequential

# generator
from keras.layers import Dense, Activation, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D

# discriminator takes input of an image 
# and give the output to tell whether it is a fake image

# argument size refers to the size of images from dataset
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
def generator(size=28):
    input_signal = Input(shape=(100,))
    dens1 = Dense(1024,activation='relu')(input_signal)
    start_sz = int(size/4)
    dens2 = Dense(start_sz*start_sz*128,activation='relu')(dens1)
    resh1 = Reshape([start_sz,start_sz,128])(dens2)
    upsa1 = UpSampling2D()(resh1) # now the size becomes (size/2,size/2)
    conv1 = Conv2D(64,(3,3),padding='same',activation='relu')(upsa1)
    conv1 = Conv2D(64,(3,3),padding='same',activation='relu')(conv1)
    upsa2 = UpSampling2D()(conv1) # now the size becomes (size,size)
    conv2 = Conv2D(128,(3,3),padding='same',activation='relu')(upsa2)
    conv2 = Conv2D(128,(3,3),padding='same',activation='relu')(conv2)
    # goes to one image
    conv3 = Conv2D(1,(3,3),padding='same',activation='relu')(conv2)
    g = Model(inputs=input_signal,outputs=conv3)
    return g

def generator_containing_discriminator(g,d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model