import os
import csv
import numpy as np
from numpy import array
import time
import glob

import tensorflow as tf 

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import SGD, Adam

import keras_tuner as kt
from qkeras import *

from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input



def modelKDQP_2D(bestHP):
    
    '''

    Model to be compressed. Defined with quantization strategies. 
    Input: best hyper-params from BO process.
    Output: compressed model. 

    '''
    ######## ---------------------------  Model definition - 2D STUDENT -----------------------------------------

    # Number of bits 
    ## 4-bits
    kernelQ_4b = "quantized_bits(4,2,alpha=1)"
    biasQ_4b = "quantized_bits(4,2,alpha=1)"
    activationQ_4b = 'quantized_bits(4, 0)'
    ## 8-bits
    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    activationQ = 'quantized_bits(8)'
  
    # Input
    x = x_in = Input(shape=(80,80,3))

    # Block 1
    x = QConv2DBatchnorm(int(bestHP[0]), kernel_size=(3,3), 
                            padding='same',
                            kernel_quantizer = kernelQ, 
                            bias_quantizer = biasQ,
                            kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
                            name='conv1')(x)
    x = QActivation(activationQ ,name='relu1')(x)
    x = QConv2DBatchnorm(int(bestHP[1]), kernel_size=(3,3), 
                            padding='same',
                            kernel_quantizer = kernelQ,
                            bias_quantizer = biasQ,
                            kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
                            name='conv2')(x) 
    x = QActivation(activationQ, name='relu2')(x)
    x = MaxPooling2D(pool_size = (2,2),name='pool_0')(x)

    # Block 2
    x = QConv2DBatchnorm(int(bestHP[2]), kernel_size=(3,3), 
                            padding='same',
                            kernel_quantizer = kernelQ,
                            bias_quantizer = biasQ,
                            kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
                            name='conv3')(x)
    x = QActivation(activationQ, name='relu3')(x)
    x = QConv2DBatchnorm(int(bestHP[3]), kernel_size=(3,3), 
                            padding='same',
                            kernel_quantizer = kernelQ,
                            bias_quantizer = biasQ,
                            kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
                            name='conv4')(x) 
    x = QActivation(activationQ, name='relu4')(x)
    x = MaxPooling2D(pool_size = (2,2),name='pool_1')(x)

    # Block 3
    # Commented for MobileNetV2
    # x = QConv2DBatchnorm(int(bestHP[4]), kernel_size=(3,3), 
    #                         padding='same',
    #                         kernel_quantizer = kernelQ,
    #                         bias_quantizer = biasQ,
    #                         kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
    #                         name='conv5')(x)
    # x = QActivation(activationQ, name='relu5')(x)
    # x = QConv2DBatchnorm(int(bestHP[5]), kernel_size=(3,3), 
    #                         padding='same',
    #                         kernel_quantizer = kernelQ,
    #                         bias_quantizer = biasQ,
    #                         kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
    #                         name='conv6')(x) 
    # x = QActivation(activationQ, name='relu6')(x)
    # x = MaxPooling2D(pool_size = (2,2),name='pool_2')(x)

    # # Block 4
    # x = QConv2DBatchnorm(int(bestHP[6]), kernel_size=(3,3), 
    #                         padding='same',
    #                         kernel_quantizer = kernelQ,
    #                         bias_quantizer = biasQ,
    #                         kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
    #                         name='conv7')(x)
    # x = QActivation(activationQ,name='relu7')(x)
    # x = QConv2DBatchnorm(int(bestHP[7]), kernel_size=(3,3), 
    #                         padding='same',
    #                         kernel_quantizer = kernelQ,
    #                         bias_quantizer = biasQ,
    #                         kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
    #                         name='conv8')(x) 
    # x = QActivation(activationQ, name='relu8')(x)

    # x = MaxPooling2D(pool_size = (2,2),name='pool_3')(x)
                                        
    x = Flatten()(x)
  
    # x = QDense(bestHP[8],
    #              kernel_quantizer=kernelQ, bias_quantizer=activationQ,
    #              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001))(x)
    # x = QActivation(activation=activationQ,  name='relu1_D')(x)

    # x = QDense(bestHP[9],
    #              kernel_quantizer=kernelQ, bias_quantizer=activationQ,
    #              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001))(x)
    # x = QActivation(activation=activationQ,  name='relu2_D')(x)
              
    # x = QDense(bestHP[10],
    #              kernel_quantizer=kernelQ, bias_quantizer=activationQ,
    #              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001))(x)
    # x = QActivation(activation=activationQ,  name='relu3_D')(x)
    
    
    # Output Layer with Softmax activation
    x = QDense(2, name='output',
                kernel_quantizer=kernelQ, bias_quantizer=activationQ,
                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001))(x)
    
    #x_out = QActivation('softmax', name='output_softmax')(x)
    x_out = Activation(activation='softmax', name='softmax')(x)
    
    qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
    
    qmodel.summary()

    return qmodel