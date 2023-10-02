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



def modelKDQP_1D(bestHP):
    
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
  

    
    studentQ_MLP = keras.Sequential(
            [   
                Input(shape=(30,)),
                QDense(bestHP[0], name='fc1',
                        kernel_quantizer= kernelQ, bias_quantizer= biasQ,
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)),
                QActivation(activation= activationQ ,  name='relu1'),
                    
        #        QDense(bestHP.get("fc2"), name='fc2',
        #                kernel_quantizer=quantized_bits(9,1,alpha=1), bias_quantizer=quantized_bits(23,15,alpha=1),
        #                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)),
        #        QActivation(activation=quantized_relu(16,15), name='relu2'),
                
        #        QDense(bestHP.get("fc3"), name='fc3',
        #                 kernel_quantizer=quantized_bits(9,1,alpha=1), bias_quantizer=quantized_bits(23,15,alpha=1),
        #                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)),
        #        QActivation(activation=quantized_relu(16,15), name='relu3'), 


                QDense(4, name='output',
                        kernel_quantizer= kernelQ, bias_quantizer= biasQ,
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)),
                Activation(activation='softmax', name='softmax')

                
            ],
            name="student",
        )


    return studentQ_MLP