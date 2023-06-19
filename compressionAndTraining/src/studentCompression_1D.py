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
  

    
    x_out = QActivation('softmax',name='output_softmax')(x)
    
    qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
    
    qmodel.summary()

    return qmodel