import os
import numpy as np
import shutil, sys


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
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.sparsity import keras as sparsity

import keras_tuner as kt
from qkeras import *

from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm


from src.distillationClassKeras import *
import src.config

def topology_student_SOTA(hp):

    kernelQ_4b = "quantized_bits(4,2,alpha=1)"
    kernelQ = "quantized_bits(8,2,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    biasQ_4b = "quantized_bits(4,2,alpha=1)"
    activationQ = 'quantized_relu(8,2)'
    activationQ_4b = 'quantized_relu(4, 0)'

    CONSTANT_SPARSITY = 0.5
    
    # INPUT_SHAPE = (32, 32, 3)
    model = Sequential()
    # inputShape = (32, 32, 3)
    chanDim = -1

# First block

    model.add(QConv2DBatchnorm(hp.Int("conv_1", min_value=1, max_value=10, step=1), kernel_size=(3,3), 
                               padding='same',
                               kernel_quantizer = kernelQ, 
                               bias_quantizer = biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True,
                               input_shape=(32, 32, 3)
                               ))
    model.add(QActivation(activationQ ,name='relu1'))

    model.add(QConv2DBatchnorm(hp.Int("conv_2", min_value=1, max_value=10, step=1), 
                               kernel_size=(3,3), 
                               padding='same',
                               kernel_quantizer = kernelQ, 
                               bias_quantizer = biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True
                               ))
    model.add(QActivation(activationQ ,name='relu2'))    
    
    model.add(MaxPooling2D(pool_size=(2, 2)))        

# Second block
    model.add(QConv2DBatchnorm(hp.Int("conv_3", min_value=1, max_value=10, step=1), 
                               kernel_size=(3,3), 
                               padding='same',
                               kernel_quantizer = kernelQ, 
                               bias_quantizer = biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True
                               ))
    model.add(QActivation(activationQ ,name='relu3'))

    model.add(QConv2DBatchnorm(hp.Int("conv_4", min_value=1, max_value=10, step=1), 
                               kernel_size=(3,3), 
                               padding='same',
                               kernel_quantizer = kernelQ, 
                               bias_quantizer = biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True
                               ))
    model.add(QActivation(activationQ ,name='relu4'))    
    
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    
# Third block
    model.add(QConv2DBatchnorm(hp.Int("conv_5", min_value=1, max_value=10, step=1), 
                               kernel_size=(3,3), 
                               padding='same',
                               kernel_quantizer = kernelQ, 
                               bias_quantizer = biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True
                               ))
    model.add(QActivation(activationQ ,name='relu5'))

    model.add(QConv2DBatchnorm(hp.Int("conv_6", min_value=1, max_value=10, step=1), 
                               kernel_size=(3,3), 
                               padding='same',
                               kernel_quantizer = kernelQ, 
                               bias_quantizer = biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True
                               ))
    model.add(QActivation(activationQ ,name='relu6'))    

    
# Fourth block    
    model.add(QConv2DBatchnorm(hp.Int("conv_7", min_value=1, max_value=10, step=1), 
                               kernel_size=(3,3), 
                               padding='same',
                               kernel_quantizer = kernelQ, 
                               bias_quantizer = biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True
                               ))
    model.add(QActivation(activationQ ,name='relu7'))

    model.add(QConv2DBatchnorm(hp.Int("conv_8", min_value=1, max_value=10, step=1), 
                               kernel_size=(3,3), 
                               padding='same',
                               kernel_quantizer = kernelQ, 
                               bias_quantizer = biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001), use_bias=True
                               ))
    model.add(QActivation(activationQ ,name='relu8'))    
    

    model.add(MaxPooling2D(pool_size=(2, 2)))    
    
    model.add(Flatten())
    
    model.add(QDense(hp.Int("fc1", min_value=5, max_value=10, step=10),
                 kernel_quantizer=quantized_bits(8,1,alpha=1), bias_quantizer=quantized_bits(8,1,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001))),
    model.add(QActivation(activation=quantized_relu(8),  name='relu1_D'))
    model.add(QDense(hp.Int("fc2", min_value=5, max_value=10, step=10),
                 kernel_quantizer=quantized_bits(8,1,alpha=1), bias_quantizer=quantized_bits(8,1,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001))),
    model.add(QActivation(activation=quantized_relu(8),  name='relu2_D'))
    model.add(QDense(hp.Int("fc3", min_value=5, max_value=10, step=10),
                 kernel_quantizer=quantized_bits(8,1,alpha=1), bias_quantizer=quantized_bits(8,1,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001))),
    model.add(QActivation(activation=quantized_relu(8),  name='relu3_D'))
    
    
    # Output Layer with Softmax activation
    model.add(QDense(10, name='output',
                kernel_quantizer=quantized_bits(8,1,alpha=1), bias_quantizer=quantized_bits(8,1,alpha=1),
                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001))),
    model.add(Activation(activation='softmax', name='softmax'))
 
    # initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate",
                   values=[1e-1, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)
    
    NSTEPS = int(31188*0.9) // 128
    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(CONSTANT_SPARSITY, begin_step = NSTEPS*2,  end_step = NSTEPS*10, frequency = NSTEPS)} #2000
    model = prune.prune_low_magnitude(model, **pruning_params)
    
    # compile the model
    model.compile(optimizer=opt, loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model


ù
