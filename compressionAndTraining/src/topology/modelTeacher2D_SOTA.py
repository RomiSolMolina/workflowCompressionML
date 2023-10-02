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
from tensorflow.keras import layers


def topologyTeacher_2D_SOTA(bestHP):

    """ 
    Topology definition for the 2D teacher model, using the results of the hyperparameters optimization obtained
    """


    teacherCNN_SOTA = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            
            layers.Conv2D(bestHP[0], (3, 3), padding="same", name='conv_1', kernel_regularizer=l2(0.0001)),
            layers.BatchNormalization(),
            layers.Activation(activation='relu', name='relu1'),        
            layers.Conv2D(bestHP[1], (3, 3), padding="same", name='conv_2', kernel_regularizer=l2(0.0001)),
            layers.BatchNormalization(),        
            layers.Activation(activation='relu', name='relu2'),
            layers.MaxPooling2D(pool_size=(2, 2)), 
            
            layers.Conv2D(bestHP[2], (3, 3), padding="same", name='conv_3', kernel_regularizer=l2(0.0001)),
            layers.BatchNormalization(),        
            layers.Activation(activation='relu', name='relu3'),
            layers.Conv2D(bestHP[3],  (3, 3), padding="same", name='conv_4', kernel_regularizer=l2(0.0001)),
            layers.BatchNormalization(),          
            layers.Activation(activation='relu', name='relu4'),
            layers.MaxPooling2D(pool_size=(2, 2)),       
            
            layers.Conv2D(bestHP[4], (3, 3), padding="same", name='conv_5', kernel_regularizer=l2(0.0001)),
            layers.Conv2D(bestHP[5], (3, 3), padding="same", name='conv_6', kernel_regularizer=l2(0.0001)),

            layers.Activation(activation='relu', name='relu5'),
            layers.MaxPooling2D(pool_size=(2, 2)), 
            
            layers.Flatten(),
            
            layers.Dense(bestHP[6], name='fc1', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='relu', name='relu6'),
            
            layers.Dense(bestHP[7], name='fc2', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='relu', name='relu7'),
            layers.Dense(bestHP[8], name='fc3', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='relu', name='relu8'),

            
            layers.Dense(10, name='output', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='softmax', name='softmax'),
            
        ],
        name="teacherCNN_SOTA",
    )
    teacherCNN_SOTA.summary()


    return teacherCNN_SOTA