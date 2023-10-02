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


def modelTeacherTopology_1D(bestHP):

    teacher = keras.Sequential(
        [
            keras.Input(shape=(30,)),
            
            layers.Dense(bestHP[0], name='fc1', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='relu', name='relu1'),
 
            # layers.Dense(200, name='fc3', kernel_regularizer=l2(0.0001)),
            # layers.Activation(activation='relu', name='relu3'),

            layers.Dense(bestHP[1], name='fc2', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='relu', name='relu2'),
            layers.Dense(bestHP[2], name='fc3', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='relu', name='relu3'),
            layers.Dense(bestHP[3], name='fc4', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='relu', name='relu4'),
            
            layers.Dense(4, name='output', kernel_regularizer=l2(0.0001)),
            layers.Activation(activation='softmax', name='softmax'),
            
        ],
        name="teacher",
    )

    teacher.summary()


    return teacher