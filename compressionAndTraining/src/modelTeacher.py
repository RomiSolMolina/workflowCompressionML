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


def modelTeacherDefinition(bestHP):




    return model