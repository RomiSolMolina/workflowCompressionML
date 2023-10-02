
import os

import numpy as np
from numpy import array
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

import keras_tuner as kt


from src.config import *
from src.topology.modelTeacher_HPO_2D import *


def teacherBO_2D (images_train, y_train, images_test, y_test):

    """ 
    This function performs huperparameters optimization for the 2D teacher model. 
    The topology to used for the teacher is defined in src.topology.modelTeacher_HPO_2D
    
    """    

    bestHP = []

    callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, verbose=1),
            ]  
    
    if (os.path.exists(OUTPUT_PATH_TEACHER) == 'True'):
        shutil.rmtree(OUTPUT_PATH_TEACHER, ignore_errors = True)
        
    tuner = kt.BayesianOptimization(
        topologyTeacher_HPO_2D,
        objective = "val_accuracy",
        max_trials = N_ITERATIONS_TEACHER,
        seed = 37,
        directory = OUTPUT_PATH_TEACHER
)

    tuner.search(

        x=images_train, y=y_train,
        validation_data=(images_test, y_test),
        batch_size = BATCH_TEACHER,
        callbacks=[callbacks],
        epochs= EPOCHS_TEACHER
    )


    tuner.get_best_hyperparameters(num_trials=1)[0] 
       
    bestHP = tuner.get_best_hyperparameters()[0]
    
    return bestHP