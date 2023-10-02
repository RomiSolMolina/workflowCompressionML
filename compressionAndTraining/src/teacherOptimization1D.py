
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

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

import shutil, sys

from src.config import *
from src.topology.modelTeacher_HPO_1D import *

# Keras tuner
# https://www.tensorflow.org/tutorials/keras/keras_tuner
# 

# Defintion of the teacher model. 
# Architecture: Multi-layer perceptron



def teacherBO_1D(xTrain, xTest, yTrain, yTest):


    # Clean OUTPUT_PATH
    if (os.path.exists(OUTPUT_PATH_TEACHER) == 'True'):
        shutil.rmtree(OUTPUT_PATH_TEACHER, ignore_errors = True)
       
    es = EarlyStopping(
        monitor="val_loss",
        patience= EARLY_STOPPING_PATIENCE_TEACHER,
        restore_best_weights=True)

    tuner = kt.BayesianOptimization(
        topologyTeacher1D,
        objective = "val_accuracy",
        max_trials = N_ITERATIONS_TEACHER, 
        seed = 37,
        directory = OUTPUT_PATH_TEACHER
    )

    tuner.search(
        x=xTrain, y=yTrain,
        validation_data=(xTest, yTest),
        batch_size = BATCH_TEACHER,
        callbacks=[es],
        epochs = EPOCHS_TEACHER
    )

    #tuner.get_best_hyperparameters(num_trials=1)[0] 
     
    bestHP = tuner.get_best_hyperparameters()[0]
    

    return bestHP
