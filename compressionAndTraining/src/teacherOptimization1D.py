import os

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

import config

# Defiition of the teacher model. 
# Architecture: Multi-layer perceptron

def build_model(hp):

    model = Sequential()
    inputShape = config.INPUT_SHAPE
    chanDim = -1
    
    # Model definition 
    model.add(Dense(
        hp.Int("fc1", min_value=32, max_value=300, step=10),
        kernel_regularizer=l2(0.0001), input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc2", min_value=32, max_value=100, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc3", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc4", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    
    # Output Layer with Softmax activation
    model.add(Dense(4, name='output'))
    model.add(Activation("softmax"))
    
    # Initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate",
                   values=[1e-1, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)
    
    # Compile the model
    model.compile(optimizer=opt, loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model


def teacherBO(xTrain, yTrain, xTest, yTest, it):


    OUTPUT_PATH = "tuner_teacher"

    if (os.path.exists(OUTPUT_PATH) == 'True'):
        shutil.rmtree(OUTPUT_PATH, ignore_errors = True)

        
    es = EarlyStopping(
        monitor="val_loss",
        patience= config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True)


    tuner = kt.BayesianOptimization(
        build_model,
        objective = "val_accuracy",
        max_trials = 5,
        seed = 37,
        directory = OUTPUT_PATH
    )

    tuner.search(
        x=xTrain, y=yTrain,
        validation_data=(xTest, yTest),
        batch_size = config.BS,
        callbacks=[es],
        epochs = config.EPOCHS
    )

    tuner.get_best_hyperparameters(num_trials=1)[0] 

    summary = tuner.results_summary(num_trials=it)
      
    bestHP = tuner.get_best_hyperparameters()[0]
    
    model = tuner.hypermodel.build(bestHP)

    return bestHP
