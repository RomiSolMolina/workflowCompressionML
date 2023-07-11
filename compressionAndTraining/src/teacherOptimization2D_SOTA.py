
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




import src.config

def build_model(hp):
    INPUT_SHAPE = (32, 32, 3)
    
    model = Sequential()


# Model definition 
# First block
    model.add(Conv2D(
        hp.Int("conv_1", min_value=32, max_value=64, step=32),
        (3, 3), padding="same",
        kernel_regularizer=l2(0.0001), input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
        
    model.add(Conv2D(
        hp.Int("conv_2", min_value=32, max_value=64, step=32),
        (3, 3), padding="same",
        kernel_regularizer=l2(0.0001), input_shape=(80, 80, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))        

# Second block
    model.add(Conv2D(
        hp.Int("conv_3", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(
        hp.Int("conv_4", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))          
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))                 
 
 # Third block
    model.add(Conv2D(
        hp.Int("conv_5", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(
        hp.Int("conv_6", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))          
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))   

 # Fourth block
    model.add(Conv2D(
        hp.Int("conv_7", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(
        hp.Int("conv_8", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))          
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))   


    model.add(Flatten())
    
    model.add(Dense(
        hp.Int("fc1", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc2", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc3", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    
    model.add(Activation("relu"))
    
    # Output Layer with Softmax activation
    model.add(Dense(10, activation='softmax')) 
     
    # Initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate",
                   values=[1e-1, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)
    
    # Compile the model
    model.compile(optimizer=opt, loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model


def teacherBO_SOTA (images_train, y_train, images_test, y_test, N_ITERATIONS_TEACHER):
    
    bestHP = []

    OUTPUT_PATH = "tuner_teacher"

    callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, verbose=1),
            ]  
    
    if (os.path.exists(OUTPUT_PATH) == 'True'):
        shutil.rmtree(OUTPUT_PATH, ignore_errors = True)
        
    tuner = kt.BayesianOptimization(
        build_model,
        objective = "val_accuracy",
        max_trials = N_ITERATIONS_TEACHER,
        seed = 37,
        directory = OUTPUT_PATH
)

    tuner.search(

        x=images_train, y=y_train,
        validation_data=(images_test, y_test),
        batch_size = 64,
        callbacks=[callbacks],
        epochs= 32
    )


    tuner.get_best_hyperparameters(num_trials=1)[0] 
        
    bestHP = tuner.get_best_hyperparameters()[0]

    return bestHP