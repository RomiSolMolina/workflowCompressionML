
# Import libraries

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
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE, MDS
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

import seaborn as sn
import pandas as pd

import keras_tuner as kt
from qkeras import *

from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm

#pip install scikit-image
import skimage.data
import skimage.transform
from skimage import io

import shutil, sys

from itertools import cycle

from keras.applications.vgg16 import preprocess_input

# Custom functions
from src.distillationClassKeras import *

# Plot confusion matrix
from src.confMatrix import *


from src.teacherOptimization1D import *
from src.teacherOptimization2D import *
from src.teacherOptimization2D_SOTA import *
from src.studentOptimization2D_SOTA import *

from src.loadDataset import *

from src.config import *

# Pre-trained model 
# MobileNet
from keras.applications import MobileNet

from src.auxFunctions import *

from teacherTraining_1D import *
from teacherTraining_2D import *
from studentOptimization_2D import *
from studentOptimization_1D import *

from studentCompression1D import *
from src.studentCompression2D import *

def startDNNTrainingAndCompression():

    """
    The function performs training and compression of the teacher and student architectures
    """

    ## ----------------  Load DATASET -----------------

    if D_SIGNAL == 1:
    # Load 1D signal dataset
        xTrain, xTest, xTest_df_Final, yTrain, yTest, yTest_Final = loadDataset_1D(ROOT_PATH_1D, nLabels_1D, SAMPLES)
    elif D_SIGNAL == 2:
    # Load 2D signal dataset
        images_train, images_validation, images_test, y_train, y_test = loadDataset_2D(ROOT_PATH_2D, nLabels_2D, ROWS, COLS)
    elif D_SIGNAL == 3:
    # CIFAR-10 dataset provided by keras
        from keras.datasets import cifar10
        (images_train, y_train), (images_test, y_test) = keras.datasets.cifar10.load_data()
        images_train, images_test = normalizationPix(images_train, images_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    
    ## ----------------  TEACHER optimization -----------------

    # Decide if optimize a teacher architecture or load a pre-trained network as teacher
    # Grab the best hyperparameters

    if TEACHER_OP == 0:
        # optimize teacher architecture
        print("Teacher optimization")
      
        if D_SIGNAL == 1:
            print("1D signal")
            bestHP_BO_teacher = teacherBO_1D(xTrain, xTest, yTrain, yTest)
        elif D_SIGNAL == 2:
            print("2D signal")
            bestHP_BO_teacher = teacherBO_2D(images_train, y_train, images_test, y_test)
        else: 
            bestHP_BO_teacher = teacherBO_SOTA(images_train, y_train, images_test, y_test, N_ITERATIONS_TEACHER)
    else: 

        # Load pre-trained model
        #teacherModel = load_model('models/CNN/teacher_NEW_v2_ok.h5')    #VGG-16 based
        teacherModel = load_model('models/teacherModel_MobileNetV2.h5')
        
        teacherModel.summary()

    ## ----------------- TEACHER training -----------------

    # Grab the best hyperparameters for teacher training
    if TEACHER_OP == 0:
        if D_SIGNAL == 1:
            lr = bestHP_BO_teacher.get('learning_rate')

            # Grab best hyperparams
            bestHP_BO_teacher = bestHPBO_computation(bestHP_BO_teacher, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC)
            print(bestHP_BO_teacher)
            
            # Train 1D teacher model
            teacherModel = teacherTrainingAfterBPO_1D(bestHP_BO_teacher, xTrain, xTest, yTrain, yTest, lr)

            # Save model 1D teacher model
            teacherModel.save(PATH_MODEL_TEACHER)

        elif D_SIGNAL == 2:

            lr = bestHP_BO_teacher.get('learning_rate')
          
            # Grab best hyperparams    
            bestHP_BO_teacher = bestHPBO_computation(bestHP_BO_teacher, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC)

            # Train 2D teacher model
            teacherModel = teacherTrainingAfterBPO_2D(bestHP_BO_teacher, images_train, y_train, teacherModel, lr)

            # Save model 2D teacher model
            teacherModel.save(PATH_MODEL_TEACHER)
        
        elif D_SIGNAL == 3:
            
            from src.teacherTraining2D_SOTA import teacherTrainingAfterBPO_SOTA

            lr = bestHP_BO_teacher.get('learning_rate')
   
            # Grab best hyperparams - SOTA dataset   
            bestHP_BO_teacher = bestHPBO_computation(bestHP_BO_teacher, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC)
        
            # Train 2D teacher model - SOTA dataset
            teacherModel = teacherTrainingAfterBPO_SOTA(bestHP_BO_teacher, images_train, images_test, y_train, y_test, lr)
        
            # Save model 2D teacher model - SOTA dataset
            teacherModel.save(PATH_MODEL_TEACHER)
    
    teacherModel.summary()

    # Bayesian optimization for student architecture
    if D_SIGNAL == 1:
        bestHP_BO = studentBO_1D(xTrain, xTest, yTrain, yTest, teacherModel, N_ITERATIONS_STUDENT)
    elif D_SIGNAL == 2:
        bestHP_BO = studentBO_2D(images_train, y_train, images_test, y_test, teacherModel, N_ITERATIONS_STUDENT)
    elif D_SIGNAL == 3:
        bestHP_BO = studentBO_2D_SOTA(images_train, y_train, images_test, y_test, teacherModel, N_ITERATIONS_STUDENT)

    if D_SIGNAL == 1:
        lr = bestHP_BO.get('learning_rate')
        # Grab best hyperparams
        bestHP_BO_student = bestHPBO_computation(bestHP_BO, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC)
        print(bestHP_BO_student)
            
        # Train 1D student model
        studentModel = studentCompression_1D(bestHP_BO_student, xTrain, xTest, yTrain, yTest, teacherModel, lr)
      
        # Save model 1D student model
        studentModel.save(PATH_MODEL_STUDENT)

    
    elif D_SIGNAL == 2:

        lr = bestHP_BO.get('learning_rate')

        # Grab best hyperparams    
        bestHP_BO_student = bestHPBO_computation(bestHP_BO, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC)

        # Training to obtain compressed model
        studentModel = studentCompression_2D(bestHP_BO_student, images_train, y_train, teacherModel, lr)

        # Save model 2D student model
        studentModel.save(PATH_MODEL_STUDENT)

    elif D_SIGNAL == 3:

        lr = bestHP_BO.get('learning_rate')

        # Grab best hyperparams    
        bestHP_BO_student = bestHPBO_computation(bestHP_BO, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC)

        # Training to obtain compressed model
        studentModel = studentCompression_2D_SOTA(bestHP_BO_student, images_train, images_test, y_train, y_test, teacherModel, lr)

        studentModel.save(PATH_MODEL_STUDENT)
        # Save model 2D student model
        #studentModel.save("models/studentModel_2D_SOTA.h5")

    # Model summary
    studentModel.summary()

    # Plot confusion matrix for accuracy evaluation
    if D_SIGNAL == 1:
        # 1D signal
        confusionMatrixPlot(studentModel, xTest_df_Final, yTest_Final)

    else:
        # 2D signal
        confusionMatrixPlot(studentModel, images_train, y_train)