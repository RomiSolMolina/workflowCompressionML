# src/student_utils.py

import os
from tensorflow.keras.models import load_model
from src.student_training import train_student_model
from src.auxFunctions import bestHPBO_computation
from src.config.config import DatasetConfig, StudentConfig, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC

# QKeras model definitions
from src.topology.student.student_1d import modelKDQP_1D
from src.topology.student.student_2d import modelStudent2D
from src.topology.student.student_2d_sota import modelStudent2D_SOTA

from src.student_optimization import studentBO

# Mapping configuration based on D_SIGNAL
MODEL_MAP = {
    1: (modelKDQP_1D, (DatasetConfig.SAMPLES,), DatasetConfig.nLabels_1D),
    2: (modelStudent2D, (80, 80, 3), DatasetConfig.nLabels_2D),
    3: (modelStudent2D_SOTA, (32, 32, 3), DatasetConfig.nLabels_2D)
}


def optimize_student(xTrain=None, xTest=None, yTrain=None, yTest=None,
                     images_train=None, y_train=None,
                     images_test=None, y_test=None,
                     teacher_model=None,
                     use_kd=False, use_quant=False, use_prune=False):
    """
    Bayesian Optimization for student model with optional compression strategies.
    """
    x_train = xTrain if xTrain is not None else images_train
    y_train = yTrain if yTrain is not None else y_train
    x_val   = xTest if xTest is not None else images_test
    y_val   = yTest if yTest is not None else y_test

    return studentBO(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        teacher_model=teacher_model,
        use_kd=use_kd,
        use_quant=use_quant,
        use_prune=use_prune
    )


def train_student(bestHP, teacher_model,
                  xTrain=None, xTest=None, yTrain=None, yTest=None,
                  images_train=None, images_test=None, y_train=None, y_test=None,
                  use_kd=False, use_quant=False, use_prune=False):
    """
    Trains the student model using compression flags: quantization, pruning, KD.
    """
    lr = bestHP.get("learning_rate")
    bestHP_student = bestHPBO_computation(bestHP, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC)

    model_fn, input_shape, n_classes = MODEL_MAP[DatasetConfig.D_SIGNAL]

    x_train = xTrain if xTrain is not None else images_train
    y_train = yTrain if yTrain is not None else y_train
    x_val   = xTest if xTest is not None else images_test
    y_val   = yTest if yTest is not None else y_test

    studentModel, history = train_student_model(
        model_fn=model_fn,
        bestHP=bestHP_student,
        teacher_model=teacher_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        lr=lr,
        input_shape=input_shape,
        n_classes=n_classes,
        use_kd=use_kd,
        use_quant=use_quant,
        use_prune=use_prune
    )

    studentModel.save(StudentConfig.MODEL_PATH)
    return studentModel, history


def load_student_model(path=StudentConfig.MODEL_PATH):
    """
    Load pre-trained student model.
    """
    model = load_model(path)
    model.summary()
    return model
