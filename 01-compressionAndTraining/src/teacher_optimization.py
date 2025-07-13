
import os
import shutil
import tensorflow as tf
import keras_tuner as kt
from src.config.config import TeacherConfig, DatasetConfig
from src.topology.teacher.teacher_1d import topologyTeacher_HPO_1D
from src.topology.teacher.teacher_2d import topologyTeacher_HPO_2D
from src.topology.teacher.teacher_2d_sota import topologyTeacher_HPO_2D_SOTA


def teacherBO(images_train=None, y_train=None, images_test=None, y_test=None,
              x_train=None, x_test=None, y_train_1d=None, y_test_1d=None):
    """
    Generic Bayesian Optimization dispatcher for teacher models based on DatasetConfig.D_SIGNAL
    """
    if os.path.exists(TeacherConfig.OUTPUT_PATH):
        shutil.rmtree(TeacherConfig.OUTPUT_PATH, ignore_errors=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", factor=0.5, patience=3, verbose=1),
    ]

    if DatasetConfig.D_SIGNAL == 1:
        # 1D optimization
        tuner = kt.BayesianOptimization(
            hypermodel=topologyTeacher_HPO_1D,
            objective="val_accuracy",
            max_trials=TeacherConfig.N_ITERATIONS,
            seed=42,
            directory=TeacherConfig.OUTPUT_PATH
        )

        tuner.search(
            x=x_train,
            y=y_train_1d,
            validation_data=(x_test, y_test_1d),
            batch_size=TeacherConfig.BATCH_SIZE,
            epochs=TeacherConfig.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

    elif DatasetConfig.D_SIGNAL == 2:
        # 2D optimization
        tuner = kt.BayesianOptimization(
            hypermodel=topologyTeacher_HPO_2D,
            objective="val_accuracy",
            max_trials=TeacherConfig.N_ITERATIONS,
            seed=42,
            directory=TeacherConfig.OUTPUT_PATH
        )

        tuner.search(
            x=images_train,
            y=y_train,
            validation_data=(images_test, y_test),
            batch_size=TeacherConfig.BATCH_SIZE,
            epochs=TeacherConfig.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

    else:
        # SOTA optimization
        tuner = kt.BayesianOptimization(
            hypermodel=topologyTeacher_HPO_2D_SOTA,
            objective="val_accuracy",
            max_trials=TeacherConfig.N_ITERATIONS,
            seed=42,
            directory=TeacherConfig.OUTPUT_PATH
        )

        tuner.search(
            x=images_train,
            y=y_train,
            validation_data=(images_test, y_test),
            batch_size=TeacherConfig.BATCH_SIZE,
            epochs=TeacherConfig.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hp