import os
import shutil
import tensorflow as tf
import keras_tuner as kt

from src.config.config import TeacherConfig, DatasetConfig
from src.topology.teacher_hpo.teacher_hpo_1d import topologyTeacher_HPO_1D
from src.topology.teacher_hpo.teacher_hpo_2d import topologyTeacher_HPO_2D
from src.topology.teacher_hpo.teacher_hpo_2d_sota import topology_teacher_hpo_2D_SOTA


def _run_bayesian_opt(hpo_function, x_train, y_train, x_val, y_val, output_path):
    """
    Runs Bayesian Optimization with the provided topology function and data.
    """
    # Clean output directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)

    tuner = kt.BayesianOptimization(
        hypermodel=hpo_function,
        objective="val_accuracy",
        max_trials=TeacherConfig.N_ITERATIONS,
        seed=37,
        directory=output_path
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", factor=0.5, patience=3, verbose=1),
    ]

    tuner.search(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=TeacherConfig.BATCH_SIZE,
        epochs=TeacherConfig.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hp


def run_teacher_hpo(x_train, y_train, x_val, y_val):
    """
    Selects the appropriate topology based on the signal type and runs optimization.
    """
    if DatasetConfig.D_SIGNAL == 1:
        return _run_bayesian_opt(
            topologyTeacher_HPO_1D,
            x_train, y_train,
            x_val, y_val,
            output_path=TeacherConfig.OUTPUT_PATH
        )
    elif DatasetConfig.D_SIGNAL == 2:
        return _run_bayesian_opt(
            topologyTeacher_HPO_2D,
            x_train, y_train,
            x_val, y_val,
            output_path=TeacherConfig.OUTPUT_PATH
        )
    else:
        return _run_bayesian_opt(
            topology_teacher_hpo_2D_SOTA,
            x_train, y_train,
            x_val, y_val,
            output_path=TeacherConfig.OUTPUT_PATH
        )
