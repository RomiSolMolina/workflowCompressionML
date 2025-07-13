import os
import shutil
import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_model_optimization.sparsity.keras import UpdatePruningStep

from src.config import DatasetConfig, StudentConfig
from src.topology.student_hpo.student_hpo_1d import modelStudent1D
from src.topology.student_hpo.student_hpo_2d import build_model_hpo_student_2d
from src.topology.student_hpo.student_hpo_2d_sota import build_model_student_hpo_2D_sota
from src.distiller_hypermodel import DistillerHyperModel


def studentBO(x_train, y_train, x_val, y_val, teacher_model, max_trials=StudentConfig.N_ITERATIONS):
    """
    Runs Bayesian optimization to find best hyperparameters for the student model
    using knowledge distillation and quantization-aware training.
    """

    # Select proper model builder depending on data modality
    if DatasetConfig.D_SIGNAL == 1:
        model_fn = modelStudent1D
    elif DatasetConfig.D_SIGNAL == 2:
        model_fn = build_model_hpo_student_2d
    elif DatasetConfig.D_SIGNAL == 3:
        model_fn = build_model_student_hpo_2D_sota
    else:    
        raise ValueError("Unsupported DatasetConfig.D_SIGNAL")

    # Wrap student and teacher into DistillerHyperModel
    hypermodel = DistillerHyperModel(
        student_builder_fn=model_fn,
        teacher_model=teacher_model,
        alpha=0.1,
        temperature=10
    )

    tuner_dir = StudentConfig.OUTPUT_PATH
    if os.path.exists(tuner_dir):
        shutil.rmtree(tuner_dir)

    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        seed=49,
        directory=tuner_dir,
        project_name="student_qat_kd"
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="accuracy", factor=0.5, patience=3),
        UpdatePruningStep()
    ]

    tuner.search(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=StudentConfig.BATCH_SIZE,
        epochs=StudentConfig.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hp
