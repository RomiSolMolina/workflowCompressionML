import os
import shutil
import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_model_optimization.sparsity.keras import UpdatePruningStep

from src.config.config import DatasetConfig, StudentConfig
from src.topology.student_hpo.student_hpo_1d import modelStudent1D
from src.topology.student_hpo.student_hpo_2d import build_model_hpo_student_2d
from src.topology.student_hpo.student_hpo_2d_sota import build_model_student_hpo_2D_sota
from src.distiller_hypermodel import DistillerHyperModel


def studentBO(
    x_train, y_train, x_val, y_val,
    teacher_model,
    max_trials=StudentConfig.N_ITERATIONS,
    use_kd=True,
    use_quant=True,
    use_prune=True
):
    """
    Runs Bayesian optimization to find best hyperparameters for the student model
    using optional knowledge distillation, quantization, and pruning.
    """

    # === Select proper model builder === #
    if DatasetConfig.D_SIGNAL == 1:
        model_fn = modelStudent1D
    elif DatasetConfig.D_SIGNAL == 2:
        model_fn = build_model_hpo_student_2d
    elif DatasetConfig.D_SIGNAL == 3:
        model_fn = build_model_student_hpo_2D_sota
    else:
        raise ValueError("Unsupported DatasetConfig.D_SIGNAL")

    # === Select hypermodel === #
    if use_kd:
        hypermodel = DistillerHyperModel(
            student_builder_fn=model_fn,
            teacher_model=teacher_model,
            alpha=0.1,
            temperature=10,
            use_prune=use_prune  # optional if your distiller handles it
        )
    else:
        # No KD, build a plain HyperModel-like wrapper
        def student_wrapper(hp):
            model = model_fn(hp)
            return model

        hypermodel = student_wrapper  # Must be callable: def build(hp)

    # === Clean tuning directory === #
    tuner_dir = StudentConfig.OUTPUT_PATH
    if os.path.exists(tuner_dir):
        shutil.rmtree(tuner_dir)

    # === Bayesian Optimization === #
    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        seed=49,
        directory=tuner_dir,
        project_name="student_qat_kd"
    )

    # === Callbacks === #
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="accuracy", factor=0.5, patience=3),
        UpdatePruningStep() if use_prune else None
    ]
    callbacks = [cb for cb in callbacks if cb is not None]

    # === Launch search === #
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

