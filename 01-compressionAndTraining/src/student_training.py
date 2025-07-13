# src/student_training.py

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity.keras import (
    prune_low_magnitude,
    strip_pruning,
    ConstantSparsity,
    UpdatePruningStep
)
from qkeras.utils import _add_supported_quantized_objects

from src.distillationClassKeras import Distiller


def train_student_model(model_fn, bestHP, teacher_model,
                        x_train, y_train, x_val, y_val,
                        lr, input_shape, n_classes,
                        batch_size=128, epochs=64,
                        use_kd=True, use_quant=True, use_prune=True, use_lowrank=False):
    """
    Train student with optional quantization, pruning, knowledge distillation, and low-rank compression.
    """

    # === Build model === #
    student_model = model_fn(bestHP, use_quant=use_quant, use_prune=False, use_lowrank=use_lowrank)

    # === Apply pruning if selected === #
    if use_prune:
        print("Pruning compression is enabled")
        NSTEPS = len(x_train) // batch_size
        pruning_params = {
            "pruning_schedule": ConstantSparsity(
                target_sparsity=0.5,
                begin_step=NSTEPS * 2,
                end_step=NSTEPS * 10,
                frequency=NSTEPS
            )
        }
        student_model = prune_low_magnitude(student_model, **pruning_params)

    # === Apply low-rank compression if selected === #
    if use_lowrank:
        print("Low-rank compression is enabled (apply custom decomposition if defined)")
        
        # student_model = apply_low_rank_approx(student_model)

    # === Compile === #
    if use_kd:
        print("KD compression is enabled")
        distiller = Distiller(student=student_model, teacher=teacher_model)
        distiller.compile(
            optimizer=Adam(learning_rate=lr),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10,
        )
        training_model = distiller
    else:
        student_model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )
        training_model = student_model

    # === Callbacks === #
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        UpdatePruningStep() if use_prune else None
    ]
    callbacks = [cb for cb in callbacks if cb is not None]

    # === Train === #
    history = training_model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # === Finalize student model === #
    final_model = strip_pruning(student_model) if use_prune else student_model

    # Register QKeras layers (optional)
    _add_supported_quantized_objects({})

    final_model.summary()

    return final_model, history
