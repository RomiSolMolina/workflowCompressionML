

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
                        batch_size=128, epochs=64):
    """
    Train student with pruning + quantization-aware training (QAT) and knowledge distillation (KD).
    """

    # Build initial model
    student_model = model_fn(bestHP)

    # Calculate pruning schedule steps
    NSTEPS = len(x_train) // batch_size
    pruning_params = {
         "pruning_schedule": ConstantSparsity(
            target_sparsity=0.5,
            begin_step=NSTEPS * 2,
            end_step=NSTEPS * 10,
            frequency=NSTEPS
        )
    }

    # Apply pruning wrapper
    student_model = prune_low_magnitude(student_model, **pruning_params)

    # Wrap with distiller for KD
    distiller = Distiller(student=student_model, teacher=teacher_model)

    distiller.compile(
        optimizer=Adam(learning_rate=lr),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    # Prepare callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        UpdatePruningStep()
    ]

    # Train
    history = distiller.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Strip pruning and prepare final model
    final_model = strip_pruning(distiller.student)

    # Enable QKeras layer saving support
    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)

    final_model.summary()
    
    return final_model, history
