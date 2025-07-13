# src/distiller_hypermodel.py

from keras_tuner import HyperModel
from src.distillationClassKeras import Distiller
import tensorflow as tf


class DistillerHyperModel(HyperModel):
    def __init__(self, student_builder_fn, teacher_model, alpha=0.1, temperature=10):
        self.student_builder_fn = student_builder_fn
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    def build(self, hp):
        # Construir el modelo del estudiante con los hiperparámetros
        student_model = self.student_builder_fn(hp)

        # Crear el distiller
        distiller = Distiller(student=student_model, teacher=self.teacher_model)

        # Compilar con pérdidas y optimizador definidos
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
            ),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=self.alpha,
            temperature=self.temperature
        )

        return distiller
