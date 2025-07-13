from keras_tuner import HyperModel
from src.distillationClassKeras import Distiller
import tensorflow as tf


class DistillerHyperModel(HyperModel):
    def __init__(self, student_builder_fn, teacher_model, alpha=0.1, temperature=10, use_prune=False):
        self.student_builder_fn = student_builder_fn
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.use_prune = use_prune

    def build(self, hp):
        # Pass use_prune if student_builder_fn expects it
        student_model = self.student_builder_fn(hp, use_prune=self.use_prune)

        distiller = Distiller(student=student_model, teacher=self.teacher_model)

        distiller.compile(
            optimizer=student_model.optimizer if hasattr(student_model, "optimizer") else tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=self.alpha,
            temperature=self.temperature
        )

        return distiller
