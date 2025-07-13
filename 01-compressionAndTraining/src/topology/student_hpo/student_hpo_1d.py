from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, ConstantSparsity
from qkeras import QDense, QActivation
import tensorflow as tf

from src.config import DatasetConfig


def modelStudent1D(hp):
    """
    Topology definition of the 1D quantized & pruned student model.
    """
    input_shape = (DatasetConfig.SAMPLES,)
    num_classes = DatasetConfig.nLabels_1D

    kernelQ = "quantized_bits(8,2,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    activationQ = "quantized_relu(8)"

    model = Sequential()
    model.add(QDense(
        hp.Int("fc1", min_value=5, max_value=20, step=10),
        kernel_quantizer=kernelQ,
        bias_quantizer=biasQ,
        kernel_initializer='lecun_uniform',
        kernel_regularizer=tf.keras.regularizers.l1(0.001),
        input_shape=input_shape
    ))
    model.add(QActivation(activationQ, name='relu1'))
    model.add(QDense(
        num_classes,
        name='output',
        kernel_quantizer=kernelQ,
        bias_quantizer=biasQ,
        kernel_initializer='lecun_uniform',
        kernel_regularizer=tf.keras.regularizers.l1(0.001)
    ))
    model.add(Activation('softmax', name='softmax'))

    # Pruning wrapper
    n_samples = 31188
    batch_size = 128
    n_steps_per_epoch = int(n_samples * 0.9) // batch_size
    pruning_params = {
        "pruning_schedule": ConstantSparsity(
            target_sparsity=0.3,
            begin_step=n_steps_per_epoch * 2,
            end_step=n_steps_per_epoch * 10,
            frequency=n_steps_per_epoch
        )
    }

    model = prune_low_magnitude(model, **pruning_params)

    # Learning rate choice
    lr = hp.Choice("learning_rate", values=[1e-1, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
