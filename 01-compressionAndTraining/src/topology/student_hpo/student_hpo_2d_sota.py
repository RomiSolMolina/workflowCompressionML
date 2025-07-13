# src/topology/student_hpo/student_hpo_2d_sota.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPooling2D, Flatten, Activation
)
from tensorflow.keras.regularizers import l1, l2
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, ConstantSparsity
from qkeras import QConv2DBatchnorm, QDense, QActivation


def build_model_student_hpo_2D_sota(hp, input_shape=(32, 32, 3), n_classes=10,
                                    use_quant=True, use_prune=True):
    """
    Build a student 2D CNN with optional quantization and pruning support.
    """
    # Layer selector
    ConvLayer = QConv2DBatchnorm if use_quant else Conv2D
    DenseLayer = QDense if use_quant else Dense
    ActivationLayer = lambda name: QActivation("quantized_bits(8)", name=name) if use_quant else Activation("relu", name=name)

    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"

    x = x_in = Input(shape=input_shape)

    # Block 1
    for i in range(2):
        conv_args = {
            "filters": int(hp.Choice(f"conv{i+1}", [8, 16, 32])),
            "kernel_size": (3, 3),
            "padding": "same",
            "kernel_initializer": "lecun_uniform",
            "kernel_regularizer": l2(0.0001),
            "name": f"conv{i+1}"
        }
        if use_quant:
            conv_args["kernel_quantizer"] = kernelQ
            conv_args["bias_quantizer"] = biasQ
        x = ConvLayer(**conv_args)(x)
        x = ActivationLayer(f"relu{i+1}")(x)

    x = MaxPooling2D(pool_size=(2, 2), name="pool_0")(x)

    # Block 2
    for i in range(2, 4):
        conv_args = {
            "filters": int(hp.Choice(f"conv{i+1}", [8, 16, 32])),
            "kernel_size": (3, 3),
            "padding": "same",
            "kernel_initializer": "lecun_uniform",
            "kernel_regularizer": l2(0.0001),
            "name": f"conv{i+1}"
        }
        if use_quant:
            conv_args["kernel_quantizer"] = kernelQ
            conv_args["bias_quantizer"] = biasQ
        x = ConvLayer(**conv_args)(x)
        x = ActivationLayer(f"relu{i+1}")(x)

    x = MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)

    x = Flatten()(x)

    # Output Layer
    dense_args = {
        "units": n_classes,
        "kernel_initializer": "lecun_uniform",
        "kernel_regularizer": l1(0.001),
        "name": "output"
    }
    if use_quant:
        dense_args["kernel_quantizer"] = kernelQ
        dense_args["bias_quantizer"] = biasQ

    x = DenseLayer(**dense_args)(x)
    x_out = Activation("softmax", name="softmax")(x)

    model = Model(inputs=x_in, outputs=x_out, name="student_qkeras_2d_sota")

    # === Optional Pruning === #
    if use_prune:
        pruning_params = {
            "pruning_schedule": ConstantSparsity(
                target_sparsity=0.5,
                begin_step=200,
                end_step=2000,
                frequency=100
            )
        }
        model = prune_low_magnitude(model, **pruning_params)

    return model
