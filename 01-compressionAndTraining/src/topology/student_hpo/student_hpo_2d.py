# src/topology/student_hpo/student_hpo_2d.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, MaxPooling2D, Flatten, Activation
)
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity import keras as sparsity
from qkeras import QConv2DBatchnorm, QDense, QActivation


def build_model_hpo_student_2d(hp, use_quant=True, use_prune=True):
    """
    Build a 2D student CNN model with optional quantization and pruning.
    """
    # Layer selection
    ConvLayer = QConv2DBatchnorm if use_quant else Conv2D
    DenseLayer = QDense if use_quant else Dense
    ActivationLayer = lambda name: QActivation("quantized_relu(8,2)", name=name) if use_quant else Activation("relu", name=name)

    kernelQ = "quantized_bits(8,2,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"

    model = Sequential()

    # === Convolutional Blocks === #
    for i in range(8):
        filters = hp.Int(f"conv_{i+1}", 1, 10, 1)
        conv_args = {
            "filters": filters,
            "kernel_size": (3, 3),
            "padding": "same",
            "kernel_initializer": "lecun_uniform",
            "kernel_regularizer": l2(0.0001),
            "name": f"conv_{i+1}"
        }
        if i == 0:
            conv_args["input_shape"] = (80, 80, 3)
        if use_quant:
            conv_args["kernel_quantizer"] = kernelQ
            conv_args["bias_quantizer"] = biasQ
        model.add(ConvLayer(**conv_args))
        model.add(ActivationLayer(f"relu{i+1}"))

        # After conv_2 and conv_4 and conv_8: add max pooling
        if i in [1, 3, 7]:
            model.add(MaxPooling2D(pool_size=(2, 2), name=f"pool_{i+1}"))

    model.add(Flatten())

    # === Dense Layers === #
    for j in range(1, 4):
        units = hp.Int(f"fc{j}", 5, 10, 10)
        dense_args = {
            "units": units,
            "kernel_initializer": "lecun_uniform",
            "kernel_regularizer": l1(0.001),
            "name": f"fc{j}"
        }
        if use_quant:
            dense_args["kernel_quantizer"] = kernelQ
            dense_args["bias_quantizer"] = biasQ
        model.add(DenseLayer(**dense_args))
        model.add(ActivationLayer(f"relu{j}_D"))

    # Output Layer
    output_args = {
        "units": 2,
        "kernel_initializer": "lecun_uniform",
        "kernel_regularizer": l1(0.001),
        "name": "output"
    }
    if use_quant:
        output_args["kernel_quantizer"] = kernelQ
        output_args["bias_quantizer"] = biasQ
    model.add(DenseLayer(**output_args))
    model.add(Activation("softmax", name="softmax"))

    # === Optional Pruning === #
    if use_prune:
        NSTEPS = int(31188 * 0.9) // 128
        pruning_params = {
            "pruning_schedule": sparsity.ConstantSparsity(
                target_sparsity=0.5,
                begin_step=NSTEPS * 2,
                end_step=NSTEPS * 10,
                frequency=NSTEPS
            )
        }
        model = sparsity.prune_low_magnitude(model, **pruning_params)

    # === Compile === #
    lr = hp.Choice("learning_rate", [1e-1, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
