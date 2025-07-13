# src/topology/student/student_2d_lowrank.py

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation
from qkeras import QConv2D, QDense, QActivation
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude

def low_rank_conv_block(x, filters, kernel_size, name_prefix, use_quant, use_prune, pruning_params):
    """
    Low-rank separable convolution block (Depthwise + Pointwise)
    """
    Conv = QConv2D if use_quant else Conv2D
    ActivationLayer = lambda name: QActivation("quantized_relu(8)", name=name) if use_quant else Activation("relu", name=name)

    # Depthwise conv
    x = prune_low_magnitude(DepthwiseConv2D(
        kernel_size=kernel_size, padding='same',
        depth_multiplier=1,
        kernel_regularizer=regularizers.l2(0.0001),
        name=f'{name_prefix}_depthwise'
    ), **pruning_params)(x) if use_prune else DepthwiseConv2D(
        kernel_size=kernel_size, padding='same',
        depth_multiplier=1,
        kernel_regularizer=regularizers.l2(0.0001),
        name=f'{name_prefix}_depthwise'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = ActivationLayer(f'{name_prefix}_act1')(x)

    # Pointwise conv
    x = prune_low_magnitude(Conv(
        filters, kernel_size=(1, 1), padding='same',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=regularizers.l2(0.0001),
        name=f'{name_prefix}_pointwise'
    ), **pruning_params)(x) if use_prune else Conv(
        filters, kernel_size=(1, 1), padding='same',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=regularizers.l2(0.0001),
        name=f'{name_prefix}_pointwise'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = ActivationLayer(f'{name_prefix}_act2')(x)

    return x

def modelStudentLowRank2D(bestHP, input_shape=(32, 32, 3), n_classes=10,
                          use_quant=True, use_prune=False):
    """
    2D Student Model: Low-Rank + Quantization + Optional Pruning
    """
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=0.5,
            begin_step=2000,
            end_step=10000,
            frequency=100
        )
    }

    inputs = Input(shape=input_shape, name="input")
    x = inputs

    # Block 1
    x = low_rank_conv_block(x, int(bestHP[0]), (3, 3), "block1", use_quant, use_prune, pruning_params)
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = low_rank_conv_block(x, int(bestHP[1]), (3, 3), "block2", use_quant, use_prune, pruning_params)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Flatten + Dense
    x = Flatten(name="flatten")(x)

    DenseLayer = QDense if use_quant else tf.keras.layers.Dense
    ActivationLayer = lambda name: QActivation("quantized_relu(8)", name=name) if use_quant else Activation("relu", name=name)

    x = DenseLayer(int(bestHP[2]), kernel_initializer="lecun_uniform", name="fc1")(x)
    x = ActivationLayer("relu_fc1")(x)

    x = DenseLayer(n_classes, name="output")(x)
    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=inputs, outputs=x, name="student_lowrank_qprune")

    return model
