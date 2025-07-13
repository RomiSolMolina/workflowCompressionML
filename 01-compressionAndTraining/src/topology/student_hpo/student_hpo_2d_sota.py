import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, MaxPooling2D, Flatten, Activation
)
from tensorflow.keras.regularizers import l1, l2
from qkeras import QConv2DBatchnorm, QDense, QActivation, quantized_bits


def build_model_student_hpo_2D_sota(bestHP, input_shape=(32, 32, 3), n_classes=10):
    """
    Build a quantized 2D CNN model using QKeras for student distillation.

    Args:
        bestHP (list): Best hyperparameter values from Bayesian Optimization.
        input_shape (tuple): Shape of the input images.
        n_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled quantized student model.
    """
    # Quantization settings
    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    activationQ = "quantized_bits(8)"

    # Input
    x = x_in = Input(shape=input_shape)

    # Block 1
    for i in range(2):
        x = QConv2DBatchnorm(
            int(bestHP[i]), kernel_size=(3, 3), padding='same',
            kernel_quantizer=kernelQ, bias_quantizer=biasQ,
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l2(0.0001),
            use_bias=True, name=f'conv{i+1}'
        )(x)
        x = QActivation(activationQ, name=f'relu{i+1}')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool_0')(x)

    # Block 2
    for i in range(2, 4):
        x = QConv2DBatchnorm(
            int(bestHP[i]), kernel_size=(3, 3), padding='same',
            kernel_quantizer=kernelQ, bias_quantizer=biasQ,
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l2(0.0001),
            use_bias=True, name=f'conv{i+1}'
        )(x)
        x = QActivation(activationQ, name=f'relu{i+1}')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)

    # Optional additional blocks can be added here

    # Dense Output
    x = Flatten()(x)
    x = QDense(
        n_classes,
        kernel_quantizer=kernelQ,
        bias_quantizer=activationQ,
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.001),
        name='output'
    )(x)
    x_out = Activation('softmax', name='softmax')(x)

    model = Model(inputs=x_in, outputs=x_out, name='student_qkeras_2d_sota')
    model.summary()

    return model
