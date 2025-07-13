# src/topology/student/student_2d.py

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import MaxPooling2D, Flatten, Activation
from qkeras import QConv2DBatchnorm, QDense, QActivation


def modelStudent2D(bestHP, input_shape=(80, 80, 3), n_classes=2):
    """
    Define a 2D student model using QKeras for quantization.
    Parameters:
        bestHP: list of hyperparameter values
        input_shape: shape of input images
        n_classes: number of output classes
    Returns:
        Compiled QKeras model
    """

    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    activationQ = "quantized_bits(8)"

    x = x_in = Input(shape=input_shape)

    # Block 1
    x = QConv2DBatchnorm(int(bestHP[0]), kernel_size=(3, 3), padding="same",
                         kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                         kernel_initializer="lecun_uniform", kernel_regularizer=regularizers.l2(0.0001),
                         use_bias=True, name="conv1")(x)
    x = QActivation(activationQ, name="relu1")(x)

    x = QConv2DBatchnorm(int(bestHP[1]), kernel_size=(3, 3), padding="same",
                         kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                         kernel_initializer="lecun_uniform", kernel_regularizer=regularizers.l2(0.0001),
                         use_bias=True, name="conv2")(x)
    x = QActivation(activationQ, name="relu2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool_0")(x)

    # Block 2
    x = QConv2DBatchnorm(int(bestHP[2]), kernel_size=(3, 3), padding="same",
                         kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                         kernel_initializer="lecun_uniform", kernel_regularizer=regularizers.l2(0.0001),
                         use_bias=True, name="conv3")(x)
    x = QActivation(activationQ, name="relu3")(x)

    x = QConv2DBatchnorm(int(bestHP[3]), kernel_size=(3, 3), padding="same",
                         kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                         kernel_initializer="lecun_uniform", kernel_regularizer=regularizers.l2(0.0001),
                         use_bias=True, name="conv4")(x)
    x = QActivation(activationQ, name="relu4")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)

    x = Flatten()(x)

    # Output Layer
    x = QDense(n_classes, name="output",
               kernel_quantizer=kernelQ, bias_quantizer=activationQ,
               kernel_initializer="lecun_uniform", kernel_regularizer=regularizers.l1(0.001))(x)
    x_out = Activation("softmax", name="softmax")(x)

    model = Model(inputs=x_in, outputs=x_out, name="student_qkeras_2d")
    model.summary()
    return model
