# src/topology/student/student_2d_sota.py

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, BatchNormalization
from qkeras import QConv2DBatchnorm, QDense, QActivation


def modelStudent2D_SOTA(bestHP, input_shape=(32, 32, 3), n_classes=10, use_quant=True):
    """
    2D Student model (SOTA version) with optional QKeras quantization.
    """
    x = x_in = Input(shape=input_shape)

    # === Choose layer types === #
    ConvLayer = QConv2DBatchnorm if use_quant else Conv2D
    DenseLayer = QDense if use_quant else Dense
    ActivationLayer = lambda name: QActivation("quantized_bits(8)", name=name) if use_quant else Activation("relu", name=name)

    # Quantization params
    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"

    def conv_args(filters, name):
        args = {
            "filters": int(filters),
            "kernel_size": (3, 3),
            "padding": "same",
            "kernel_regularizer": regularizers.l2(0.0001),
            "name": name
        }
        if use_quant:
            args["kernel_quantizer"] = kernelQ
            args["bias_quantizer"] = biasQ
        return args

    # === Conv Block 1 === #
    x = ConvLayer(**conv_args(bestHP[0], "conv1"))(x)
    x = ActivationLayer("relu1")(x)

    x = ConvLayer(**conv_args(bestHP[1], "conv2"))(x)
    x = ActivationLayer("relu2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

    # === Conv Block 2 === #
    x = ConvLayer(**conv_args(bestHP[2], "conv3"))(x)
    x = ActivationLayer("relu3")(x)

    x = ConvLayer(**conv_args(bestHP[3], "conv4"))(x)
    x = ActivationLayer("relu4")(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)

    x = Flatten(name="flatten")(x)

    # === Output === #
    dense_args = {
        "units": n_classes,
        "kernel_regularizer": regularizers.l1(0.001),
        "name": "output"
    }
    if use_quant:
        dense_args["kernel_quantizer"] = kernelQ
        dense_args["bias_quantizer"] = biasQ

    x = DenseLayer(**dense_args)(x)
    x_out = Activation("softmax", name="softmax")(x)

    model = Model(inputs=x_in, outputs=x_out, name="student_sota_quant" if use_quant else "student_sota_fp")
    model.summary()
    return model
