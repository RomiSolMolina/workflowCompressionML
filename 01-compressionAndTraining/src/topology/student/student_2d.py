# src/topology/student/student_2d.py

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation
from tensorflow.keras.layers import BatchNormalization
from qkeras import QConv2DBatchnorm, QDense, QActivation

def modelStudent2D(bestHP, input_shape=(80, 80, 3), n_classes=2, use_quant=True):
    """
    Define a 2D student model with optional quantization using QKeras.
    Parameters:
        bestHP: list of hyperparameter values
        input_shape: shape of input images
        n_classes: number of output classes
        use_quant: whether to use quantized layers
    Returns:
        Compiled model
    """
    
    x = x_in = Input(shape=input_shape)

    # === Select layers based on quantization === #
    ConvLayer = QConv2DBatchnorm if use_quant else Conv2D
    DenseLayer = QDense if use_quant else Dense
    ActivationLayer = lambda name: QActivation("quantized_bits(8)", name=name) if use_quant else Activation("relu", name=name)

    # Quantization parameters
    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"

    def conv_args(filters, name):
        args = {
            "filters": int(filters),
            "kernel_size": (3, 3),
            "padding": "same",
            "kernel_initializer": "lecun_uniform",
            "kernel_regularizer": regularizers.l2(0.0001),
            "use_bias": True,
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
    x = MaxPooling2D(pool_size=(2, 2), name="pool_0")(x)

    # === Conv Block 2 === #
    x = ConvLayer(**conv_args(bestHP[2], "conv3"))(x)
    x = ActivationLayer("relu3")(x)

    x = ConvLayer(**conv_args(bestHP[3], "conv4"))(x)
    x = ActivationLayer("relu4")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)

    x = Flatten()(x)

    # === Output Layer === #
    dense_args = {
        "units": n_classes,
        "kernel_initializer": "lecun_uniform",
        "kernel_regularizer": regularizers.l1(0.001),
        "name": "output"
    }
    if use_quant:
        dense_args["kernel_quantizer"] = kernelQ
        dense_args["bias_quantizer"] = biasQ

    x = DenseLayer(**dense_args)(x)
    x_out = Activation("softmax", name="softmax")(x)

    model = Model(inputs=x_in, outputs=x_out, name="student_qkeras_2d" if use_quant else "student_fp_2d")
    model.summary()
    return model
