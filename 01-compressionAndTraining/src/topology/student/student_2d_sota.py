# src/topology/student/student_2d_sota.py

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import MaxPooling2D, Flatten, Activation
from qkeras import QConv2DBatchnorm, QDense, QActivation


def modelStudent2D_SOTA(bestHP, input_shape=(32, 32, 3), n_classes=10):
    """
    Quantized SOTA 2D Student model using QKeras.
    """
    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    activationQ = "quantized_bits(8)"

    x = x_in = Input(shape=input_shape)

    # Block 1
    x = QConv2DBatchnorm(int(bestHP[0]), (3, 3), padding='same',
                         kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                         kernel_regularizer=regularizers.l2(0.0001), name='conv1')(x)
    x = QActivation(activationQ, name='relu1')(x)

    x = QConv2DBatchnorm(int(bestHP[1]), (3, 3), padding='same',
                         kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                         kernel_regularizer=regularizers.l2(0.0001), name='conv2')(x)
    x = QActivation(activationQ, name='relu2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

    # Block 2
    x = QConv2DBatchnorm(int(bestHP[2]), (3, 3), padding='same',
                         kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                         kernel_regularizer=regularizers.l2(0.0001), name='conv3')(x)
    x = QActivation(activationQ, name='relu3')(x)

    x = QConv2DBatchnorm(int(bestHP[3]), (3, 3), padding='same',
                         kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                         kernel_regularizer=regularizers.l2(0.0001), name='conv4')(x)
    x = QActivation(activationQ, name='relu4')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)

    # Optional blocks removed for lightweight SOTA version

    x = Flatten(name="flatten")(x)

    # Output layer
    x = QDense(n_classes, name='output',
               kernel_quantizer=kernelQ, bias_quantizer=activationQ,
               kernel_regularizer=regularizers.l1(0.001))(x)
    x_out = Activation("softmax", name="softmax")(x)

    model = Model(inputs=x_in, outputs=x_out, name="student_qkeras_sota")
    model.summary()

    return model
