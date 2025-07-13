# src/topology/student/student_1d.py

from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from qkeras import QDense, QActivation

def modelKDQP_1D(bestHP, input_shape=(2031,), n_classes=2):
    """
    Define a 1D student model with quantization support (QKeras).
    """
    # Quantization formats
    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    activationQ = "quantized_bits(8)"

    model = Sequential(name="student_qkeras_1d")

    # Input
    model.add(Input(shape=input_shape))

    # FC Layer 1
    model.add(QDense(bestHP[0],
                     kernel_quantizer=kernelQ,
                     bias_quantizer=biasQ,
                     kernel_initializer='lecun_uniform',
                     kernel_regularizer=regularizers.l1(0.001),
                     name="fc1"))
    model.add(QActivation(activationQ, name="relu1"))

    # Optional: Add more layers if bestHP has more
    if len(bestHP) > 1:
        model.add(QDense(bestHP[1],
                         kernel_quantizer=kernelQ,
                         bias_quantizer=biasQ,
                         kernel_initializer='lecun_uniform',
                         kernel_regularizer=regularizers.l1(0.001),
                         name="fc2"))
        model.add(QActivation(activationQ, name="relu2"))

    if len(bestHP) > 2:
        model.add(QDense(bestHP[2],
                         kernel_quantizer=kernelQ,
                         bias_quantizer=biasQ,
                         kernel_initializer='lecun_uniform',
                         kernel_regularizer=regularizers.l1(0.001),
                         name="fc3"))
        model.add(QActivation(activationQ, name="relu3"))

    # Output
    model.add(QDense(n_classes,
                     kernel_quantizer=kernelQ,
                     bias_quantizer=biasQ,
                     kernel_initializer='lecun_uniform',
                     kernel_regularizer=regularizers.l1(0.001),
                     name="output"))
    model.add(Activation("softmax", name="softmax"))

    return model
