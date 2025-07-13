# src/topology/student/student_1d.py

from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from qkeras import QDense, QActivation

def modelStudent_1D(bestHP, input_shape=(2031,), n_classes=2, use_quant=True, use_prune=False):
    """
    Define a 1D student model with quantization support (QKeras).
    """
    # Puedes usar los flags en el futuro si quieres condicionar capas
    kernelQ = "quantized_bits(8,1,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    activationQ = "quantized_bits(8)"

    model = Sequential(name="student_qkeras_1d")

    model.add(Input(shape=input_shape))

    model.add(QDense(bestHP[0],
                     kernel_quantizer=kernelQ,
                     bias_quantizer=biasQ,
                     kernel_initializer='lecun_uniform',
                     kernel_regularizer=regularizers.l1(0.001),
                     name="fc1"))
    model.add(QActivation(activationQ, name="relu1"))

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

    model.add(QDense(n_classes,
                     kernel_quantizer=kernelQ,
                     bias_quantizer=biasQ,
                     kernel_initializer='lecun_uniform',
                     kernel_regularizer=regularizers.l1(0.001),
                     name="output"))
    model.add(Activation("softmax", name="softmax"))

    return model
