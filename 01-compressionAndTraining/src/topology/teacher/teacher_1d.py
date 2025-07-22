from tensorflow.keras import layers, models, regularizers


def modelTeacherTopology_1D(bestHP, input_shape=(2031,), n_classes=2):
    """
    Builds a 1D teacher model using functional API.
    Args:
        bestHP (list): List of hyperparameters for layer sizes.
        input_shape (tuple): Shape of the input data.
        n_classes (int): Number of output classes.
    Returns:
        keras.Model: Compiled Keras model.
    """
    print(f"\n1D signal - MLP model")

    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Dense(bestHP[0], activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="fc1")(inputs)
    x = layers.Dense(bestHP[1], activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.Dense(bestHP[2], activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="fc3")(x)
    x = layers.Dense(bestHP[3], activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="fc4")(x)
    outputs = layers.Dense(n_classes, activation='softmax', name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="teacher_1d")
    model.summary()
    return model