
from tensorflow.keras import layers, models, regularizers


def topologyTeacher_2D(bestHP, input_shape=(80, 80, 3), n_classes=10):
    """
    Builds a 2D CNN teacher model using functional API.
    Args:
        bestHP (list): List of hyperparameters for conv and dense layers.
        input_shape (tuple): Input image shape.
        n_classes (int): Number of output classes.
    Returns:
        keras.Model: Compiled Keras model.
    """

    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv2D(bestHP[0], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name='conv1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="relu1")(x)

    x = layers.Conv2D(bestHP[1], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(bestHP[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name='conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="relu3")(x)

    x = layers.Conv2D(bestHP[3], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name='conv4')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(bestHP[4], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name='conv5')(x)
    x = layers.Conv2D(bestHP[5], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name='conv6')(x)
    x = layers.Activation("relu", name="relu5")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(bestHP[6], activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.Dense(bestHP[7], activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.Dense(bestHP[8], activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc3")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="teacher_2d")
    model.summary()
    return model