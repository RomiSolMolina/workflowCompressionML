
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2


def topologyTeacher_2D_SOTA(bestHP, input_shape=(32, 32, 3), n_classes=10):
    """
    Functional API definition of a 2D teacher CNN model (SOTA-style) using best hyperparameters.
    """

    x = x_in = Input(shape=input_shape)

    # --- Conv Block 1 ---
    x = layers.Conv2D(bestHP[0], (3, 3), padding="same", kernel_regularizer=l2(1e-4), name="conv_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="relu1")(x)

    x = layers.Conv2D(bestHP[1], (3, 3), padding="same", kernel_regularizer=l2(1e-4), name="conv_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # --- Conv Block 2 ---
    x = layers.Conv2D(bestHP[2], (3, 3), padding="same", kernel_regularizer=l2(1e-4), name="conv_3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="relu3")(x)

    x = layers.Conv2D(bestHP[3], (3, 3), padding="same", kernel_regularizer=l2(1e-4), name="conv_4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # --- Conv Block 3 ---
    x = layers.Conv2D(bestHP[4], (3, 3), padding="same", kernel_regularizer=l2(1e-4), name="conv_5")(x)
    x = layers.Conv2D(bestHP[5], (3, 3), padding="same", kernel_regularizer=l2(1e-4), name="conv_6")(x)
    x = layers.Activation("relu", name="relu5")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)

    # --- Fully Connected ---
    x = layers.Dense(bestHP[6], kernel_regularizer=l2(1e-4), name="fc1")(x)
    x = layers.Activation("relu", name="relu6")(x)

    x = layers.Dense(bestHP[7], kernel_regularizer=l2(1e-4), name="fc2")(x)
    x = layers.Activation("relu", name="relu7")(x)

    x = layers.Dense(bestHP[8], kernel_regularizer=l2(1e-4), name="fc3")(x)
    x = layers.Activation("relu", name="relu8")(x)

    # --- Output ---
    x = layers.Dense(n_classes, kernel_regularizer=l2(1e-4), name="output")(x)
    x_out = layers.Activation("softmax", name="softmax")(x)

    model = Model(inputs=x_in, outputs=x_out, name="teacherCNN_SOTA")
    model.summary()

    return model