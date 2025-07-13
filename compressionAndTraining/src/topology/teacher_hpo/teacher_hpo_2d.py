
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def topologyTeacher_HPO_2D(hp):
    """
    HPO search space definition for 2D teacher model.
    """
    model = Sequential()
    input_shape = (80, 80, 3)

    # Block 1
    model.add(Conv2D(hp.Int("conv_1", 32, 64, step=32), (3, 3), padding="same", kernel_regularizer=l2(0.0001), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(hp.Int("conv_2", 32, 64, step=32), (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(hp.Int("conv_3", 32, 128, step=32), (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(hp.Int("conv_4", 32, 128, step=32), (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(hp.Int("conv_5", 32, 128, step=32), (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(hp.Int("conv_6", 32, 128, step=32), (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 4
    model.add(Conv2D(hp.Int("conv_7", 32, 128, step=32), (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(hp.Int("conv_8", 32, 128, step=32), (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected
    model.add(Dense(hp.Int("fc1", 10, 50, step=10), kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(hp.Int("fc2", 10, 50, step=10), kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(hp.Int("fc3", 10, 50, step=10), kernel_regularizer=l2(0.0001)))

    # Output
    model.add(Dense(3, activation="softmax"))

    # Compile
    lr = hp.Choice("learning_rate", values=[1e-1, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model