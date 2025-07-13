from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Activation, BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def topology_teacher_hpo_2D_SOTA(hp):
    input_shape = (32, 32, 3)
    model = Sequential()

    # Block 1
    model.add(Conv2D(hp.Int("conv_1", 32, 64, step=32), (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(hp.Int("conv_2", 32, 64, step=32), (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(hp.Int("conv_3", 32, 128, step=32), (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(hp.Int("conv_4", 32, 128, step=32), (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(hp.Int("conv_5", 32, 128, step=32), (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(hp.Int("conv_6", 32, 128, step=32), (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 4
    model.add(Conv2D(hp.Int("conv_7", 32, 128, step=32), (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(hp.Int("conv_8", 32, 128, step=32), (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(hp.Int("fc1", 10, 50, step=10), kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(hp.Int("fc2", 10, 50, step=10), kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(hp.Int("fc3", 10, 50, step=10), kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))

    model.add(Dense(10, activation="softmax"))

    lr = hp.Choice("learning_rate", values=[1e-1, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
