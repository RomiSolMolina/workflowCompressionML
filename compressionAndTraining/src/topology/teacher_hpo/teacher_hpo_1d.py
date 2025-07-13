
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def topologyTeacher_HPO_1D(hp):
    """
    HPO search space definition for 1D teacher model.
    """
    model = Sequential()
    input_shape = (2031, )

    # Hidden layers
    model.add(Dense(hp.Int("fc1", min_value=32, max_value=100, step=10),
                    kernel_regularizer=l2(0.0001), input_shape=input_shape))
    model.add(Activation("relu"))

    model.add(Dense(hp.Int("fc2", min_value=32, max_value=100, step=10),
                    kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))

    model.add(Dense(hp.Int("fc3", min_value=10, max_value=50, step=10),
                    kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))

    model.add(Dense(hp.Int("fc4", min_value=10, max_value=50, step=10),
                    kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))

    # Output
    model.add(Dense(2, name="output"))
    model.add(Activation("softmax"))

    # Optimizer
    lr = hp.Choice("learning_rate", values=[1e-1, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model