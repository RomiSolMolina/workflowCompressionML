

def topologyTeacher1D(hp):

    model = Sequential()
    inputShape = (30, ) #config.INPUT_SHAPE
    
    # Model definition 
    model.add(Dense(
        hp.Int("fc1", min_value=32, max_value=300, step=10),
        kernel_regularizer=l2(0.0001), input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc2", min_value=32, max_value=100, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc3", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc4", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    
    # Output Layer with Softmax activation
    model.add(Dense(4, name='output'))
    model.add(Activation("softmax"))
    
    # Initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate",
                   values=[1e-1, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)
    
    # Compile the model
    model.compile(optimizer=opt, loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model
