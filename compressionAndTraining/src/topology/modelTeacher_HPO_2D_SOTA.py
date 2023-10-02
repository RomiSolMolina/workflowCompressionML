

def topology2DSOTA(hp):
    INPUT_SHAPE = (32, 32, 3)
    
    model = Sequential()


# Model definition 
# First block
    model.add(Conv2D(
        hp.Int("conv_1", min_value=32, max_value=64, step=32),
        (3, 3), padding="same",
        kernel_regularizer=l2(0.0001), input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
        
    model.add(Conv2D(
        hp.Int("conv_2", min_value=32, max_value=64, step=32),
        (3, 3), padding="same",
        kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))        

# Second block
    model.add(Conv2D(
        hp.Int("conv_3", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(
        hp.Int("conv_4", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))          
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))                 
 
 # Third block
    model.add(Conv2D(
        hp.Int("conv_5", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(
        hp.Int("conv_6", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))          
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))   

 # Fourth block
    model.add(Conv2D(
        hp.Int("conv_7", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))
    model.add(Conv2D(
        hp.Int("conv_8", min_value=32, max_value=128, step=32),
        (3, 3), padding="same", kernel_regularizer=l2(0.0001)))          
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))   


    model.add(Flatten())
    
    model.add(Dense(
        hp.Int("fc1", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc2", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dense(
        hp.Int("fc3", min_value=10, max_value=50, step=10),
        kernel_regularizer=l2(0.0001)))
    
    model.add(Activation("relu"))
    
    # Output Layer with Softmax activation
    model.add(Dense(10, activation='softmax')) 
     
    # Initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate",
                   values=[1e-1, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)
    
    # Compile the model
    model.compile(optimizer=opt, loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model
