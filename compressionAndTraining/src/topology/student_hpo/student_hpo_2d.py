

def build_model_hpo_student_2d(hp):
    """
    Build a 2D quantized + pruned CNN student model using QKeras and TF pruning.
    """

    kernelQ = "quantized_bits(8,2,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"
    activationQ = 'quantized_relu(8,2)'
    CONSTANT_SPARSITY = 0.5

    model = Sequential()

    # Block 1
    model.add(QConv2DBatchnorm(hp.Int("conv_1", 1, 10, 1), (3,3), padding='same',
                               kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001),
                               input_shape=(80, 80, 3)))
    model.add(QActivation(activationQ, name='relu1'))

    model.add(QConv2DBatchnorm(hp.Int("conv_2", 1, 10, 1), (3,3), padding='same',
                               kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)))
    model.add(QActivation(activationQ, name='relu2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(QConv2DBatchnorm(hp.Int("conv_3", 1, 10, 1), (3,3), padding='same',
                               kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)))
    model.add(QActivation(activationQ, name='relu3'))

    model.add(QConv2DBatchnorm(hp.Int("conv_4", 1, 10, 1), (3,3), padding='same',
                               kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)))
    model.add(QActivation(activationQ, name='relu4'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(QConv2DBatchnorm(hp.Int("conv_5", 1, 10, 1), (3,3), padding='same',
                               kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)))
    model.add(QActivation(activationQ, name='relu5'))

    model.add(QConv2DBatchnorm(hp.Int("conv_6", 1, 10, 1), (3,3), padding='same',
                               kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)))
    model.add(QActivation(activationQ, name='relu6'))

    # Block 4
    model.add(QConv2DBatchnorm(hp.Int("conv_7", 1, 10, 1), (3,3), padding='same',
                               kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)))
    model.add(QActivation(activationQ, name='relu7'))

    model.add(QConv2DBatchnorm(hp.Int("conv_8", 1, 10, 1), (3,3), padding='same',
                               kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                               kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)))
    model.add(QActivation(activationQ, name='relu8'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # Dense Block
    model.add(QDense(hp.Int("fc1", 5, 10, 10),
                     kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)))
    model.add(QActivation('quantized_relu(8)', name='relu1_D'))

    model.add(QDense(hp.Int("fc2", 5, 10, 10),
                     kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)))
    model.add(QActivation('quantized_relu(8)', name='relu2_D'))

    model.add(QDense(hp.Int("fc3", 5, 10, 10),
                     kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)))
    model.add(QActivation('quantized_relu(8)', name='relu3_D'))

    # Output
    model.add(QDense(2, name='output',
                     kernel_quantizer=kernelQ, bias_quantizer=biasQ,
                     kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)))
    model.add(Activation('softmax', name='softmax'))

    # Learning rate
    lr = hp.Choice("learning_rate", [1e-1, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)

    # Pruning
    NSTEPS = int(31188 * 0.9) // 128
    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(
            target_sparsity=CONSTANT_SPARSITY,
            begin_step=NSTEPS * 2,
            end_step=NSTEPS * 10,
            frequency=NSTEPS
        )
    }

    model = prune.prune_low_magnitude(model, **pruning_params)

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
