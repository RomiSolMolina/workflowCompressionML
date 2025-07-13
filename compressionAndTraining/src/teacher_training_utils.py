from tensorflow.keras.optimizers import Adam


def train_teacher_model(
    model_fn,
    bestHP,
    x_train,
    y_train,
    x_val,
    y_val,
    lr,
    batch_size=128,
    epochs=32,
    input_shape=None,
    n_classes=None
):
    """
    Generic training function for teacher models (1D, 2D, SOTA).

    Args:
        model_fn (function): The model constructor function.
        bestHP (list): Hyperparameters.
        x_train, y_train: Training data.
        x_val, y_val: Validation data.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of training epochs.
        input_shape (tuple): Optional input shape for model constructor.
        n_classes (int): Optional number of output classes.

    Returns:
        model: Trained Keras model.
    """

    # Flexible call to model_fn depending on signature
    if input_shape is not None and n_classes is not None:
        model = model_fn(bestHP, input_shape=input_shape, n_classes=n_classes)
    else:
        model = model_fn(bestHP)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        # validation_data=(x_val, y_val),
        validation_split=0.2, 
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    return  model,  history
