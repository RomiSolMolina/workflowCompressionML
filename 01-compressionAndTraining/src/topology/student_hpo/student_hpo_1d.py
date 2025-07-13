# src/topology/student_hpo/student_hpo_1d.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, ConstantSparsity
from qkeras import QDense, QActivation
import tensorflow as tf

from src.config.config import DatasetConfig

def modelStudent1D(hp, use_quant=True, use_prune=True):
    """
    1D Student model with optional quantization and pruning.
    """
    input_shape = (DatasetConfig.SAMPLES,)
    num_classes = DatasetConfig.nLabels_1D

    # Choose layer type
    DenseLayer = QDense if use_quant else Dense
    ActivationLayer = lambda name: QActivation("quantized_relu(8)", name=name) if use_quant else Activation("relu", name=name)

    # Quantization parameters
    kernelQ = "quantized_bits(8,2,alpha=1)"
    biasQ = "quantized_bits(8,2,alpha=1)"

    # === Build model === #
    model = Sequential()
    dense_args = {
        "units": hp.Int("fc1", min_value=5, max_value=20, step=10),
        "kernel_initializer": "lecun_uniform",
        "kernel_regularizer": tf.keras.regularizers.l1(0.001),
        "input_shape": input_shape
    }
    if use_quant:
        dense_args["kernel_quantizer"] = kernelQ
        dense_args["bias_quantizer"] = biasQ

    model.add(DenseLayer(**dense_args))
    model.add(ActivationLayer(name="relu1"))

    output_args = {
        "units": num_classes,
        "kernel_initializer": "lecun_uniform",
        "kernel_regularizer": tf.keras.regularizers.l1(0.001),
        "name": "output"
    }
    if use_quant:
        output_args["kernel_quantizer"] = kernelQ
        output_args["bias_quantizer"] = biasQ

    model.add(DenseLayer(**output_args))
    model.add(Activation("softmax", name="softmax"))

    # === Optional pruning === #
    if use_prune:
        n_samples = 31188
        batch_size = 128
        n_steps_per_epoch = int(n_samples * 0.9) // batch_size
        pruning_params = {
            "pruning_schedule": ConstantSparsity(
                target_sparsity=0.3,
                begin_step=n_steps_per_epoch * 2,
                end_step=n_steps_per_epoch * 10,
                frequency=n_steps_per_epoch
            )
        }
        model = prune_low_magnitude(model, **pruning_params)

    # === Compile model === #
    lr = hp.Choice("learning_rate", values=[1e-1, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
