# Libraries

import os

import tensorflow as tf 

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import SGD, Adam

import keras_tuner as kt
from qkeras import *

from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.sparsity import keras as sparsity

from src.distillationClassKeras import *

# from src.modelKDQP import *
# from src.modelKDQP_1D import *
# from src.modelKDQP_2D_SOTA import *
from src.topology.modelStudent_KDQP_2D import *
from src.config import *


def studentCompression_2D(bestHP, images_train, y_train, teacher_baseline, lr):

    qmodel = modelKDQP_2D(bestHP)

    ######## ---------------------------  P -----------------------------------------

    NSTEPS = int(31188*0.9) // 128
    from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.5, begin_step = NSTEPS*2,  end_step = NSTEPS*10, frequency = NSTEPS)} #2000
    studentQ_CNN = prune.prune_low_magnitude(qmodel, **pruning_params)
    train_labels = np.argmax(y_train, axis=1)

    # ######## ---------------------------  KD + QAT -----------------------------------------


    distilledCNN = Distiller(student=studentQ_CNN, teacher=teacher_baseline)

    train = True
    if train:
        adam = Adam(lr)
        distilledCNN.compile(
        optimizer=adam,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1, 
        temperature=10,
    )
    callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, verbose=1),
                ]  
    callbacks.append(pruning_callbacks.UpdatePruningStep())

    # Distill teacher to student 

    # Commented for mobileNEtV2
    # train_labels = np.argmax(y_train, axis=1)

    history = distilledCNN.fit(images_train, train_labels, batch_size = 16, epochs= 64, callbacks = callbacks)
    
    # For MobileNetV2 compression
    #history = distilledCNN.fit(images_train, train_labels, batch_size = 32, epochs= 32, callbacks = callbacks)

    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)
    model = strip_pruning(distilledCNN.student)
    model.summary()

    model.save(PATH_MODEL_STUDENT)

    return model

