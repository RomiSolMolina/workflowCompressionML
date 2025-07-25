


def gpuInitialization():

# GPU initialization
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    import tensorflow as tf
    print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)




# Pixel normalization for datasets composed of images

def normalizationPix(train, test):
    """
    The function performs pixel normalization
    
    """
    # convert from integers to floats
    train_ = train.astype('float32')
    test_ = test.astype('float32')
    # normalize to range 0-1
    train_ = train_ / 255.0
    test_ = test_ / 255.0
    # return normalized images
    
    return train_, test_


import re

def bestHPBO_computation(bestHP_BO, CONV_VAR, FC_VAR, UPPER_CONV=10, UPPER_FC=10):
    """
    Extract best hyperparameters dynamically from the tuning result,
    checking which keys actually exist.
    """
    bestHP = []

    # Try conv layers
    for i in range(1, UPPER_CONV + 1):
        key = f"{CONV_VAR}{i}"
        if key in bestHP_BO:
            bestHP.append(bestHP_BO[key])
        else:
            break  # No more conv layers

    # Try fc layers
    for j in range(1, UPPER_FC + 1):
        key = f"{FC_VAR}{j}"
        if key in bestHP_BO:
            bestHP.append(bestHP_BO[key])
        else:
            break  # No more fc layers

    print("[INFO] Best hyper-parameter configuration: ", bestHP)
    return bestHP

