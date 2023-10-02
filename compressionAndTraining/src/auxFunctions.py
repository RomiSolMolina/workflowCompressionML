


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


def bestHPBO_computation(bestHP_BO, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC):
    """
    Grab the best hyperparameters after the optimization process
    
    """
    bestHP = []
    # Grab hyper-params
    for i in range (1,UPPER_CONV+1):
        bestHP.append(bestHP_BO.get(CONV_VAR + str(i)))
    for j in range (1, UPPER_FC+1):
        bestHP.append(bestHP_BO.get(FC_VAR + str(j)))
   
    print("Best hyper-parameter configuration: ", bestHP)

    return bestHP


