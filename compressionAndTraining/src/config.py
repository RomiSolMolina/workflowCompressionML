## --------------------- DEFINITIONS AND PARAMETERS ----------------

## TEACHER OPTIMIZATION
## Define if the teacher will be trained from scratch or a pre-trained model will be used
# 0: train teacher from scratch, 1: pre-trained model
TEACHER_OP = 1

## TYPE OF INPUT
## Define if the input will be 1D or 2D signal for custon datasets. The value 3 is reserved for SOTA datasets.
# Type of input -->  1: 1D signal, 2: 2D signal, 3: state-of-the art dataset
D_SIGNAL = 2


## CUSTOM DATASET PATHS
ROOT_PATH_2D = "/home/ro/kaleido/datasets/unpolished"
nLabels_2D = 2
SAMPLES = 30

ROOT_PATH_1D = r'dataset/psd/csv'
nLabels_1D = 4
ROWS = 80
COLS = 80
DEPTH = 3

# Teacher optimization and training
# -----------------------------------------------------------------

# define the path to the output directory (the algorithm will save intermediate results)
OUTPUT_PATH_TEACHER = "tunerTeacher"
# initialize the input shape and number of classes
# INPUT_SHAPE = (30,)
# NUM_CLASSES = 4

# Number of iterations for BO
N_ITERATIONS_TEACHER = 200

# define the total number of epochs to train, batch size, and the
# early stopping patience
EPOCHS_TEACHER = 16
BATCH_TEACHER = 32
EARLY_STOPPING_PATIENCE_TEACHER = 5

N_ITERATIONS_TEACHER = 2

PATH_MODEL_TEACHER = "models/teacherModel.h5"

# Student optimization and training
# -----------------------------------------------------------------
# define the path to the output directory (the algorithm will save intermediate results)
OUTPUT_PATH_STUDENT = "tunerStudent"

# Number of iterations for BO
N_ITERATIONS_STUDENT = 2

EPOCHS_STUDENT = 32
BATCH_STUDENT = 32
EARLY_STOPPING_PATIENCE_STUDENT = 5

# Pruning
CONSTANT_SPARSITY = 0.5


PATH_MODEL_STUDENT = "models/compressedModel.h5"



## Hyperparameters 

CONV_VAR = 'conv_'
FC_VAR = 'fc'

## 
# Quantity of layers based on the topology defined by the user
UPPER_CONV = 4
UPPER_FC = 3    

