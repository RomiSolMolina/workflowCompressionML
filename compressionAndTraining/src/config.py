
# Teacher optimization and training
# -----------------------------------------------------------------

# define the path to our output directory
OUTPUT_PATH = "outputhls4ml_particle"
# initialize the input shape and number of classes
INPUT_SHAPE = (30,)
NUM_CLASSES = 4

# define the total number of epochs to train, batch size, and the
# early stopping patience
EPOCHS = 16
BS = 32
EARLY_STOPPING_PATIENCE = 5

N_ITERATIONS_TEACHER = 2



# Student optimization and training
# -----------------------------------------------------------------

N_ITERATIONS_STUDENT = 2

EPOCHS_STUDENT = 32
BS_STUDENT = 32


INPUT_SHAPE_2d = (80, 80, 3)
NUM_CLASSES = 3

BS_2D = 32
EPOCHS_2D = 32

EPOCHS_STUDENT = 32
BS_STUDENT = 32

# Pruning
CONSTANT_SPARSITY = 0.5