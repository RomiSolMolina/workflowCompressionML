from src.teacher_hpo_utils import run_teacher_hpo
from src.auxFunctions import bestHPBO_computation
from src.config import DatasetConfig, TeacherConfig, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC
from tensorflow.keras.models import load_model
from src.teacher_training_utils import train_teacher_model

# Import topology functions for training (non-HPO)
from src.topology.teacher.teacher_1d import modelTeacherTopology_1D
from src.topology.teacher.teacher_2d import topologyTeacher_2D
from src.topology.teacher.teacher_2d_sota import topologyTeacher_2D_SOTA


def optimize_teacher(xTrain=None, xTest=None, yTrain=None, yTest=None,
                     images_train=None, y_train=None,
                     images_test=None, y_test=None):
    """
    Perform Bayesian optimization on teacher model depending on dataset type.
    """
    if DatasetConfig.D_SIGNAL == 1:
        return run_teacher_hpo(xTrain, yTrain, xTest, yTest)
    elif DatasetConfig.D_SIGNAL == 2:
        return run_teacher_hpo(images_train, y_train, images_test, y_test)
    else:
        return run_teacher_hpo(images_train, y_train, images_test, y_test)


def train_teacher(bestHP, xTrain=None, xTest=None, yTrain=None, yTest=None,
                  images_train=None, images_test=None, y_train=None, y_test=None):
    """
    Train the teacher model using the unified training function.
    """
    
    bestHP = bestHPBO_computation(bestHP, CONV_VAR, FC_VAR, UPPER_CONV, UPPER_FC)
    lr = bestHP.get("learning_rate")
    
    if DatasetConfig.D_SIGNAL == 1:
        model_fn = modelTeacherTopology_1D
        teacherModel, history  = train_teacher_model(
            model_fn=model_fn,
            bestHP=bestHP,
            x_train=xTrain,
            y_train=yTrain,
            x_val=xTest,
            y_val=yTest,
            lr=lr,
            input_shape=(DatasetConfig.SAMPLES,),
            n_classes=DatasetConfig.nLabels_1D
        )

    elif DatasetConfig.D_SIGNAL == 2:
        model_fn = topologyTeacher_2D
        teacherModel, history  = train_teacher_model(
            model_fn=model_fn,
            bestHP=bestHP,
            x_train=images_train,
            y_train=y_train,
            x_val=images_test,
            y_val=y_test,
            lr=lr,
            input_shape=(DatasetConfig.COLS, DatasetConfig.ROWS, 3),
            n_classes=DatasetConfig.nLabels_2D
        )

    else:
        model_fn = topologyTeacher_2D_SOTA
        teacherModel, history  = train_teacher_model(
            model_fn=model_fn,
            bestHP=bestHP,
            x_train=images_train,
            y_train=y_train,
            x_val=images_test,
            y_val=y_test,
            lr=lr,
            input_shape=(32, 32, 3),
            n_classes=10
        )

    teacherModel.save(TeacherConfig.MODEL_PATH)
    return teacherModel, history


def load_teacher_model(path=TeacherConfig.MODEL_PATH):
    """
    Load pre-trained teacher model from file.
    """
    model = load_model(path)
    model.summary()
    return model
