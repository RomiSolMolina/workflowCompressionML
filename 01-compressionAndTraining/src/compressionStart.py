from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from src.utils import normalizationPix, confusionMatrixPlot
from src.dataset_loader import loadDataset_1D, loadDataset_2D
from src.config import DatasetConfig, TeacherConfig, dataset_config
from src.teacher_utils import optimize_teacher, train_teacher, load_teacher_model
from src.student_utils import optimize_student, train_student
from src.evaluation import plot_training_curves, plot_confusion_matrix

from src.student_utils import optimize_student, train_student


def startDNNTrainingAndCompression():
    # === Dataset Loading === #
    dataset_type = DatasetConfig.D_SIGNAL
    dataset_info = {}

    if dataset_type == 1:
        xTrain, xTest, xTest_df_Final, yTrain, yTest, yTest_Final = loadDataset_1D(
            dataset_config.ROOT_PATH_1D,
            dataset_config.nLabels_1D,
            dataset_config.SAMPLES
        )
        dataset_info = {
            "xTrain": xTrain,
            "xTest": xTest,
            "yTrain": yTrain,
            "yTest": yTest,
        }

    elif dataset_type == 2:
        images_train, images_validation, images_test, y_train, y_test = loadDataset_2D(
            dataset_config.ROOT_PATH_2D,
            dataset_config.nLabels_2D,
            dataset_config.ROWS,
            dataset_config.COLS
        )
        dataset_info = {
            "images_train": images_train,
            "images_validation": images_validation,
            "images_test": images_test,
            "y_train": y_train, "y_test": y_test
        }

    elif dataset_type == 3:
        (images_train, y_train), (images_test, y_test) = cifar10.load_data()
        images_train, images_test = normalizationPix(images_train, images_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        dataset_info = {
            "images_train": images_train,
            "images_test": images_test,
            "y_train": y_train,
            "y_test": y_test
        }

    # === Teacher Training === #
    
    if TeacherConfig.TEACHER_OP == 0:
            bestHP_teacher = optimize_teacher(**dataset_info)
            teacherModel, history = train_teacher(bestHP_teacher, **dataset_info)
    else:
        teacherModel = load_teacher_model()

    # === Teacher Evaluation === #
    if DatasetConfig.D_SIGNAL == 1:
        plot_confusion_matrix(teacherModel, xTest_df_Final, yTest_Final)
    else:
        plot_confusion_matrix(teacherModel, images_test, y_test)

    plot_training_curves(history)

  
    # === Student Optimization & Training === #
    bestHP_student = optimize_student(**dataset_info, teacher_model=teacherModel)
    studentModel, history = train_student(bestHP_student, teacherModel, **dataset_info)

    # === Student Evaluation === #
    if dataset_type == 1:
         plot_confusion_matrix(studentModel, xTest_df_Final, yTest_Final)
    else:
        plot_confusion_matrix(studentModel, images_test, y_test)
