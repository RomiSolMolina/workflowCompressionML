from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from src.utils import normalizationPix, confusionMatrixPlot
from src.dataset_loader import loadDataset_1D, loadDataset_2D
from src.config.config import DatasetConfig, TeacherConfig, dataset_config
from src.teacher_utils import optimize_teacher, train_teacher, load_teacher_model
from src.student_utils import optimize_student, train_student
from src.evaluation import plot_training_curves, plot_confusion_matrix

from src.config.CompressionConfig import SELECTED_COMPRESSION, CompressionMode


def startDNNTrainingAndCompression(
    use_kd=None,
    use_quant=None,
    use_prune=None,
    selected_compression=None
):
    """
    Entrena el modelo DNN con compresión opcional (KD, Q, P), usando configuración automática o explícita.
    """

    # === Resolución de estrategia de compresión === #
    if selected_compression is None:
        selected_compression = SELECTED_COMPRESSION

    if use_kd is None or use_quant is None or use_prune is None:
        use_quant = selected_compression in [
            CompressionMode.QUANTIZATION,
            CompressionMode.Q_KD,
            CompressionMode.Q_PRUNING,
            CompressionMode.Q_KD_PRUNING
        ]

        use_prune = selected_compression in [
            CompressionMode.PRUNING,
            CompressionMode.Q_PRUNING,
            CompressionMode.Q_KD_PRUNING
        ]

        use_kd = selected_compression in [
            CompressionMode.KD,
            CompressionMode.Q_KD,
            CompressionMode.Q_KD_PRUNING
        ]
        # use_lowrank = SELECTED_COMPRESSION == CompressionMode.LOWRANK  # TODO: implement

    print(f"\n[INFO] Selected Compression Strategy: {selected_compression}")
    print(f"[INFO] use_kd={use_kd}, use_quant={use_quant}, use_prune={use_prune}")

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
            "y_train": y_train,
            "y_test": y_test
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
        bestHP_teacher = optimize_teacher(
            xTrain=dataset_info.get("xTrain"),
            xTest=dataset_info.get("xTest"),
            yTrain=dataset_info.get("yTrain"),
            yTest=dataset_info.get("yTest")
        )
        teacherModel, history = train_teacher(bestHP_teacher, **dataset_info)
    else:
        teacherModel = load_teacher_model()
        history = None

    # === Teacher Evaluation === #
    if dataset_type == 1:
        plot_confusion_matrix(teacherModel, xTest_df_Final, yTest_Final)
    else:
        plot_confusion_matrix(teacherModel, dataset_info["images_test"], dataset_info["y_test"])

    if history:
        plot_training_curves(history)

    # === Student Optimization & Training === #
    bestHP_student = optimize_student(
        teacher_model=teacherModel,
        use_kd=use_kd,
        use_quant=use_quant,
        use_prune=use_prune,
        **dataset_info
    )

    studentModel, history = train_student(
        bestHP_student,
        teacherModel,
        use_kd=use_kd,
        use_quant=use_quant,
        use_prune=use_prune,
        **dataset_info
    )

    # === Student Evaluation === #
    if dataset_type == 1:
        plot_confusion_matrix(studentModel, xTest_df_Final, yTest_Final)
    else:
        plot_confusion_matrix(studentModel, dataset_info["images_test"], dataset_info["y_test"])
