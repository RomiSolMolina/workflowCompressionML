from pathlib import Path
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    D_SIGNAL: int = 1   #1: 1D, 2: 2D, 3: Sota dataset
    ROOT_PATH_1D: Path = Path("/media/ro/Data/datasets/gammaNeutron_diamondDetector/fft/")
    ROOT_PATH_2D: Path = Path("/home/ro/kaleido/datasets/unpolished")
    nLabels_1D: int = 2
    nLabels_2D: int = 2
    SAMPLES: int = 2031    # INPUT SIZE
    ROWS: int = 80
    COLS: int = 80
    DEPTH: int = 3



@dataclass
class TeacherConfig:
    TEACHER_OP: int = 0  # 0: train from scratch, 1: pre-trained
    OUTPUT_PATH: Path = Path("tunerTeacher")
    N_ITERATIONS: int = 2
    EPOCHS: int = 16
    BATCH_SIZE: int = 32
    EARLY_STOPPING_PATIENCE: int = 5
    MODEL_PATH: Path = Path("models/teacherModel.h5")


@dataclass
class StudentConfig:
    OUTPUT_PATH: Path = Path("tunerStudent")
    N_ITERATIONS: int = 2
    EPOCHS: int = 32
    BATCH_SIZE: int = 32
    EARLY_STOPPING_PATIENCE: int = 5
    CONSTANT_SPARSITY: float = 0.5
    MODEL_PATH: Path = Path("models/compressedModel.h5")


@dataclass
class HyperParamConfig:
    CONV_VAR: str = "conv_"
    FC_VAR: str = "fc"
    UPPER_CONV: int = 4
    UPPER_FC: int = 3

## Hyperparameters 
CONV_VAR = "conv_"
FC_VAR = "fc"
UPPER_CONV = 3
UPPER_FC = 4

# === Instancias de configuración global === #
dataset_config = DatasetConfig()
teacher_config = TeacherConfig()
student_config = StudentConfig()
hp_config = HyperParamConfig()
