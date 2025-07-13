# src/config/CompressionConfig.py

from enum import Enum

class CompressionMode(Enum):
    BASELINE = "baseline"
    QUANTIZATION = "quant"
    PRUNING = "prune"
    KD = "kd"
    Q_KD = "quant+kd"
    Q_PRUNING = "quant+prune"
    Q_LOW_RANK = "q_lowrank"
    Q_KD_PRUNING = "quant+kd+prune"

# SELECT THE STRATEGY
SELECTED_COMPRESSION = CompressionMode.Q_KD_PRUNING
# SELECTED_COMPRESSION = CompressionMode.BASELINE
