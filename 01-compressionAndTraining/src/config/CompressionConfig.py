# CompressionConfig.py

class CompressionMode:
    BASELINE = "baseline"        # No compression
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    KD = "knowledge_distillation"
    Q_KD = "quantization_kd"
    Q_PRUNING = "quantization_pruning"
    Q_KD_PRUNING = "quantization_kd_pruning"

# Example usage
SELECTED_COMPRESSION = CompressionMode.Q_KD_PRUNING
