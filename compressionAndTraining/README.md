# DNN training and compression

This project provides a modular framework for training and compressing deep neural networks using quantization, pruning, and knowledge distillation. It supports both 1D and 2D signal data and allows flexible experimentation with different compression strategies.

- **Quantization-Aware Training (QAT)** with [QKeras](https://github.com/google/qkeras)
- **Pruning** via [TensorFlow Model Optimization Toolkit (TF-MOT)](https://www.tensorflow.org/model_optimization)
- **Knowledge Distillation** for student-teacher optimization
- A clean pipeline for combining all or some of the above techniques


## Features

| Feature              | Description |
|----------------------|-------------|
| **Quantization**     | QKeras layers (QConv, QDense, QActivation) with `quantized_bits` |
| **Pruning**          | TF-MOT pruning via `prune_low_magnitude()` and `ConstantSparsity` |
| **KD Distillation**  | Student mimics teacher's logits with temperature scaling and α-weighted loss |
| **Bayesian Optimization** | Keras Tuner integration for HPO (Bayesian) on student architectures |
| **1D/2D Support**    | Works for both tabular and image data (`D_SIGNAL` switchable) |


## Module Descriptions

- **`config.py`**  
  Configuration file containing global variables for training and compression. Modify this to customize dataset settings, architecture details, or training parameters.

- **`compressionMain.ipynb`**  
  Main Jupyter Notebook interface for performing training, hyperparameter tuning, and model compression.

- **`compressionStart.py`**  
  Contains the logic to select the appropriate training flow depending on the signal type (1D or 2D).

- **`topology/` folder**  
  Houses modular templates used for building teacher and student networks, and for defining student search spaces:
  
  - For **1D signals**: includes topology files for both teacher and student models, and their hyperparameter tuning variants.
  - For **2D signals**: similar structure for image-based input.
  
  These files are fully customizable and can be extended for other tasks or datasets.


### Datasets

- Custom datasets will be provided under reasonable request.
- The methodology can be implemented using SOTA datasets, such as MNIST and CIFAR.

## How to Run

### 1. Install dependencies

If needed, install dependencies
```bash
pip install -r ../requeriments/requirements.txt
```

### 2. Launch the compression and training pipeline
 Open the Jupyter notebook file `compressionMain.ipynb` and follow the instructions. 

---


### Final remarks

**Have fun!!** 

And remember, this is a methodology to facilitate the training and compression process when targetting resource-constrained devices, it is not (yet ;) ) an automatic process.
