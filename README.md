# Workflow to efficiently compress and deploy DNN classifiers for SoC/FPGA

Efficient end-to-end workflow for deploying DNNs on an SoC/FPGA by integrating hyperparameter tuning through Bayesian optimization with an ensemble of compression techniques (quantization, pruning, and knowledge distillation). 

The proposed workflow is based on three stages: DNN training and compression, integration with a hardware synthesis tool for ML, and hardware assessment.

![image](https://github.com/RomiSolMolina/workflowCompressionML/assets/13749513/56617ba0-e711-4241-b44b-67b1caa40c31)

## DNN training and compression

![image](https://github.com/RomiSolMolina/workflowCompressionML/assets/13749513/e234abec-ab56-4e16-8806-7f6859aaf384)

## Hardware assessment framework

![image](https://github.com/RomiSolMolina/workflowCompressionML/assets/13749513/833e0652-d0cc-4e96-b6b0-ce70107de034)


## Required libraries and tools

#### Libraries
Check the file `requirements.txt` inside the environment folder.

#### SoC/FPGA tools
- Vivado Design Suite - HLx Editions 2019.1, 2019.2, 2022.2

## What's in this repository?

Repository tree:

- The workflow comprises the following folders:
    - 00-environment
    - 01-compressionAndTraining
    - 02-hls4mlIntegration
    - 03-assessmentFramework
    - 04-integrationPYNQ*
       
* *The folder integrationPYNQ is for those willing to integrate the ML IP core into the PYNQ framework.

## Custom datasets

- Available under reasonable request.

## Current Branches

| Branch | Purpose |
|--------|---------|
| `main` | Clean refactor of original code |
| `backup_original` | Legacy structure linked to initial publication |
| `full_compression` | Combined pipeline for QAT + KD + pruning in a single loop |
| _coming soon..._ | Separate branches for isolated quantization, pruning, and KD |

---

## Citation

If this codebase is linked to a publication, cite it as:

```
@ARTICLE{10360204,
  author={Molina, Romina Soledad and Morales, Iván René and Crespo, Maria Liz and Costa, Veronica Gil and Carrato, Sergio and Ramponi, Giovanni},
  journal={IEEE Embedded Systems Letters}, 
  title={An End-to-End Workflow to Efficiently Compress and Deploy DNN Classifiers on SoC/FPGA}, 
  year={2024},
  volume={16},
  number={3},
  pages={255-258},
  doi={10.1109/LES.2023.3343030}}

```

---

## Acknowledgements

Built on top of:

- [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [QKeras](https://github.com/google/qkeras)
- [Keras Tuner](https://keras.io/keras_tuner/)

---

## Final remarks

**Have fun!!** 
And remember, this is a methodology to facilitate the training and compression process when targetting resource-constrained devices, it is not (yet ;) ) an automatic process.


