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
- Python 3.9.13
- Tensorflow 2.4.1
- Qkeras 0.9.0
- hls4ml 0.6.0
- Pandas 1.5.2
- Seaborn 0.11.2
- Keras-Tuner 1.1.2
- Scikit-learn 1.1.1

#### SoC/FPGA tools
- Vivado Design Suite - HLx Editions 2019.1 or 2019.2

## What's in this repository?

Repository tree:

- The workflow comprises these main folders:
    - compressionAndTraining,
    - assessmentFramework, 
    - hls4mlIntegration.
      
- The folder integrationPYNQ is for those willing to integrate the ML IP core into the PYNQ framework.

## Custom datasets

- Available under reasonable request.

## Final remarks

**Have fun!!** 
And remember, this is a methodology to facilitate the training and compression process when targetting resource-constrained devices, it is not (yet ;) ) an automatic process.


