# Workflow to efficiently compress and deploy DNN classifiers for SoC/FPGA

Efficient end-to-end workflow for deploying DNNs on an SoC/FPGA by integrating hyperparameter tuning through Bayesian optimization with an ensemble of compression techniques (quantization, pruning, and knowledge distillation).

![image](https://github.com/RomiSolMolina/workflowCompressionML/assets/13749513/026ecb86-a3ed-4f0a-a257-c505a097c374)

## DNN training and compression

![image](https://github.com/RomiSolMolina/workflowCompressionML/assets/13749513/dc4b9b0d-4454-40df-aeaf-7502d29ce8df)

## Hardware assessment framework

![image](https://github.com/RomiSolMolina/workflowCompressionML/assets/13749513/70b1a633-cb2a-402a-b707-dd2bdea9dd9d)


## Required libraries
- Python 3.9.13
- Tensorflow 2.4.1
- Qkeras 0.9.0
- hls4ml 0.6.0
- Pandas 1.5.2
- Seaborn 0.11.2
- Keras-Tuner 1.1.2
- Scikit-learn 1.1.1

## SoC/FPGA
- Vivado 2019.1 or Vivado 2019.2

