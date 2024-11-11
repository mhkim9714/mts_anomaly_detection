# Multivariate Time Series (MTS) Anomaly Detection

## Setup
1. Create a new Conda environment:
   ```
   conda create -n {env_name} python=3.8
   ```
2. Install Pytorch:
   ```
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
   ```
3. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation
* We use synthetic data provided by the [MSCRED](https://github.com/wxdang/MSCRED) repository as a sample dataset. The sample dataset contains 10,000 timestamps and 30 variables each for training and testing.
* Preprocessing was done using the code provided in the [TranAD](https://github.com/imperial-qore/TranAD) repository.

## Available Models
* Reconstruction-based anomaly detection
    * EncDec-AD: An implementation of the ["LSTM-based Encoder-Decoder model for multi-sensor anomaly detection"](https://arxiv.org/abs/1607.00148) from the *ICML 2016* paper.
        * To train and evaluate using EncDec-AD, use:
          ```
          python main.py --model "EncDec-AD"
          ```
        * To change model configuration details, edit the settings in `config.py`.
* Forecasting-based anomaly detection
    * Transformer: Based on [TranAD](https://github.com/imperial-qore/TranAD), this is a modified version of the ["Attention is all you need"](https://arxiv.org/abs/1706.03762) (*NIPS 2017*) adapted for multivariate time series anomaly detection.
        * To train and evaluate using Transformer, use:
          ```
          python main.py --model "Transformer"
          ```
        * To change model configuration details, edit the settings in `config.py`.

## Evaluation
To evaluate with pretrained weights, add the following argument:
```
--load_path "./results/{directory_containing_pretrained_weight_pth}"
```
Logs will be saved in the specified directory.

## Results
Logs and evaluation results are stored in:
```
MTS/results/{%m-%d-%HH%MM%Ss}_{model_name}_{comment}/
```
Generated files include:
* `best_model.pt`: Best model weights saved during training.
* `.txt` file: Contains argument settings, data configuration, model configuration, training logs, and anomaly detection results.
