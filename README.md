# Strategies for Pre-training Graph Neural Networks

This is a Pytorch implementation of the following paper: 

Weihua Hu*, Bowen Liu*, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec. Strategies for Pre-training Graph Neural Networks. ICLR 2020.
[arXiv](https://arxiv.org/abs/1905.12265) [OpenReview](https://openreview.net/forum?id=HJlWWJSFDH) 

If you make use of the code/experiment in your work, please cite our paper (Bibtex below).

```
@inproceedings{
hu2020pretraining,
title={Strategies for Pre-training Graph Neural Networks},
author={Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJlWWJSFDH},
}
```

## Installation
We used the following Python packages for core development. We tested on `Python 3.7`.
```
pytorch                   1.0.1
torch-cluster             1.2.4              
torch-geometric           1.0.3
torch-scatter             1.1.2 
torch-sparse              0.2.4
torch-spline-conv         1.0.6
rdkit                     2019.03.1.0
tqdm                      4.31.1
tensorboardx              1.6
```

## Dataset download
All the necessary data files can be downloaded from the following links.

For the chemistry dataset, download from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `chem/`.
For the biology dataset, download from [bio data](http://snap.stanford.edu/gnn-pretrain/data/bio_dataset.zip) (2GB), unzip it, and put it under `bio/`.

## Pre-training and fine-tuning
In each directory, we have three kinds of files used to train GNNs.

#### 1. Self-supervised pre-training
```
python pretrain_contextpred.py --output_model_file OUTPUT_MODEL_PATH
python pretrain_masking.py --output_model_file OUTPUT_MODEL_PATH
python pretrain_edgepred.py --output_model_file OUTPUT_MODEL_PATH
python pretrain_deepgraphinfomax.py --output_model_file OUTPUT_MODEL_PATH
```
This will save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

#### 2. Supervised pre-training
```
python pretrain_supervised.py --output_model_file OUTPUT_MODEL_PATH --input_model_file INPUT_MODEL_PATH
```
This will load the pre-trained model in `INPUT_MODEL_PATH`, further pre-train it using supervised pre-training, and then save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

#### 3. Fine-tuning
```
python finetune.py --model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET --filename OUTPUT_FILE_PATH
```
This will finetune pre-trained model specified in `INPUT_MODEL_PATH` using dataset `DOWNSTREAM_DATASET.` The result of fine-tuning will be saved to `OUTPUT_FILE_PATH.`

## Saved pre-trained models
We release pre-trained models in `model_gin/` and `model_architecture/` for both biology (`bio/`) and chemistry (`chem/`) applications. Feel free to take the models and use them in your applications!

## Reproducing results in the paper
Our results in the paper can be reproduced by running `sh finetune_tune.sh SEED DEVICE`, where `SEED` is a random seed ranging from 0 to 9, and `DEVICE` specifies the GPU ID to run the script. This script will finetune our saved pre-trained models on each downstream dataset.

