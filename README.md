# AWS-AD: [AWS-AD: A Scalable and Effective Framework for Anomaly Detection in Automatic Weather Station Data]

This repository contains the official anonymous PyTorch implementation for the paper "[AWS-AD: A Scalable and Effective Framework for Anomaly Detection in Automatic Weather Station Data".

## 1. Environment Setup

Please ensure you have Python 3.8+ installed. We highly recommend using Anaconda to manage the environment.

```bash
# Create a new conda environment
conda create -n awsad python=3.8
conda activate awsad

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

```

## 2. Dataset Preparation

Due to the strict confidentiality regulations of the meteorological administration, the Temperate Continental dataset is proprietary and cannot be publicly released. However, the two other datasets can be downloaded at https://arxiv.org/abs/2406.14399v1.
The ERA5 dataset can be downloaded at https://cds.climate.copernicus.eu/datasets.
The DEM dataset can be downloaded at https://download.gebco.net/.

## 3. Quick Start

We provide ready-to-use scripts to reproduce the anomaly detection results.

```bash
python main.py \
  --device cuda:0 \
  --data_path /home/sh/data/MyCode/AWS_AD/data/numpy_Xinjiang \
  --terrain_file /home/sh/data/MyCode/AWS_AD/data/DEM/Xinjiang.tif \
  --station_info_path /home/sh/data/MyCode/AWS_AD/data/Xinjiang_station_info.json \
  --time_window 72 \
  --time_embedding_dim 32 \
  --num_layers 2 \
  --gnn_hidden_dim 32 \
  --gnn_out_dim 360 \
  --mlp_hidden_dim 32 \
  --terrain_R 80 \
  --runs 1 \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 1e-2 \
  --patience 5 \
  --pca_compo 45 \
  --score_method pca
```