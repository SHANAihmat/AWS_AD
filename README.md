# AWS-AD: A Scalable and Effective Framework for Anomaly Detection in Automatic Weather Station Data

This repository contains the official implementation of **AWS-AD**, a scalable and effective framework for anomaly detection in Automatic Weather Station (AWS) time series, using reanalysis data (ERA5) as supervision and dynamic graph modeling.

- Paper: *AWS-AD: A Scalable and Effective Framework for Anomaly Detection in Automatic Weather Station Data*
- Code entry point: `main.py`
- Python: 3.8
- PyTorch: 2.5

## Repository Structure

Key scripts/modules:

- `main.py`: end-to-end runner (training -> anomaly detection -> evaluation)
- `data_processing.py`: CSV-to-NumPy conversion, z-score normalization, GeoTIFF terrain patch extraction
- `dataset.py`: `TimeWindowDataset` for windowed training/inference
- `graph_construction.py` (name may differ in your repo): dynamic graph building and exporting `edge_index/edge_attr`
- `embedding.py`: temporal + terrain embedding generator
- `gnn.py`: GNN model (with Kalman filter layer) and decoder
- `loss.py`: combined loss (reconstruction + temporal constraint)
- `train.py`: training loop, checkpointing, and dataset-wide reconstruction dumping
- `anomaly_dd.py`: anomaly scoring module
- `evaluate.py`: pointwise evaluation utilities

If any filename differs from the list above, please follow the actual filenames in this repository.

