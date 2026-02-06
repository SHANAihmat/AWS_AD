import os
import torch
import argparse
import numpy as np
import pandas as pd
import logging
import sys
from datetime import datetime
from loss import CombinedLoss

from data_processing import load_geotiff, read_json
from Kalman_residual_GNN import GNNWithMLP
from train import train_model
from evaluate import pointwise_evaluation
from PCA_based_scorer import anomaly_dd


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def main(runid, args):
    np.random.seed(runid)
    torch.manual_seed(runid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(runid)
        torch.cuda.manual_seed_all(runid)
    os.environ["PYTHONHASHSEED"] = str(runid)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Run {runid}: Using device: {device}")

    station_info = read_json(args.station_info_path)
    station_lon = [station["longitude"] for station in station_info["stations"]]
    station_lat = [station["latitude"] for station in station_info["stations"]]

    terrain_data = []
    for i in range(len(station_lon)):
        terrain = load_geotiff(
            args.terrain_file, station_lat[i], station_lon[i], args.terrain_R
        )
        terrain = torch.tensor(terrain, dtype=torch.float32)
        terrain_data.append(terrain)
    terrain_data = torch.stack(terrain_data).to(device)

    num_features = 5
    in_dim = num_features * 2
    gnn_with_mlp_model = GNNWithMLP(
        num_layers=args.num_layers,
        in_dim=in_dim,
        hidden_dim=args.gnn_hidden_dim,
        out_dim=args.gnn_out_dim,
        time_embedding_dim=args.time_embedding_dim,
        time_input_dim=args.time_window,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_out_dim=num_features,
    ).to(device)

    optimizer = torch.optim.Adam(
        gnn_with_mlp_model.parameters(), lr=args.learning_rate
    )
    loss_function = CombinedLoss(lambda_temporal=0.1)

    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        gnn_with_mlp_model.load_state_dict(
            checkpoint["gnn_with_mlp_model_state_dict"]
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
    else:
        print("No checkpoint found or specified, starting training from scratch.")

    train_recon, train_obs, val_recon, val_obs, test_recon, test_obs = train_model(
        gnn_with_mlp_model=gnn_with_mlp_model,
        optimizer=optimizer,
        loss_function=loss_function,
        terrain_data=terrain_data,
        time_window=args.time_window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        path=args.data_path,
        start_epoch=start_epoch,
        patience=args.patience,
        run_id=runid,
    )

    print("Starting anomaly detection...")
    anomaly_detector = anomaly_dd(
        train_obs=train_obs,
        val_obs=val_obs,
        test_obs=test_obs,
        train_forecast=train_recon,
        val_forecast=val_recon,
        test_forecast=test_recon,
        window_length=args.time_window,
        batch_size=args.batch_size,
    )

    indicator, prediction = anomaly_detector.scorer(args.pca_compo)

    np.save(
        f"data/indicator&prediction/anomaly_indicator_run{runid}.npy",
        indicator,
    )
    np.save(
        f"data/indicator&prediction/anomaly_prediction_run{runid}.npy",
        prediction,
    )

    print("indicator:", indicator)
    print("prediction:", prediction)

    labels_path = os.path.join(args.data_path, "obs", "indicate_matrix.npy")
    labels = np.load(labels_path)

    label = labels.transpose(1, 0, 2)
    label = label.any(axis=(1, 2))
    label = label[: len(prediction)]

    pointwise = pointwise_evaluation(label, prediction, indicator)
    print(f"Run {runid} - Pointwise Evaluation: {pointwise}")

    return pointwise


if __name__ == "__main__":
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.__stdout__),
        ],
    )

    sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

    sys.__stdout__.write(f"All output will be recorded to: {log_filepath}")

    parser = argparse.ArgumentParser(
        description="Graph Neural Network for Anomaly Detection in AWS"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device to use for training (e.g., "cuda:0" or "cpu")',
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/numpy_Xinjiang",
        help="Root directory for ERA5 and observation data",
    )
    parser.add_argument(
        "--terrain_file",
        type=str,
        default="data/DEM/Xinjiang.tif",
        help="Path to the terrain GeoTIFF file",
    )
    parser.add_argument(
        "--station_info_path",
        type=str,
        default="data/Xinjiang_station_info.json",
        help="Path to the station information JSON file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a model checkpoint to resume training. Leave empty to train from scratch.",
    )

    parser.add_argument(
        "--time_window",
        type=int,
        default=72,
        help="Size of the time window for input sequences",
    )
    parser.add_argument(
        "--time_embedding_dim",
        type=int,
        default=32,
        help="Dimension for the time embedding",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--gnn_hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension of GNN layers",
    )
    parser.add_argument(
        "--gnn_out_dim",
        type=int,
        default=72 * 5,
        help="Output dimension of the GNN module",
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension of the MLP decoder",
    )
    parser.add_argument(
        "--terrain_R",
        type=int,
        default=80,
        help="Radius for loading terrain data around stations",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run the experiment",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping",
    )

    parser.add_argument(
        "--pca_compo",
        type=int,
        default=45,
        help="Number of principal components for the anomaly scorer",
    )

    args = parser.parse_args()
    print("Running with the following arguments:")
    print(args)

    overall_results = []

    for i in range(args.runs):
        print(f"----------- Starting Run {i + 1}/{args.runs} -----------")
        pointwise = main(i, args)
        overall_results.append(pointwise)

    if not overall_results:
        print("No results were generated.")
    else:
        df = pd.DataFrame(overall_results)
        mean = dict(df.mean().round(4))
        std = dict(df.std().round(4))

        print("----------- Overall Detection Results -----------")

        if "roc" in mean and "prc" in mean:
            print("---- AUC result ----")
            table_data_auc = [
                ["Metric:", "ROC-AUC", "PRC-AUC"],
                ["mean:", mean.get("roc", "N/A"), mean.get("prc", "N/A")],
                ["std:", std.get("roc", "N/A"), std.get("prc", "N/A")],
            ]
            for row in table_data_auc:
                print("{: >20} {: >20} {: >20}".format(*row))

        print("---- Best F1 result ----")
        table_data_best_f1 = [
            ["Metric:", "Precision", "Recall", "F1"],
            [
                "mean:",
                mean.get("best_precision", "N/A"),
                mean.get("best_recall", "N/A"),
                mean.get("best_f1", "N/A"),
            ],
            [
                "std:",
                std.get("best_precision", "N/A"),
                std.get("best_recall", "N/A"),
                std.get("best_f1", "N/A"),
            ],
        ]
        for row in table_data_best_f1:
            print("{: >20} {: >20} {: >20} {: >20}".format(*row))

        print("---- Automatic threshold ----")
        table_data_auto_f1 = [
            ["Metric:", "Precision", "Recall", "F1"],
            [
                "mean:",
                mean.get("auto_precision", "N/A"),
                mean.get("auto_recall", "N/A"),
                mean.get("auto_f1", "N/A"),
            ],
            [
                "std:",
                std.get("auto_precision", "N/A"),
                std.get("auto_recall", "N/A"),
                std.get("auto_f1", "N/A"),
            ],
        ]
        for row in table_data_auto_f1:
            print("{: >20} {: >20} {: >20} {: >20}".format(*row))
