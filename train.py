import torch
from torch.utils.data import Dataset, DataLoader
from dataset import TimeWindowDataset
import datetime
import numpy as np
from data_processing import z_score_normalize, z_score_inverse
import pickle
import os


def train_model(
    gnn_with_mlp_model,
    optimizer,
    loss_function,
    terrain_data,
    time_window,
    batch_size,
    epochs,
    device,
    path,
    start_epoch=0,
    patience=10,
    run_id=0,
):
    time_window_for_test = time_window
    terrain_data_for_test = terrain_data

    train_ori_obs_data = np.load(os.path.join(path, "obs/train_obs_data.npy"))
    train_obs_data = z_score_normalize(train_ori_obs_data, path, type="obs")
    train_era5_data = np.load(os.path.join(path, "era5/train_era5_data.npy"))
    train_era5_data = z_score_normalize(train_era5_data, path, type="era5")

    with open(os.path.join(path, "graph_structure/train_edge_attr.pkl"), "rb") as f:
        train_edge_attr = pickle.load(f)
    with open(os.path.join(path, "graph_structure/train_edge_index.pkl"), "rb") as f:
        train_edge_index = pickle.load(f)

    train_dataset = TimeWindowDataset(
        obs_data=train_obs_data,
        era5_data=train_era5_data,
        ori_obs_data=train_ori_obs_data,
        edge_index_list=train_edge_index,
        edge_attr_list=train_edge_attr,
        terrain_data=terrain_data,
        time_window=time_window,
    )
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_ori_obs_data = np.load(os.path.join(path, "obs/val_obs_data.npy"))
    val_obs_data = z_score_normalize(val_ori_obs_data, path, type="obs")
    val_era5_data = np.load(os.path.join(path, "era5/val_era5_data.npy"))
    val_era5_data = z_score_normalize(val_era5_data, path, type="era5")

    with open(os.path.join(path, "graph_structure/val_edge_attr.pkl"), "rb") as f:
        val_edge_attr = pickle.load(f)
    with open(os.path.join(path, "graph_structure/val_edge_index.pkl"), "rb") as f:
        val_edge_index = pickle.load(f)

    val_dataset = TimeWindowDataset(
        obs_data=val_obs_data,
        era5_data=val_era5_data,
        ori_obs_data=val_ori_obs_data,
        edge_index_list=val_edge_index,
        edge_attr_list=val_edge_attr,
        terrain_data=terrain_data,
        time_window=time_window,
    )
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        gnn_with_mlp_model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            aws_data, era5_supervise, terrain_data, labels, time_window, edge_index, edge_attr = batch
            aws_data = aws_data.to(device).float()
            era5_supervise = era5_supervise.to(device).float()
            terrain_data = terrain_data.to(device).float()
            labels = labels.to(device).float()
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            time_window_gpu = time_window.to(device)

            optimizer.zero_grad()

            output = gnn_with_mlp_model(
                aws_data,
                era5_supervise,
                terrain_data,
                time_window_gpu,
                edge_index,
                edge_attr,
            ).to(device)

            loss, _, _ = loss_function(output, aws_data, era5_supervise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        gnn_with_mlp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                aws_data, era5_supervise, terrain_data, labels, time_window, edge_index, edge_attr = batch
                aws_data = aws_data.to(device).float()
                era5_supervise = era5_supervise.to(device).float()
                terrain_data = terrain_data.to(device).float()
                labels = labels.to(device).float()
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                time_window_gpu = time_window.to(device)

                output = gnn_with_mlp_model(
                    aws_data,
                    era5_supervise,
                    terrain_data,
                    time_window_gpu,
                    edge_index,
                    edge_attr,
                )

                loss, _, _ = loss_function(output, aws_data, era5_supervise)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = {
                "model": gnn_with_mlp_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs...")

        now = datetime.datetime.now()
        folder_name = now.strftime("%Y%m%d%H")
        checkpoints_path = os.path.join(
            "/home/sh/data/MyCode/AWS_AD/checkpoints", folder_name
        )
        os.makedirs(checkpoints_path, exist_ok=True)
        save_path = os.path.join(
            checkpoints_path, f"checkpoint_epoch_{epoch + 1}_run_id{run_id}.pth"
        )

        torch.save(
            {
                "gnn_with_mlp_model_state_dict": gnn_with_mlp_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": epoch_loss,
                "val_loss": avg_val_loss,
            },
            save_path,
        )

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch + 1}"
            )
            gnn_with_mlp_model.load_state_dict(best_model_state["model"])
            optimizer.load_state_dict(best_model_state["optimizer"])
            break

    if best_model_state is not None:
        gnn_with_mlp_model.load_state_dict(best_model_state["model"])
        optimizer.load_state_dict(best_model_state["optimizer"])

    final_model = gnn_with_mlp_model.to(device)

    train_recon, train_recon_obs = reconstruct_entire_dataset(
        final_model, train_dataset, batch_size, device, path, "train"
    )
    val_recon, val_recon_obs = reconstruct_entire_dataset(
        final_model, val_dataset, batch_size, device, path, "val"
    )

    test_ori_obs_data = np.load(os.path.join(path, "obs/test_obs_data.npy"))
    test_obs_data = z_score_normalize(test_ori_obs_data, path, type="obs")
    test_era5_data = np.load(os.path.join(path, "era5/test_era5_data.npy"))
    test_era5_data = z_score_normalize(test_era5_data, path, type="era5")

    with open(os.path.join(path, "graph_structure/test_edge_attr.pkl"), "rb") as f:
        test_edge_attr = pickle.load(f)
    with open(os.path.join(path, "graph_structure/test_edge_index.pkl"), "rb") as f:
        test_edge_index = pickle.load(f)

    test_dataset = TimeWindowDataset(
        obs_data=test_obs_data,
        era5_data=test_era5_data,
        ori_obs_data=test_ori_obs_data,
        edge_index_list=test_edge_index,
        edge_attr_list=test_edge_attr,
        terrain_data=terrain_data_for_test,
        time_window=time_window_for_test,
    )
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    test_recon, test_recon_obs = reconstruct_entire_dataset(
        final_model, test_dataset, batch_size, device, path, "test"
    )

    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            aws_data, era5_supervise, terrain_data, labels, time_window, edge_index, edge_attr = batch
            aws_data = aws_data.to(device).float()
            era5_supervise = era5_supervise.to(device).float()
            terrain_data = terrain_data.to(device).float()
            labels = labels.to(device).float()
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            time_window = time_window.to(device)

            output = gnn_with_mlp_model(
                aws_data, era5_supervise, terrain_data, time_window, edge_index, edge_attr
            )

            loss, _, _ = loss_function(output, aws_data, era5_supervise)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.6f}")

    return (
        train_recon,
        train_recon_obs,
        val_recon,
        val_recon_obs,
        test_recon,
        test_recon_obs,
    )


def reconstruct_entire_dataset(model, dataset, batch_size, device, path, dataset_name):
    model.eval()
    loader = DataLoader(dataset, batch_size, shuffle=False)

    all_reconstructions = []
    all_originals = []

    with torch.no_grad():
        for batch in loader:
            aws_data, era5_supervise, terrain_data, labels, time_window, edge_index, edge_attr = batch

            aws_data = aws_data.to(device).float()
            era5_supervise = era5_supervise.to(device).float()
            terrain_data = terrain_data.to(device).float()
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            time_window = time_window.to(device)

            output = model(
                aws_data, era5_supervise, terrain_data, time_window, edge_index, edge_attr
            )
            output = z_score_inverse(output, device, type="obs")

            all_reconstructions.append(output.cpu().numpy())
            all_originals.append(aws_data.cpu().numpy())

    reconstructed_data_0 = np.concatenate(all_reconstructions, axis=0)
    original_data_0 = np.concatenate(all_originals, axis=0)

    reconstructed_data = (
        reconstructed_data_0.transpose(1, 0, 2, 3).reshape(
            reconstructed_data_0.shape[1], -1, reconstructed_data_0.shape[3]
        )
    )
    original_data = (
        original_data_0.transpose(1, 0, 2, 3).reshape(
            original_data_0.shape[1], -1, original_data_0.shape[3]
        )
    )

    recon_path = os.path.join(path, f"reconstructed/{dataset_name}")
    os.makedirs(recon_path, exist_ok=True)

    np.save(
        os.path.join(recon_path, f"reconstructed_{dataset_name}_data.npy"),
        reconstructed_data,
    )
    np.save(
        os.path.join(recon_path, f"original_{dataset_name}_data.npy"),
        original_data,
    )

    print(f"Saved reconstructed {dataset_name} data at: {recon_path}")

    return reconstructed_data, original_data
