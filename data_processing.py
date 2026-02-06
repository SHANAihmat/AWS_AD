import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from anomaly_injection_baseline import inject_weather_anomalies
import torch
import rasterio


def read_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def csv_to_numpy(area):
    csv_path = "data"
    json_path = f"data/{area}_station_info.json"
    output_path = f"data/numpy_{area}"

    data_obs_path = os.path.join(csv_path, f"WEATHER-5K/{area}")
    data_era5_path = os.path.join(csv_path, f"ERA5/{area}")
    station_info = read_json(json_path)

    station_num = len(station_info["stations"])
    station_lon = [station["longitude"] for station in station_info["stations"]]
    station_lat = [station["latitude"] for station in station_info["stations"]]

    obs_data_sum = []
    era5_data_sum = []

    for i in range(station_num):
        station_name = station_info["stations"][i]["name"]
        obs_file = os.path.join(
            data_obs_path, f"{station_lat[i]}_{station_lon[i]}_weather-5k.csv"
        )
        print(obs_file)
        era5_file = os.path.join(
            data_era5_path, f"{station_lat[i]}_{station_lon[i]}_era5.csv"
        )

        if not os.path.exists(obs_file) or not os.path.exists(era5_file):
            print(f"File for station {station_name} does not exist. Skipping...")
            continue

        obs_data = pd.read_csv(obs_file)
        era5_data = pd.read_csv(era5_file)

        obs_data = obs_data.drop(obs_data.columns[[0, 3, 4]], axis=1)
        obs_data = obs_data.values
        obs_data = obs_data.reshape((1, obs_data.shape[0], obs_data.shape[1]))

        era5_data = era5_data.drop(era5_data.columns[[0, 3, 4]], axis=1)
        era5_data = era5_data.values
        era5_data = era5_data.reshape((1, era5_data.shape[0], era5_data.shape[1]))

        obs_data_sum.append(obs_data)
        era5_data_sum.append(era5_data)

    obs_data_sum = np.concatenate(obs_data_sum, axis=0)
    era5_data_sum = np.concatenate(era5_data_sum, axis=0)

    print(f"obs_data_sum shape: {obs_data_sum.shape}")
    print(f"era5_data_sum shape: {era5_data_sum.shape}")

    print(where := np.argwhere(np.isnan(obs_data_sum)))
    obs_data_sum[5, 2680:2690, 0:2] = era5_data_sum[5, 2680:2690, 0:2]

    if np.isnan(obs_data_sum).any():
        print("obs_data_sum contains NaN values.")

    np.save(os.path.join(output_path, "obs.npy"), obs_data_sum)
    np.save(os.path.join(output_path, "era5.npy"), era5_data_sum)
    print("Data saved successfully.")


def z_score_normalize(data, path, type="obs"):
    if type == "obs":
        mean_path = os.path.join(path, "z_score/obs_norm_mean.npy")
        std_path = os.path.join(path, "z_score/obs_norm_std.npy")
        mean = np.load(mean_path)
        std = np.load(std_path)
    elif type == "era5":
        mean_path = os.path.join(path, "z_score/era5_norm_mean.npy")
        std_path = os.path.join(path, "z_score/era5_norm_std.npy")
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        raise ValueError(f"Unsupported type: {type}")

    std_no_zero = np.where(std < 1e-20, 1.0, std)
    norm_data = (data - mean) / std_no_zero
    return norm_data


def save_z_score_params(data, out_path, type="obs"):
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))

    if type == "obs":
        np.save(os.path.join(out_path, "z_score/obs_norm_mean.npy"), mean)
        np.save(os.path.join(out_path, "z_score/obs_norm_std.npy"), std)
    elif type == "era5":
        np.save(os.path.join(out_path, "z_score/era5_norm_mean.npy"), mean)
        np.save(os.path.join(out_path, "z_score/era5_norm_std.npy"), std)
    else:
        raise ValueError(f"Unsupported type: {type}")


def z_score_inverse(data, device, type="obs"):
    path = "data/numpy_without_uv"

    if type == "obs":
        mean = np.load(os.path.join(path, "z_score/obs_norm_mean.npy"))
        std_no_zero = np.load(os.path.join(path, "z_score/obs_norm_std.npy"))
    elif type == "era5":
        mean = np.load(os.path.join(path, "z_score/era5_norm_mean.npy"))
        std_no_zero = np.load(os.path.join(path, "z_score/era5_norm_std.npy"))
    else:
        raise ValueError(f"Unsupported type: {type}")

    mean = torch.tensor(mean, dtype=torch.float32).to(device)
    std_no_zero = torch.tensor(std_no_zero, dtype=torch.float32).to(device)
    inv_data = data * std_no_zero + mean
    return inv_data


def split_data_by_time(data, start_time, end_time):
    time_format = "%Y-%m-%d %H:%M:%S"
    start_datetime = datetime.strptime(start_time, time_format)
    end_datetime = datetime.strptime(end_time, time_format)

    data_start_time = datetime(2020, 1, 1, 0, 0, 0)

    start_index = int((start_datetime - data_start_time) / timedelta(hours=1))
    end_index = int((end_datetime - data_start_time) / timedelta(hours=1)) + 1

    return data[:, start_index:end_index]


def data_split(data, out_path, type="obs"):
    train_start = "2020-01-01 00:00:00"
    train_end = "2022-12-31 23:00:00"
    val_start = "2020-01-01 00:00:00"
    val_end = "2022-12-31 23:00:00"
    test_start = "2023-01-01 00:00:00"
    test_end = "2023-12-31 23:00:00"

    if type == "obs":
        train_obs_data = split_data_by_time(data, train_start, train_end)
        val_obs_data = split_data_by_time(data, val_start, val_end)
        test_obs_data = split_data_by_time(data, test_start, test_end)

        test_obs_data = torch.from_numpy(test_obs_data).clone()
        test_obs_data, test_obs_ours_data, indicate_matrix = inject_weather_anomalies(
            test_obs_data,
            max_anomaly_ratio=0.01,
            local_point_window_size=72,
            seq_min_len=12,
            seq_max_len=36,
        )

        np.save(os.path.join(out_path, "obs/train_obs_data.npy"), train_obs_data)
        np.save(os.path.join(out_path, "obs/val_obs_data.npy"), val_obs_data)
        np.save(os.path.join(out_path, "obs/test_obs_data.npy"), test_obs_data)
        np.save(os.path.join(out_path, "obs/test_obs_ours_data.npy"), test_obs_ours_data)
        np.save(os.path.join(out_path, "obs/indicate_matrix.npy"), indicate_matrix)

    elif type == "era5":
        train_era5_data = split_data_by_time(data, train_start, train_end)
        val_era5_data = split_data_by_time(data, val_start, val_end)
        test_era5_data = split_data_by_time(data, test_start, test_end)

        np.save(os.path.join(out_path, "era5/train_era5_data.npy"), train_era5_data)
        np.save(os.path.join(out_path, "era5/val_era5_data.npy"), val_era5_data)
        np.save(os.path.join(out_path, "era5/test_era5_data.npy"), test_era5_data)

    else:
        raise ValueError(f"Unsupported type: {type}")


def load_geotiff(file_path, center_lat, center_lon, k):
    dataset = rasterio.open(file_path)
    transform = dataset.transform
    window = dataset.read(1)

    center_row, center_col = ~transform * (center_lon, center_lat)
    center_row = int(center_row)
    center_col = int(center_col)

    row_start = max(center_row - k, 0)
    row_end = min(center_row + k + 1, window.shape[0])
    col_start = max(center_col - k, 0)
    col_end = min(center_col + k + 1, window.shape[1])

    local_patch = window[row_start:row_end, col_start:col_end]
    return local_patch


if __name__ == "__main__":
    area = "France"

    csv_to_numpy(area)
    out_path = f"data/numpy_{area}"

    obs_data = np.load(f"data/numpy_{area}/obs.npy")
    era5_data = np.load(f"data/numpy_{area}/era5.npy")
    print("obs_data shape:", obs_data.shape, "era5_data shape:", era5_data.shape)

    data_split(era5_data, out_path, type="era5")
    data_split(obs_data, out_path, type="obs")
    save_z_score_params(obs_data, out_path, type="obs")
    save_z_score_params(era5_data, out_path, type="era5")
