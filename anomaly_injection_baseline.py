import torch
import numpy as np
import random
from collections import defaultdict

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")

FEATURE_MAP = {
    "temp": 0,
    "dew_point": 1,
    "u_wind": 2,
    "v_wind": 3,
    "pressure": 4,
}

TEMP_IDX = FEATURE_MAP["temp"]
DEW_POINT_IDX = FEATURE_MAP["dew_point"]
U_WIND_IDX = FEATURE_MAP["u_wind"]
V_WIND_IDX = FEATURE_MAP["v_wind"]
PRESSURE_IDX = FEATURE_MAP["pressure"]


def inject_global_point_anomaly(X_station, t_idx, features_to_inject):
    for feat_idx in features_to_inject:
        feature_data = X_station[:, feat_idx]
        mean_val = torch.mean(feature_data)
        std_val = torch.std(feature_data)

        magnitude = random.uniform(3.0, 5.0)
        direction = random.choice([-1, 1])
        anomaly_value = mean_val + direction * magnitude * std_val

        X_station[t_idx, feat_idx] = anomaly_value
    return X_station


def inject_local_point_anomaly(X_station, t_idx, feat_idx, window_size=24):
    T = X_station.shape[0]
    start = max(0, t_idx - window_size // 2)
    end = min(T, t_idx + window_size // 2 + 1)

    window_data = X_station[start:end, feat_idx]

    if t_idx > start:
        valid_window_data = torch.cat(
            [window_data[: t_idx - start], window_data[t_idx - start + 1 :]]
        )
    else:
        valid_window_data = window_data[1:]

    if valid_window_data.numel() > 0:
        min_val, max_val = torch.min(valid_window_data), torch.max(valid_window_data)
        data_range = max(abs(max_val - min_val).item(), 1e-6)

        multiplier = random.uniform(2.5, 4.0)

        if random.random() < 0.5:
            anomaly_value = max_val + data_range * multiplier
        else:
            anomaly_value = min_val - data_range * multiplier

        X_station[t_idx, feat_idx] = anomaly_value
    return X_station


def inject_wind_anomaly(X_station, start_idx, subseq_len):
    end_idx = start_idx + subseq_len

    original_u_wind = X_station[start_idx:end_idx, U_WIND_IDX].clone()
    original_v_wind = X_station[start_idx:end_idx, V_WIND_IDX].clone()

    X_station[start_idx:end_idx, U_WIND_IDX] = -original_u_wind
    X_station[start_idx:end_idx, V_WIND_IDX] = -original_v_wind
    return X_station


def inject_sensor_drift(X_station, start_idx, subseq_len, feat_idx):
    subseq = X_station[start_idx : start_idx + subseq_len, feat_idx]
    std_dev = torch.std(X_station[:, feat_idx])

    slope_magnitude = std_dev * random.uniform(0.15, 0.30)
    slope = slope_magnitude * random.choice([-1, 1])

    trend = torch.linspace(
        0, slope * (subseq_len - 1), subseq_len, device=device
    )
    X_station[start_idx : start_idx + subseq_len, feat_idx] = subseq + trend
    return X_station


def inject_weather_anomalies(
    X,
    max_anomaly_ratio=0.05,
    local_point_window_size=144,
    seq_min_len=144,
    seq_max_len=240,
    seed=None,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(device)

    X = X.clone().float().to(device)

    N, T, F = X.shape
    if F != len(FEATURE_MAP):
        raise ValueError(
            f"Expected {len(FEATURE_MAP)} features, but got {F}"
        )

    total_points = N * T * F
    max_anomaly_points = int(total_points * max_anomaly_ratio)

    X_out = X.clone()
    mask = torch.zeros_like(X, dtype=torch.int8)

    injection_plan = []
    wind_anomaly_sites = set()
    sensor_drift_sites = set()

    for n in range(N):
        ratio_single_point = 0.001

        for _ in range(
            random.randint(int(T * ratio_single_point), int(T * ratio_single_point) + 1)
        ):
            t_idx = random.randint(0, T - 1)
            feats = random.sample(
                [TEMP_IDX, DEW_POINT_IDX, PRESSURE_IDX],
                k=random.randint(1, 3),
            )
            injection_plan.append(
                ("global_point", n, {"t_idx": t_idx, "features_to_inject": feats})
            )

        for _ in range(
            random.randint(int(T * ratio_single_point), int(T * ratio_single_point) + 1)
        ):
            t_idx = random.randint(0, T - 1)
            feat_idx = random.randint(0, F - 1)
            injection_plan.append(
                (
                    "local_point",
                    n,
                    {
                        "t_idx": t_idx,
                        "feat_idx": feat_idx,
                        "window_size": local_point_window_size,
                    },
                )
            )

        if len(wind_anomaly_sites) < 2:
            for _ in range(random.randint(1, 3)):
                subseq_len = random.randint(seq_min_len, min(T, seq_max_len))
                start_idx = random.randint(0, T - subseq_len)
                injection_plan.append(
                    (
                        "wind_anomaly",
                        n,
                        {"start_idx": start_idx, "subseq_len": subseq_len},
                    )
                )
            wind_anomaly_sites.add(n)

        if len(sensor_drift_sites) < 2:
            for _ in range(random.randint(1, 3)):
                subseq_len = random.randint(seq_min_len, min(T, seq_max_len))
                start_idx = random.randint(0, T - subseq_len)
                feat_idx = random.choice([TEMP_IDX, PRESSURE_IDX])
                injection_plan.append(
                    (
                        "sensor_drift",
                        n,
                        {
                            "start_idx": start_idx,
                            "subseq_len": subseq_len,
                            "feat_idx": feat_idx,
                        },
                    )
                )
            sensor_drift_sites.add(n)

    random.shuffle(injection_plan)

    selected_tasks = []
    selected_coords_set = set()

    for anomaly_type, n, params in injection_plan:
        task_coords = []

        if anomaly_type == "global_point":
            t = params["t_idx"]
            for f in params["features_to_inject"]:
                task_coords.append((n, t, f))

        elif anomaly_type == "local_point":
            t, f = params["t_idx"], params["feat_idx"]
            task_coords.append((n, t, f))

        elif anomaly_type == "wind_anomaly":
            start, length = params["start_idx"], params["subseq_len"]
            for t in range(start, start + length):
                for f in [U_WIND_IDX, V_WIND_IDX]:
                    task_coords.append((n, t, f))

        elif anomaly_type == "sensor_drift":
            start, length, f = (
                params["start_idx"],
                params["subseq_len"],
                params["feat_idx"],
            )
            for t in range(start, start + length):
                task_coords.append((n, t, f))

        new_coords = [
            coord for coord in task_coords if coord not in selected_coords_set
        ]

        if len(selected_coords_set) + len(new_coords) <= max_anomaly_points:
            selected_tasks.append((anomaly_type, n, params))
            selected_coords_set.update(task_coords)

    for anomaly_type, n, params in selected_tasks:
        station_data = X_out[n]
        if anomaly_type == "global_point":
            inject_global_point_anomaly(station_data, **params)
        elif anomaly_type == "local_point":
            inject_local_point_anomaly(station_data, **params)
        elif anomaly_type == "wind_anomaly":
            inject_wind_anomaly(station_data, **params)
        elif anomaly_type == "sensor_drift":
            inject_sensor_drift(station_data, **params)

    for n, t, f in selected_coords_set:
        mask[n, t, f] = 1

    anomaly_per_timestep = mask.sum(axis=(0, 2))
    unique_anomaly_timestamps = np.where(anomaly_per_timestep > 0)[0]
    num_unique_anomaly_timestamps = len(unique_anomaly_timestamps)
    unique_timestamp_ratio = num_unique_anomaly_timestamps / T

    print(f"Total timesteps: {T}")
    print(f"Number of unique anomaly timestamps: {num_unique_anomaly_timestamps}")
    print(
        f"Unique anomaly timestamp ratio: {unique_timestamp_ratio:.4f} "
        f"({unique_timestamp_ratio * 100:.2f}%)"
    )

    anomaly_timestamps = defaultdict(set)

    for anomaly_type, n, params in selected_tasks:
        if anomaly_type in ["global_point", "local_point"]:
            t = params["t_idx"]
            anomaly_timestamps[anomaly_type].add(t)
        elif anomaly_type in ["wind_anomaly", "sensor_drift"]:
            start = params["start_idx"]
            length = params["subseq_len"]
            for t in range(start, start + length):
                anomaly_timestamps[anomaly_type].add(t)

    for atype in sorted(anomaly_timestamps.keys()):
        num = len(anomaly_timestamps[atype])
        ratio = num / T
        print(f"{atype} anomaly timestamps: {num}")
        print(
            f"{atype} anomaly timestamp ratio: {ratio:.4f} "
            f"({ratio * 100:.2f}%)"
        )

    X_out_ours = X_out.clone()
    return (
        X_out.cpu().numpy(),
        X_out_ours.cpu().numpy(),
        mask.cpu().numpy(),
    )
