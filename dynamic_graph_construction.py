import torch
import numpy as np
from haversine import haversine
from fastdtw import fastdtw
import pandas as pd
from data_processing import read_json
import pickle
import os


def keep_topk_edges(adj_matrix, k):
    num_nodes = adj_matrix.size(0)
    adj_matrix_new = torch.zeros_like(adj_matrix)
    for i in range(num_nodes):
        row = adj_matrix[i]
        if torch.count_nonzero(row) > k:
            topk_values, topk_indices = torch.topk(row, k)
            adj_matrix_new[i, topk_indices] = topk_values
        else:
            adj_matrix_new[i] = row
    return adj_matrix_new


def dense_to_sparse(adj_matrix, topk=10):
    adj_matrix = keep_topk_edges(adj_matrix, topk)
    edge_mask = adj_matrix != 0
    edge_indices = torch.nonzero(edge_mask, as_tuple=False).t()
    edge_weights = adj_matrix[edge_mask]
    return edge_indices, edge_weights


def haversine_distance(json_path):
    station_info = read_json(json_path)
    station_num = len(station_info["stations"])
    station_lon = [station["longitude"] for station in station_info["stations"]]
    station_lat = [station["latitude"] for station in station_info["stations"]]

    distance_matrix = np.zeros((station_num, station_num))
    for i in range(station_num):
        for j in range(station_num):
            distance_matrix[i, j] = haversine(
                (station_lat[i], station_lon[i]),
                (station_lat[j], station_lon[j]),
            )
    return distance_matrix


def dtw_distance_per_feature(series1, series2):
    if isinstance(series1, torch.Tensor):
        series1 = series1.cpu().numpy()
    if isinstance(series2, torch.Tensor):
        series2 = series2.cpu().numpy()

    distance, _ = fastdtw(series1, series2)
    return distance


def meteorological_distance(data, time_window):
    num_points, _, num_features = data.shape
    start_idx, end_idx = time_window
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            feature_distances = []
            for f in range(num_features):
                series1 = data[i, start_idx:end_idx, f]
                series2 = data[j, start_idx:end_idx, f]
                feature_distances.append(
                    dtw_distance_per_feature(series1, series2)
                )
            distance_matrix[i, j] = np.mean(feature_distances)

    return distance_matrix


def construct_dynamic_graph(
    geo_matrix,
    meteorological_data,
    time_window,
    sigma_geo=1.0,
    sigma_met=1.0,
):
    met_matrix = meteorological_distance(meteorological_data, time_window)

    geo_adj = np.exp(-geo_matrix**2 / (2 * sigma_geo**2))
    met_adj = np.exp(-met_matrix**2 / (2 * sigma_met**2))

    lamda = 1.25
    adjacency_matrix = geo_adj + lamda * met_adj
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
    adjacency_matrix = torch.where(
        adjacency_matrix == 0, 1.00e-30, adjacency_matrix
    )
    return adjacency_matrix


def produce_all_graphs(era5_data, geo_matrix, path, flag):
    edge_index_list = []
    edge_attr_list = []
    time_windows = [
        (i, i + 72) for i in range(0, era5_data.shape[1] - 72, 72)
    ]

    for t0, t1 in time_windows:
        window = [t0, t1]
        adj_matrix = construct_dynamic_graph(
            geo_matrix, era5_data[:, t0:t1, :], window
        )
        edge_index, edge_attr = dense_to_sparse(adj_matrix)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)

    with open(
        os.path.join(path, f"graph_structure/{flag}_edge_index.pkl"), "wb"
    ) as f:
        pickle.dump(edge_index_list, f)

    with open(
        os.path.join(path, f"graph_structure/{flag}_edge_attr.pkl"), "wb"
    ) as f:
        pickle.dump(edge_attr_list, f)

    return edge_index_list, edge_attr_list


if __name__ == "__main__":
    path = "/data/numpy_France"
    geo_matrix = haversine_distance(
        "/data/France_station_info.json"
    )

    train_era5_data = np.load(
        os.path.join(path, "era5/train_era5_data.npy")
    )
    train_edge_index_list, train_edge_attr_list = produce_all_graphs(
        train_era5_data, geo_matrix, path, flag="train"
    )

    val_era5_data = np.load(
        os.path.join(path, "era5/val_era5_data.npy")
    )
    val_edge_index_list, val_edge_attr_list = produce_all_graphs(
        val_era5_data, geo_matrix, path, flag="val"
    )

    test_era5_data = np.load(
        os.path.join(path, "era5/test_era5_data.npy")
    )
    test_edge_index_list, test_edge_attr_list = produce_all_graphs(
        test_era5_data, geo_matrix, path, flag="test"
    )

    print(
        "Edge index list length (train, val, test):",
        len(train_edge_index_list),
        len(val_edge_index_list),
        len(test_edge_index_list),
    )
    print(
        "Edge attribute list length (train, val, test):",
        len(train_edge_attr_list),
        len(val_edge_attr_list),
        len(test_edge_attr_list),
    )
