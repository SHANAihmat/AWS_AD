import torch
from torch.utils.data import Dataset


class TimeWindowDataset(Dataset):
    def __init__(self, obs_data, era5_data, ori_obs_data, edge_index_list, edge_attr_list, terrain_data, time_window):
        self.obs_data = obs_data    # Tensor
        self.era5_data = era5_data
        self.ori_obs_data = ori_obs_data
        self.terrain_data = terrain_data
        self.time_windows = [(i, i + time_window) for i in range(0, obs_data.shape[1] - time_window, time_window)]

        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list


    def __len__(self):
        return len(self.time_windows)

    def __getitem__(self, index):
        t0, t1 = self.time_windows[index]
        time_window = torch.tensor([t0, t1])
        labels = self.ori_obs_data[:, t0:t1]
        era5_supervise = self.era5_data[:, t0:t1]
        aws_data = self.obs_data[:, t0:t1]
        terrain_data = self.terrain_data
        edge_index = self.edge_index_list[index]
        edge_attr = self.edge_attr_list[index]
        return aws_data, era5_supervise, terrain_data, labels, time_window, edge_index, edge_attr