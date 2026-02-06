import torch
import torch.nn as nn
from multi_source_data_fusion import EmbeddingGenerator
from torch_geometric.nn import MessagePassing


class KalmanFilterLayer(nn.Module):
    def __init__(self, state_dim, obs_dim, dt=1.0, device=None):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.A = nn.Parameter(torch.eye(state_dim, device=self.device))
        self.B = nn.Parameter(torch.zeros(state_dim, 1, device=self.device))
        self.H = nn.Parameter(
            torch.cat(
                [
                    torch.eye(obs_dim, device=self.device),
                    torch.zeros(obs_dim, obs_dim, device=self.device),
                ],
                dim=1,
            )
        )
        self.Q = nn.Parameter(torch.eye(state_dim, device=self.device) * 0.01)
        self.R = nn.Parameter(torch.eye(obs_dim, device=self.device) * 0.1)

        self.register_buffer("min_cov", torch.tensor(1e-6, device=self.device))

    def forward(self, observations, control=None):
        B, N, T, F = observations.shape
        state_dim = self.state_dim
        device = observations.device

        x = torch.zeros(B, N, state_dim, device=device)
        P = (
            torch.eye(state_dim, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(B, N, 1, 1)
            * 1.0
        )

        filtered = torch.zeros(B, N, T, F, device=device)

        for t in range(T):
            u = (
                control[:, :, t, :]
                if control is not None
                else torch.zeros(B, N, 1, device=device)
            )

            x = torch.matmul(self.A, x.unsqueeze(-1)).squeeze(-1) + torch.matmul(
                self.B, u.unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)

            P = torch.matmul(
                self.A, torch.matmul(P, self.A.transpose(-2, -1))
            ) + torch.clamp(self.Q, min=self.min_cov)

            hx = torch.matmul(self.H, x.unsqueeze(-1)).squeeze(-1)
            y = observations[:, :, t, :] - hx

            HP = torch.matmul(self.H, P)
            S = torch.matmul(HP, self.H.transpose(-2, -1)) + torch.clamp(
                self.R, min=self.min_cov
            )

            Ht = self.H.transpose(-2, -1)
            PHt = torch.matmul(P, Ht)

            S_inv_y = torch.linalg.solve(S, y.unsqueeze(-1)).squeeze(-1)
            x = x + torch.matmul(PHt, S_inv_y.unsqueeze(-1)).squeeze(-1)

            K = torch.matmul(PHt, torch.linalg.inv(S))
            P = P - torch.matmul(K, torch.matmul(self.H, P))

            filtered[:, :, t, :] = x[:, :, :F]

        return filtered


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = x + residual if x.shape == residual.shape else x

        residual = x
        x = self.fc2(x)
        x = self.norm2(x)
        x = x + residual if x.shape == residual.shape else x
        return x


class GNN(MessagePassing):
    def __init__(
        self,
        num_layers,
        in_dim,
        hidden_dim,
        out_dim,
        time_input_dim,
        time_embedding_dim,
        aggr="mean",
    ):
        super(GNN, self).__init__(aggr="add")
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            layer_in_dim = in_dim * time_embedding_dim if i == 0 else hidden_dim
            layer_out_dim = hidden_dim if i < num_layers - 1 else out_dim
            self.layers.append(nn.Linear(layer_in_dim, layer_out_dim))
            self.norms.append(nn.LayerNorm(layer_out_dim))

        self.activation = nn.ReLU()
        self.embedding_generator = EmbeddingGenerator(
            time_input_dim=time_input_dim, embed_dim=time_embedding_dim
        )

        self.learned_edge_bias = nn.Parameter(torch.tensor(0.0))

        self.fusion_linear = nn.Linear(10, 5)

        self.kf_layer = KalmanFilterLayer(state_dim=10, obs_dim=5)
        self.kf_proj_f = nn.Linear(5, 10)
        self.kf_proj_t = nn.Linear(72, 32)

    def forward(self, aws_data, era5_data, terrain_data, time_window, edge_index, edge_attr):
        B = aws_data.shape[0]
        N = aws_data.shape[1]

        embeddings = []
        for i in range(B):
            embedding = self.embedding_generator(
                aws_data[i, ...], era5_data[i, ...], terrain_data[i, ...]
            )
            embeddings.append(embedding)

        kf_filtered = self.kf_layer(era5_data)
        kf_filtered = self.kf_proj_f(kf_filtered)
        kf_filtered = self.kf_proj_t(kf_filtered.permute(0, 1, 3, 2))

        embeddings = torch.stack(embeddings, dim=0)
        embeddings = embeddings + kf_filtered

        B, N, F_, D = embeddings.shape
        T = era5_data.shape[2]

        x_t = embeddings.view(B * N, -1)

        edge_indices = []
        for i in range(B):
            offset = i * N
            edge_index_i = edge_index[i] + offset
            edge_indices.append(edge_index_i)

        edge_index_batch = torch.cat(edge_indices, dim=1)

        if edge_attr is not None:
            edge_attr_batch = edge_attr.view(-1) + self.learned_edge_bias
        else:
            edge_attr_batch = None

        norm_flag = 0
        for layer in self.layers:
            residual = x_t
            x_t = self.propagate(
                edge_index=edge_index_batch,
                x=x_t,
                edge_weight=edge_attr_batch,
                layer=layer,
            )
            x_t = self.activation(x_t)
            x_t = self.norms[norm_flag](x_t)
            norm_flag += 1
            x_t = x_t + residual if x_t.shape == residual.shape else x_t

        x_t = x_t.view(B, N, T, F_ // 2)
        outputs = self.activation(x_t)
        return outputs

    def message(self, x_j, edge_weight, layer):
        msg = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return layer(msg)

    def update(self, aggr_out):
        return self.activation(aggr_out)


class GNNWithMLP(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        hidden_dim,
        out_dim,
        time_embedding_dim,
        time_input_dim,
        mlp_hidden_dim,
        mlp_out_dim,
        aggr="mean",
    ):
        super(GNNWithMLP, self).__init__()

        self.gnn = GNN(
            num_layers=num_layers,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            time_embedding_dim=time_embedding_dim,
            time_input_dim=time_input_dim,
            aggr=aggr,
        )

        self.mlp = MLP(input_dim=out_dim, hidden_dim=mlp_hidden_dim, output_dim=mlp_out_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, aws_data, era5_data, terrain_data, time_window, edge_index, edge_attr):
        gnn_output = self.relu(
            self.gnn(aws_data, era5_data, terrain_data, time_window, edge_index, edge_attr)
        )

        batch_size, num_nodes, time_window, gnn_out_dim = gnn_output.shape
        gnn_output_flat = gnn_output.view(-1, gnn_out_dim)
        mlp_output_flat = self.mlp(gnn_output_flat)
        mlp_output = mlp_output_flat.view(batch_size, num_nodes, time_window, gnn_out_dim)

        return mlp_output
