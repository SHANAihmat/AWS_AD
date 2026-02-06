import torch
import torch.nn as nn


class EmbeddingGenerator(nn.Module):
    def __init__(self, time_input_dim, embed_dim):
        super(EmbeddingGenerator, self).__init__()
        self.fc_weather = nn.Linear(time_input_dim, embed_dim)
        self.fc_weather_lstm = nn.LSTM(
            input_size=10,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True,
        )

        self.conv_terrain = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, aws_data, era5_data, terrain_data):
        embeddings = []

        for i in range(aws_data.size(0)):
            concat_data = torch.cat(
                [aws_data[i, ...], era5_data[i, ...]], dim=-1
            )

            input_to_lstm = concat_data.unsqueeze(0)
            _, (h_n, _) = self.fc_weather_lstm(input_to_lstm)
            weather_embedding = h_n[-1]
            weather_embedding = weather_embedding.repeat(10, 1)

            terrain_data_single = terrain_data[i].unsqueeze(0)
            terrain_embedding = self.conv_terrain(
                terrain_data_single.unsqueeze(0)
            )
            terrain_embedding = (
                terrain_embedding.squeeze(-1)
                .squeeze(-1)
                .squeeze(0)
            )

            min_val = torch.min(terrain_embedding)
            max_val = torch.max(terrain_embedding)
            if max_val != min_val:
                terrain_embedding = (terrain_embedding - min_val) / (
                    max_val - min_val
                )
            else:
                terrain_embedding = torch.zeros_like(terrain_embedding)

            terrain_embedding = terrain_embedding.unsqueeze(0).expand(
                weather_embedding.size(0), -1
            )

            embedding = weather_embedding + terrain_embedding
            embeddings.append(embedding)

        return torch.stack(embeddings, dim=0)
