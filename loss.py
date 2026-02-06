import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, lambda_temporal=0.1):
        super().__init__()
        self.lambda_temporal = lambda_temporal
        self.mse_loss = nn.MSELoss()

    def forward(self, recon, obs, era5):
        mask = ~torch.isnan(obs)
        if not mask.any():
            loss_recon = torch.tensor(0.0, device=recon.device, dtype=recon.dtype)
        else:
            loss_recon = self.mse_loss(recon[mask], obs[mask])

        grad_recon_t = recon[:, 1:, :] - recon[:, :-1, :]
        grad_era5_t = era5[:, 1:, :] - era5[:, :-1, :]
        loss_constraint_temporal = self.mse_loss(grad_recon_t, grad_era5_t)

        total_loss = loss_recon + self.lambda_temporal * loss_constraint_temporal
        return total_loss, loss_recon, loss_constraint_temporal
