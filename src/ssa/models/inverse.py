import torch
from torch import Tensor, nn


class InverseModel(nn.Module):
    """Infers a pre-quantization action vector from ``(z_t, z_{t+1})``."""

    def __init__(self, dim: int = 256, action_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(), nn.Linear(hidden, action_dim)
        )
        self.action_dim = action_dim

    def forward(self, z_t: Tensor, z_tp1: Tensor) -> Tensor:
        return self.net(torch.cat([z_t, z_tp1], dim=-1))
