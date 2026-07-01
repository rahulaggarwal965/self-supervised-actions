import torch
from torch import Tensor, nn


class Dynamics(nn.Module):
    """Predicts the next-frame feature from ``(context, action)``."""

    def __init__(self, dim: int = 256, action_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + action_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )

    def forward(self, context: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([context, action], dim=-1))
