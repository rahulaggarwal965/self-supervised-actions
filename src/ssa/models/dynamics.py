import torch
from torch import Tensor, nn


class Dynamics(nn.Module):
    """Predicts the next-frame feature from ``(context, action)``.

    With ``residual=True`` the network predicts a *change* added to the context
    (``context + net(context, action)``, C-SWM-style). This additive/translational
    prior means the action can only modulate a delta on top of the current state,
    so it cannot re-encode the static scene (already in ``context``) — which pushes
    the code to carry only the action.
    """

    def __init__(
        self, dim: int = 256, action_dim: int = 64, hidden: int = 256, residual: bool = False
    ) -> None:
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(dim + action_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )

    def forward(self, context: Tensor, action: Tensor) -> Tensor:
        delta = self.net(torch.cat([context, action], dim=-1))
        return context + delta if self.residual else delta
