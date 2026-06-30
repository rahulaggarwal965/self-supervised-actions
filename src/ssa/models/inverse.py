import torch
from torch import Tensor, nn


class InverseModel(nn.Module):
    """Infers a pre-quantization action vector from a transition.

    With ``delta_input=False`` (default) the action is a function of
    ``concat(z_t, z_{t+1})``. With ``delta_input=True`` it is a function of the
    latent change ``z_{t+1} - z_t`` alone — so the code cannot encode absolute
    state (e.g. position), only the transition.
    """

    def __init__(
        self,
        dim: int = 256,
        action_dim: int = 64,
        hidden: int = 256,
        delta_input: bool = False,
    ) -> None:
        super().__init__()
        self.delta_input = delta_input
        in_dim = dim if delta_input else 2 * dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, action_dim)
        )
        self.action_dim = action_dim

    def forward(self, z_t: Tensor, z_tp1: Tensor) -> Tensor:
        x = (z_tp1 - z_t) if self.delta_input else torch.cat([z_t, z_tp1], dim=-1)
        return self.net(x)
