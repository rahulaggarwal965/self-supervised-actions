import torch
from torch import Tensor, nn


class InverseModel(nn.Module):
    """Infers a pre-quantization action vector from a transition.

    With ``delta_input=False`` (default) the action is a function of
    ``concat(z_t, z_{t+1})``. With ``delta_input=True`` it is a function of the
    latent change ``z_{t+1} - z_t`` alone — so the code cannot encode absolute
    state (e.g. position), only the transition.
    """

    spatial = False

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


class InvariantInverseModel(nn.Module):
    """Position-invariant action inference.

    Infers the pre-quantization action from the inter-frame *feature-map*
    difference ``features(I_{t+1}) - features(I_t)`` via a small conv followed by
    a global average pool over space. The conv uses circular padding so it is
    shift-equivariant; the spatial mean is then shift-invariant. The result: the
    code depends on the *local* change pattern (a left-move's signature), not on
    the agent's absolute position.

    ``feat_level`` selects which encoder feature map the action is read from
    (the model passes it to ``encoder.features(x, level=feat_level)``). ``None``
    uses the final coarse map; a smaller level (e.g. 2 → 16x16) gives finer
    spatial resolution so a small, sub-cell motion is resolvable. ``feat_ch``
    must match the channel count at that level.
    """

    spatial = True

    def __init__(
        self,
        feat_ch: int = 256,
        action_dim: int = 64,
        hidden: int = 256,
        feat_level: int | None = None,
    ) -> None:
        super().__init__()
        self.feat_level = feat_level
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.proj = nn.Linear(hidden, action_dim)
        self.action_dim = action_dim

    def forward(self, f_t: Tensor, f_tp1: Tensor) -> Tensor:
        d = f_tp1 - f_t  # (B, feat_ch, H', W') — the motion signature
        h = self.net(d).mean(dim=(2, 3))  # global average pool -> (B, hidden)
        return self.proj(h)
