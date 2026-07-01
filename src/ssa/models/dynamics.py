import torch
from torch import Tensor, nn


class Dynamics(nn.Module):
    """Predicts the next-frame feature from ``(context, action)``.

    With ``residual=True`` the network predicts a *change* added to the context
    (``context + net(context, action)``, C-SWM-style). This additive/translational
    prior means the action can only modulate a delta on top of the current state,
    so it cannot re-encode the static scene (already in ``context``) — which pushes
    the code to carry only the action.

    With ``additive=True`` (implies residual) the delta is a *pure per-action
    displacement* ``T(action)`` that ignores the context: ``z_{t+1} = z_t + T(a)``
    (TransE / AC-LAM style). This puts all the model's capacity on the action's small
    latent footprint instead of re-predicting the large ``z_t``, so the prediction
    error can no longer swamp the action signal — and swapping the code is guaranteed
    to move the prediction by a distinct, state-independent amount.
    """

    def __init__(
        self,
        dim: int = 256,
        action_dim: int = 64,
        hidden: int = 256,
        residual: bool = False,
        additive: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual or additive
        self.additive = additive
        in_dim = action_dim if additive else dim + action_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, context: Tensor, action: Tensor) -> Tensor:
        inp = action if self.additive else torch.cat([context, action], dim=-1)
        delta = self.net(inp)
        return context + delta if self.residual else delta
