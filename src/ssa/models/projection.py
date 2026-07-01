from torch import Tensor, nn


class ProjectionHead(nn.Module):
    """Small MLP that maps a latent to a contrastive embedding space.

    The counterfactual contrastive is computed in this projected space, while the
    prediction/regression loss stays on the raw latent — so a strong contrastive
    can shape action-discriminative *directions* without pulling the raw prediction
    off the teacher-latent manifold (SimCLR "throw away the projection head" trick).
    """

    def __init__(self, dim: int = 256, hidden: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
