import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    """VQ bottleneck with straight-through estimator.

    Returns the quantized vector and an info dict with commit/codebook losses,
    batch perplexity, hard code ids, and squared distances to every code (the
    usage loss consumes ``dist``).
    """

    def __init__(self, num_codes: int, dim: int) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.codebook = nn.Embedding(num_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: Tensor) -> tuple[Tensor, dict]:
        w = self.codebook.weight
        # Squared L2 distance ||z - w||^2 = ||z||^2 - 2 z.w^T + ||w||^2, in this
        # expanded form to avoid materializing the (B, num_codes, dim) diff tensor.
        dist = z.pow(2).sum(1, keepdim=True) - 2 * z @ w.t() + w.pow(2).sum(1)
        codes = dist.argmin(dim=1)
        z_q = self.codebook(codes)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()  # straight-through
        with torch.no_grad():
            usage = F.one_hot(codes, self.num_codes).float().mean(0)
            perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())
        info = {
            "codes": codes,
            "dist": dist,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
        }
        return z_q, info
