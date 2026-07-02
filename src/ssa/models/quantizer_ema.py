import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EMAVectorQuantizer(nn.Module):
    """VQ bottleneck with **EMA codebook updates** (van den Oord 2017) + dead-code reset.

    Drop-in for ``VectorQuantizer`` (same ``.codebook.weight`` interface, same info dict),
    but the codebook is updated by an exponential moving average toward the encoder outputs
    assigned to each code instead of the hard MSE ``codebook_loss``. Motivation (subexp 19):
    with the gradient-updated codebook the discovery phase transition is a violent,
    seed-dependent event — when codes reassign, ``codebook_loss`` and grad-norm spike. An EMA
    codebook tracks the encoding distribution smoothly, which should make the transition
    earlier and gentler. Only the commitment term trains the encoder here (``codebook_loss``
    is returned as 0 — the EMA, not gradient descent, moves the codes).

    Dead codes (unused in a batch) are reset to a random batch encoding so the codebook stays
    live (also prevents the EMA blow-up for persistently-unused codes).
    """

    def __init__(
        self, num_codes: int, dim: int, decay: float = 0.99, eps: float = 1e-5,
        reset_dead: bool = True,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.reset_dead = reset_dead
        self.codebook = nn.Embedding(num_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.codebook.weight.requires_grad_(False)  # EMA-updated, not by the optimizer
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", self.codebook.weight.data.clone())

    def forward(self, z: Tensor) -> tuple[Tensor, dict]:
        # detached clone so the in-place EMA update below can't invalidate the autograd
        # graph of `dist` (which the usage loss backprops through to the encoder).
        w = self.codebook.weight.detach().clone()
        dist = z.pow(2).sum(1, keepdim=True) - 2 * z @ w.t() + w.pow(2).sum(1)
        codes = dist.argmin(dim=1)
        z_q = F.embedding(codes, w)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(codes, self.num_codes).type_as(z)  # (B, K)
                n = onehot.sum(0)  # (K,)
                self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.t() @ z, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                smoothed = (  # Laplace smoothing so rarely-used codes don't blow up
                    (self.cluster_size + self.eps) / (total + self.num_codes * self.eps) * total
                )
                new_cb = self.embed_avg / smoothed.unsqueeze(1)
                if self.reset_dead:
                    dead = n < 1.0
                    if dead.any():
                        k_dead = int(dead.sum().item())
                        idx = torch.randint(0, z.shape[0], (k_dead,), device=z.device)
                        new_cb[dead] = z[idx]
                        self.embed_avg[dead] = z[idx]
                        self.cluster_size[dead] = 1.0
                self.codebook.weight.data.copy_(new_cb)
        commit_loss = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()  # straight-through
        with torch.no_grad():
            usage = F.one_hot(codes, self.num_codes).float().mean(0)
            perplexity = torch.exp(-(usage * (usage + 1e-10).log()).sum())
        return z_q, {
            "codes": codes,
            "dist": dist,
            "commit_loss": commit_loss,
            "codebook_loss": z.new_zeros(()),  # EMA moves the codebook, no gradient term
            "perplexity": perplexity,
        }
