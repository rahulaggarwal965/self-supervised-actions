import torch


class VICRegLoss:
    """VICReg variance + covariance regularization on the encoder embedding
    (``out.z_ctx``), to prevent representational collapse in latent prediction.

    - variance: a hinge encouraging each embedding dim's std to be >= ``gamma``.
    - covariance: pushes off-diagonal covariance toward zero (decorrelation).

    Off by default; intended for the latent-prediction setup where a plain
    joint-embedding objective collapses to a constant embedding.
    """

    def __init__(
        self,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma
        self.eps = eps

    def __call__(self, out, batch, model):
        z = out.z_ctx  # (B, D)
        std = torch.sqrt(z.var(dim=0) + self.eps)
        var_loss = torch.relu(self.gamma - std).mean()
        zc = z - z.mean(dim=0)
        cov = (zc.t() @ zc) / (z.shape[0] - 1)
        d = z.shape[1]
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d
        loss = self.var_weight * var_loss + self.cov_weight * cov_loss
        return loss, {
            "vicreg": loss.item(),
            "var": var_loss.item(),
            "cov": cov_loss.item(),
            "z_std": float(std.mean().detach()),
        }
