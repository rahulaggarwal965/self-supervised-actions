import torch.nn.functional as F


class UsageLoss:
    """Low per-sample code entropy (decisive) + high batch entropy (all codes
    used). Off by default."""

    def __init__(self, sample_weight: float = 1.0, batch_weight: float = 1.0) -> None:
        self.sample_weight = sample_weight
        self.batch_weight = batch_weight

    def __call__(self, out, batch, model):
        probs = F.softmax(-out.vq["dist"], dim=1)
        h_sample = -(probs * (probs + 1e-10).log()).sum(1).mean()
        p_bar = probs.mean(0)
        h_batch = -(p_bar * (p_bar + 1e-10).log()).sum()
        loss = self.sample_weight * h_sample - self.batch_weight * h_batch
        return loss, {"h_sample": h_sample.item(), "h_batch": h_batch.item()}
