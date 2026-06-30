import torch.nn.functional as F


class ActionSupervisionLoss:
    """Partial action supervision (cheap when labels are free, as in a synthetic toy).

    On a small fraction of each (shuffled) batch, pull the pre-quantization action
    vector ``a_pre`` toward the codebook row reserved for that sample's true action,
    anchoring code ``i`` to action ``i`` (codes ``0..A-1`` become the action
    prototypes; any extra codes stay free). This makes the discovered code track the
    action rather than absolute position. LAOM (arXiv:2502.00379) reports ~2.5%
    labels give a large improvement under exactly this distractor/position confound.
    """

    def __init__(self, label_frac: float = 0.025) -> None:
        self.label_frac = label_frac

    def __call__(self, out, batch, model):
        a_pre = out.a_pre  # (B, action_dim)
        actions = batch.action  # (B,) long, values in 0..A-1
        b = a_pre.shape[0]
        k = max(1, round(self.label_frac * b))  # labeled count (loader shuffles each step)
        proto = model.quantizer.codebook.weight[actions[:k]]  # (k, action_dim)
        loss = F.mse_loss(a_pre[:k], proto)
        return loss, {"action_sup": loss.item(), "n_labeled": k}
