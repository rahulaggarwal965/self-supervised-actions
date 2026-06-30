class DecorrelationLoss:
    """Decorrelate the action embedding from absolute position.

    Barlow-Twins-style: drive the cross-correlation between the pre-quantization
    action vector ``a_pre`` and the (detached) context latent ``z_ctx`` toward zero,
    so the code is pushed to carry information *not already present* in ``z_ctx`` —
    which, in this toy, is dominated by the agent's absolute position. Penalizes the
    mean squared normalized cross-correlation; uses no learnable parameters.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        self.eps = eps

    def __call__(self, out, batch, model):
        a = out.a_pre  # (B, A)
        z = out.z_ctx.detach()  # (B, D) — position lives here; don't move it
        a = (a - a.mean(0)) / (a.std(0) + self.eps)
        z = (z - z.mean(0)) / (z.std(0) + self.eps)
        b = a.shape[0]
        cross = (a.t() @ z) / (b - 1)  # (A, D) normalized cross-correlation
        loss = cross.pow(2).mean()
        return loss, {"decorr": loss.item()}
