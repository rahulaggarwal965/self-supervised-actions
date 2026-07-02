import torch
import torch.nn.functional as F


class MarginLoss:
    """No-action counterfactual: prediction with the inferred action must beat
    prediction with a zero action by margin ``m`` (off by default)."""

    def __init__(self, m: float = 0.1) -> None:
        self.m = m

    def __call__(self, out, batch, model):
        zero = torch.zeros_like(out.a_q)
        pred0 = model.head.predict(model.dynamics(out.z_ctx, zero), batch.obs[:, -1])
        err = F.mse_loss(out.pred, out.target)
        err0 = F.mse_loss(pred0, out.target)
        loss = F.relu(self.m + err - err0)
        return loss, {"margin": loss.item(), "err": err.item(), "err0": err0.item()}
