import torch.nn.functional as F


class PredictionLoss:
    """MSE between the head's prediction and its target (pixels or latents)."""

    def __call__(self, out, batch, model):
        loss = F.mse_loss(out.pred, out.target)
        return loss, {"pred": loss.item()}
