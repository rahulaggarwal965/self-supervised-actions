class VQLoss:
    """Codebook + commitment loss from the quantizer info dict."""

    def __init__(self, beta: float = 0.25) -> None:
        self.beta = beta

    def __call__(self, out, batch, model):
        loss = out.vq["codebook_loss"] + self.beta * out.vq["commit_loss"]
        return loss, {"vq": loss.item(), "perplexity": out.vq["perplexity"].item()}
