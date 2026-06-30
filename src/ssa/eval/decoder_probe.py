import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure

from ssa.models.heads import PixelDecoder


def train_decoder_probe(
    model, loader, device="cpu", steps: int = 500, lr: float = 1e-3
) -> PixelDecoder:
    """Train a pixel decoder ``D(z) -> image`` on the FROZEN encoder, so latent
    predictions can be rendered back to pixels for visualization.

    Reconstructs next frames from their encoder latents (``D(encoder(I)) ~ I``);
    the encoder / model is never updated. Used to visualize latent-head runs,
    which have no decoder of their own.
    """
    decoder = PixelDecoder(dim=model.encoder.dim).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    step = 0
    while step < steps:
        for batch in loader:
            frames = batch.next_obs.to(device)
            with torch.no_grad():
                z = model.encoder(frames)
            loss = F.mse_loss(decoder.predict(z), frames)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= steps:
                break
    return decoder


@torch.no_grad()
def decoded_counterfactual_figure(model, decoder: PixelDecoder, obs) -> Figure:
    """``I_t`` plus, for each codebook entry, the decoded predicted next frame
    ``D(dynamics(z_t, code_k))`` — a true latent->pixel counterfactual for
    latent-head models. ``obs``: ``(T, C, H, W)`` history for one example."""
    device = next(model.parameters()).device
    z = model.encoder(obs[-1:].to(device))
    codebook = model.quantizer.codebook.weight
    k = codebook.shape[0]
    fig, axes = plt.subplots(1, k + 1, figsize=(2 * (k + 1), 2), squeeze=False)
    row = axes[0]
    row[0].imshow(obs[-1].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    row[0].set_title("I_t")
    row[0].axis("off")
    for i in range(k):
        img = decoder.predict(model.dynamics(z, codebook[i : i + 1]))[0]
        row[i + 1].imshow(img.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        row[i + 1].set_title(f"code {i}")
        row[i + 1].axis("off")
    fig.tight_layout()
    return fig
