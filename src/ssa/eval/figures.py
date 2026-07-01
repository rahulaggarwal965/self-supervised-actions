import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from ssa.eval.counterfactual import counterfactual_grid


def _show(ax, img: torch.Tensor) -> None:
    ax.imshow(img.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    ax.axis("off")


@torch.no_grad()
def reconstruction_panel(model, batch, n: int = 8) -> Figure:
    """Rows of [I_t, true I_{t+1}, predicted I_{t+1}] (PixelDecoder head only).

    For a delta head the prediction is the change ``Δ = I_{t+1} - I_t``; we add ``I_t``
    back so the panel shows the actual reconstructed frame, not the (cyan) delta.
    """
    out = model(batch)
    n = min(n, batch.obs.shape[0])
    delta = getattr(model.head, "delta", False)
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n), squeeze=False)
    titles = ["I_t", "true I_{t+1}", "pred I_{t+1}"]
    for i in range(n):
        pred_frame = batch.obs[i, -1] + out.pred[i] if delta else out.pred[i]
        imgs = [batch.obs[i, -1], batch.next_obs[i], pred_frame]
        for j, img in enumerate(imgs):
            _show(axes[i][j], img)
            if i == 0:
                axes[i][j].set_title(titles[j])
    fig.tight_layout()
    return fig


@torch.no_grad()
def counterfactual_figure(model, obs) -> Figure:
    """I_t plus one decoded panel per codebook entry (PixelDecoder head only).

    ``obs``: ``(T, C, H, W)`` history for one example."""
    grid = counterfactual_grid(model, obs)  # (K, C, H, W)
    k = grid.shape[0]
    fig, axes = plt.subplots(1, k + 1, figsize=(2 * (k + 1), 2), squeeze=False)
    row = axes[0]
    _show(row[0], obs[-1])
    row[0].set_title("I_t")
    for i in range(k):
        _show(row[i + 1], grid[i])
        row[i + 1].set_title(f"code {i}")
    fig.tight_layout()
    return fig


def code_action_confusion(codes, labels, num_codes: int) -> Figure:
    """Heatmap of discovered code (rows) vs. ground-truth action (cols)."""
    codes = np.asarray([int(c) for c in codes])
    labels = np.asarray([int(x) for x in labels])
    num_actions = int(labels.max()) + 1 if labels.size else 1
    mat = np.zeros((num_codes, num_actions), dtype=int)
    for c, x in zip(codes, labels, strict=True):
        mat[c, x] += 1
    fig, ax = plt.subplots(figsize=(2 + 0.5 * num_actions, 2 + 0.35 * num_codes))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xlabel("true action")
    ax.set_ylabel("code")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def codebook_usage_bar(codes, num_codes: int) -> Figure:
    """Per-code usage count over the eval set (collapse diagnostic)."""
    counts = np.bincount(np.asarray([int(c) for c in codes]), minlength=num_codes)
    fig, ax = plt.subplots(figsize=(2 + 0.3 * num_codes, 2))
    ax.bar(range(num_codes), counts[:num_codes])
    ax.set_xlabel("code")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig
