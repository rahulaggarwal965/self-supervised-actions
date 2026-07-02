import torch


@torch.no_grad()
def counterfactual_grid(model, obs):
    """Apply every codebook entry to a single context frame and decode.

    ``obs``: ``(T, C, H, W)`` history for one example. Returns ``(K, C, H, W)``
    predictions (meaningful only for a ``PixelDecoder`` head).
    """
    device = next(model.parameters()).device
    it = obs[-1:].to(device)
    z = model.encoder(it)  # (1, dim)
    codebook = model.quantizer.codebook.weight  # (K, action_dim)
    preds = [
        model.head.predict(model.dynamics(z, codebook[k : k + 1]), it)
        for k in range(len(codebook))
    ]
    grid = torch.cat(preds, dim=0)  # (K, C, H, W)
    # a delta head predicts the change Δ = I_{t+1} - I_t; add I_t so the grid shows the
    # actual predicted next frame (agent moved) rather than the raw (cyan) delta. A
    # compositing head already returns the full frame (context passed above), so no add.
    if getattr(model.head, "delta", False):
        grid = grid + it
    return grid
