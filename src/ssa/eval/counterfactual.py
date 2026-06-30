import torch


@torch.no_grad()
def counterfactual_grid(model, obs):
    """Apply every codebook entry to a single context frame and decode.

    ``obs``: ``(T, C, H, W)`` history for one example. Returns ``(K, C, H, W)``
    predictions (meaningful only for a ``PixelDecoder`` head).
    """
    device = next(model.parameters()).device
    z = model.encoder(obs[-1:].to(device))  # (1, dim)
    codebook = model.quantizer.codebook.weight  # (K, action_dim)
    preds = [model.head.predict(model.dynamics(z, codebook[k : k + 1])) for k in range(len(codebook))]
    return torch.cat(preds, dim=0)
