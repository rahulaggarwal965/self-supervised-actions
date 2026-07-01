import torch

from ssa.models.heads import LatentHead, PixelDecoder


class SinkhornCoverageLoss:
    """Force the K code-conditioned predictions to COVER the real per-action outcomes.

    The action is a small, low-variance part of the signal, so an MSE forward model
    minimizes loss by predicting the *mean* outcome and ignoring the discrete code
    (the counterfactual becomes a static/averaged blur). This loss instead requires
    the set ``{head(dynamics(z, code_k))}`` to match-and-cover the real outcomes of
    this state ``{observed next} ∪ {same-state counterfactuals}`` via a differentiable
    **Sinkhorn** assignment: a doubly-stochastic transport plan (rows = the K codes,
    cols = the A real futures) whose entropic-balanced marginals stop every future
    collapsing onto one code. Minimizing the transported cost pulls distinct codes onto
    distinct real action-outcomes — the one signal that makes the forward model produce
    the *correct distinct* per-action result instead of the mean.

    Works in whatever space the head predicts: raw frames / frame-deltas (PixelDecoder)
    or teacher latents (LatentHead). Label-free — uses counterfactual frames, no labels.
    """

    def __init__(self, eps: float = 0.05, iters: int = 20) -> None:
        self.eps = eps
        self.iters = iters

    def _futures(self, batch, model):
        """Real per-action outcomes in the head's prediction space: (B, A, -1)."""
        it = batch.obs[:, -1]
        frames = torch.cat([batch.next_obs.unsqueeze(1), batch.next_cf], dim=1)  # (B, A, C,H,W)
        b, a = frames.shape[0], frames.shape[1]
        if isinstance(model.head, LatentHead):
            with torch.no_grad():
                return model.teacher(frames.reshape(b * a, *frames.shape[2:])).reshape(b, a, -1)
        if isinstance(model.head, PixelDecoder) and model.head.delta:
            frames = frames - it.unsqueeze(1)
        return frames.reshape(b, a, -1)

    def _sinkhorn(self, cost):
        """Entropic OT plan for cost (B, K, A) with uniform marginals 1/K, 1/A."""
        b, k, a = cost.shape
        log_t = -cost / self.eps
        log_r = torch.full((b, k, 1), -torch.log(torch.tensor(float(k))), device=cost.device)
        log_c = torch.full((b, 1, a), -torch.log(torch.tensor(float(a))), device=cost.device)
        for _ in range(self.iters):
            log_t = log_t - torch.logsumexp(log_t, dim=2, keepdim=True) + log_r
            log_t = log_t - torch.logsumexp(log_t, dim=1, keepdim=True) + log_c
        return log_t.exp()

    def __call__(self, out, batch, model):
        z = out.z_ctx
        cb = model.quantizer.codebook.weight
        k, b = cb.shape[0], z.shape[0]
        preds = torch.stack(
            [model.head.predict(model.dynamics(z, cb[i : i + 1].expand(b, -1))) for i in range(k)],
            dim=1,
        ).reshape(b, k, -1)  # (B, K, D)
        futures = self._futures(batch, model)  # (B, A, D)
        cost = (preds.unsqueeze(2) - futures.unsqueeze(1)).pow(2).mean(dim=-1)  # (B, K, A)
        plan = self._sinkhorn(cost.detach())  # transport is a target; cost carries the gradient
        cover = (plan * cost).sum(dim=(1, 2)).mean()
        with torch.no_grad():
            # coverage: mean nearest-code distance per real future, vs the futures' own spread
            near = cost.min(dim=1).values.mean().item()
            spread = torch.cdist(futures, futures).mean().item() + 1e-6
        return cover, {"sink_cover": cover.item(), "cover_ratio": near / spread}
