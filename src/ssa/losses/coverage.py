import torch


class FutureCoverageLoss:
    """Force the *forward* model to be action-conditional in raw latent space.

    The set of code-conditioned predictions ``{dynamics(z_t, code_k)}`` must COVER
    the real futures of this state ``{teacher(next under each action)}`` (the
    observed next plus the same-state counterfactuals). Each real future is matched
    by its nearest code-prediction (min over codes), so *applying different codes
    produces the different real action-outcomes* — which is exactly what makes the
    decoded counterfactual show distinct per-code moves.

    This complements the counterfactual *contrastive* (which only shapes a projected
    delta subspace and leaves the raw dynamics output code-agnostic — high NMI but a
    static rendered counterfactual). Label-free: uses counterfactual frames, never
    action labels.
    """

    def __call__(self, out, batch, model):
        z = out.z_ctx  # (B, D)
        cb = model.quantizer.codebook.weight  # (K, A)
        k, b = cb.shape[0], z.shape[0]
        preds = torch.stack(
            [model.head.predict(model.dynamics(z, cb[i : i + 1].expand(b, -1))) for i in range(k)],
            dim=1,
        )  # (B, K, D)
        cf = batch.next_cf  # (B, M, C, H, W) — same-state futures under other actions
        m = cf.shape[1]
        with torch.no_grad():
            negs = model.teacher(cf.reshape(b * m, *cf.shape[2:])).reshape(b, m, -1)
        futures = torch.cat([out.target.detach().unsqueeze(1), negs], dim=1)  # (B, A, D)
        # (B, K, A) squared distances between each code-prediction and each real future
        dist = (preds.unsqueeze(2) - futures.unsqueeze(1)).pow(2).mean(dim=-1)
        cover = dist.min(dim=1).values.mean()  # each future matched by its nearest code
        # diagnostic: fraction of futures whose nearest code is distinct (coverage spread)
        assign = dist.argmin(dim=1)  # (B, A) which code covers each future
        distinct = (assign[:, None, :] != assign[:, :, None]).float().mean()
        return cover, {"future_cov": cover.item(), "cov_distinct": distinct.item()}
