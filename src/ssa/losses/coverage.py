import torch
from scipy.optimize import linear_sum_assignment


class FutureCoverageLoss:
    """Force the *forward* model to be action-conditional in raw latent space.

    The set of code-conditioned predictions ``{dynamics(z_t, code_k)}`` must COVER
    the real futures of this state ``{teacher(next under each action)}`` (the
    observed next plus the same-state counterfactuals), so applying different codes
    produces the different real action-outcomes — which is what makes the decoded
    counterfactual show distinct per-code moves.

    With ``one_to_one=True`` (default) each real future is assigned to a **distinct**
    code via a Hungarian assignment (on detached distances) before the loss — this
    prevents the degenerate collapse where every future grabs the same nearest code
    (a plain min-over-codes coverage collapsed: cov_distinct ~0.2, static
    counterfactual). Complements the counterfactual contrastive (which only shapes a
    projected delta subspace). Label-free: uses counterfactual frames, no action labels.
    """

    def __init__(self, one_to_one: bool = True) -> None:
        self.one_to_one = one_to_one

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
        a = futures.shape[1]
        dist = (preds.unsqueeze(2) - futures.unsqueeze(1)).pow(2).mean(dim=-1)  # (B, K, A)
        if self.one_to_one:
            # assign each future to a DISTINCT code (Hungarian on detached cost), then
            # the loss pulls those assigned code-predictions onto their future.
            cost = dist.detach().cpu().numpy()
            terms = []
            for i in range(b):
                fut, code = linear_sum_assignment(
                    cost[i].T
                )  # fut: 0..A-1, code: distinct code each
                terms.append(dist[i][code, fut].mean())  # differentiable gather of assigned pairs
            cover = torch.stack(terms).mean()
            distinct = torch.tensor(min(a, k) / a)  # one future per distinct code, by construction
        else:
            cover = dist.min(dim=1).values.mean()
            assign = dist.argmin(dim=1)
            distinct = (assign[:, None, :] != assign[:, :, None]).float().mean()
        return cover, {"future_cov": cover.item(), "cov_distinct": float(distinct)}
