import torch
import torch.nn.functional as F


class CounterfactualContrastiveLoss:
    """Make the action necessary — against *real* counterfactual futures.

    ``dynamics(z_t, code)`` must predict the observed next latent better than the
    teacher encodings of the same state's next frames under the *other* actions
    (``batch.next_cf``). InfoNCE with the observed next as the positive and the
    counterfactual futures as negatives.

    Unlike the code-contrastive (which collapsed the codebook, since one code
    trivially beats the model's own bad predictions under unused codes), the
    negatives here are genuinely-different real futures: a single code cannot
    predict all of them, so the code *must* encode which action occurred —
    collapse is structurally impossible. Label-free (uses counterfactual frames,
    never action labels).

    With ``normalize=True`` (default) the similarity is **cosine** on L2-normalized
    embeddings, so the contrastive optimizes *direction* not scale: it can no longer
    "win relatively" by drifting the prediction off the teacher-latent manifold (the
    raw-L2 variant did — ``cf_acc`` saturated at 1.0 while ``action_err`` blew up
    ~10x), which let it fight the prediction loss. Normalizing decouples the two.

    ``mode="hinge"`` uses the C-SWM energy instead of a softmax: minimize the positive
    energy ``H_pos = ‖pred − teacher(next)‖²`` and push each counterfactual energy up
    to a ``margin`` (``max(0, margin − H_cf)``). The hinge is **bounded** — once a
    counterfactual is beaten by the margin it contributes no gradient — so the loss
    cannot trade prediction accuracy for an ever-larger relative gap. Pairs with an
    additive/residual dynamics (``pred = z_t + T``), whose ``H_pos`` is the on-manifold
    anchor. Best with the raw (un-normalized) energy.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        normalize: bool = True,
        mode: str = "infonce",
        margin: float = 1.0,
    ) -> None:
        self.temperature = temperature
        self.normalize = normalize
        self.mode = mode
        self.margin = margin

    def __call__(self, out, batch, model):
        pred = out.pred  # (B, D) — predicted next latent, dynamics(z_t, a_q)
        pos = out.target.detach()  # (B, D) — teacher(observed next)
        cf = batch.next_cf  # (B, M, C, H, W)
        b, m = cf.shape[0], cf.shape[1]
        with torch.no_grad():
            negs = model.teacher(cf.reshape(b * m, *cf.shape[2:])).reshape(b, m, -1)  # (B,M,D)
        if self.mode == "hinge":
            h_pos = (pred - pos).pow(2).mean(dim=-1)  # (B,)
            h_cf = (pred.unsqueeze(1) - negs).pow(2).mean(dim=-1)  # (B, M)
            loss = h_pos.mean() + torch.relu(self.margin - h_cf).mean()
            acc = (h_pos.unsqueeze(1) < h_cf).all(dim=1).float().mean()
            return loss, {"cf_contrastive": loss.item(), "cf_acc": acc.item()}
        cand = torch.cat([pos.unsqueeze(1), negs], dim=1)  # (B, 1+M, D); positive is col 0
        if self.normalize:
            p = F.normalize(pred, dim=-1)
            c = F.normalize(cand, dim=-1)
            logits = torch.einsum("bd,bkd->bk", p, c) / self.temperature  # cosine sims
        else:
            logits = -(pred.unsqueeze(1) - cand).pow(2).mean(dim=-1) / self.temperature
        labels = logits.new_zeros(b, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == 0).float().mean()
        return loss, {"cf_contrastive": loss.item(), "cf_acc": acc.item()}
