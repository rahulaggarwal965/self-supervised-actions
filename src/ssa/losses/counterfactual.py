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
    """

    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature = temperature

    def __call__(self, out, batch, model):
        pred = out.pred  # (B, D) — predicted next latent, dynamics(z_t, a_q)
        pos = out.target.detach()  # (B, D) — teacher(observed next)
        cf = batch.next_cf  # (B, M, C, H, W)
        b, m = cf.shape[0], cf.shape[1]
        with torch.no_grad():
            negs = model.teacher(cf.reshape(b * m, *cf.shape[2:])).reshape(b, m, -1)  # (B,M,D)
        d_pos = (pred - pos).pow(2).mean(dim=-1, keepdim=True)  # (B, 1)
        d_neg = (pred.unsqueeze(1) - negs).pow(2).mean(dim=-1)  # (B, M)
        logits = -torch.cat([d_pos, d_neg], dim=1) / self.temperature  # (B, 1+M)
        labels = logits.new_zeros(b, dtype=torch.long)  # positive is column 0
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == 0).float().mean()
        return loss, {"cf_contrastive": loss.item(), "cf_acc": acc.item()}
