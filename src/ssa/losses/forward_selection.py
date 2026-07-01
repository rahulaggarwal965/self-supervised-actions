import torch
import torch.nn.functional as F


class ForwardSelectionLoss:
    """Contrastive over codes — make the forward model *need* the action.

    For each transition, the code the inverse assigned must make the dynamics
    predict the true next latent better than any *other* code would. Implemented
    as a cross-entropy over codes: logits are the negative distance between each
    code's predicted next latent ``head.predict(dynamics(z_t, code_k))`` and the
    (stop-grad) target, and the label is the assigned code.

    This forces ``dynamics(z_t, code)`` to actually depend on the code (fixing the
    action-agnostic dynamics, where swapping codes barely moved the prediction)
    and pushes the code to carry the prediction-relevant change. Label-free;
    pair with a decorrelate-from-position term so that change is the *action*, not
    position.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature = temperature

    def __call__(self, out, batch, model):
        z = out.z_ctx
        target = out.target.detach()  # teacher latent of the true next frame
        codebook = model.quantizer.codebook.weight  # (K, A)
        k, b = codebook.shape[0], z.shape[0]
        per_code = []
        for i in range(k):
            code = codebook[i : i + 1].expand(b, -1)
            per_code.append(model.head.predict(model.dynamics(z, code)))
        preds = torch.stack(per_code, dim=1)  # (B, K, D)
        dist = (preds - target.unsqueeze(1)).pow(2).mean(dim=-1)  # (B, K)
        logits = -dist / self.temperature
        assigned = out.vq["codes"]  # (B,) long — the code the inverse chose
        loss = F.cross_entropy(logits, assigned)
        acc = (logits.argmax(dim=1) == assigned).float().mean()
        return loss, {"fwd_sel": loss.item(), "fwd_sel_acc": acc.item()}
