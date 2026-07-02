import torch
import torch.nn.functional as F


class AllActionPredictionLoss:
    """Directly supervise the dynamics on the real next frame for EVERY action.

    The plain prediction loss only regresses ``dynamics(z_t, a_q)`` against the one
    OBSERVED transition, so the code-conditioned predictions for the other actions get
    no target and the forward model collapses to the mean outcome (a symmetric blur).
    Here we use the same-state counterfactuals as supervision: for each real future of
    this state (the observed next plus ``batch.next_cf``), we infer its own code from the
    (I_t, future) pair and regress the code-conditioned prediction onto that future. So
    every code learns its action's true displacement — a clean, distinct counterfactual.

    Label-free: the code is discovered by the inverse; we never use action labels — only
    that these frames are the same state's outcomes under (unknown) different actions.
    """

    def __call__(self, out, batch, model):
        it = batch.obs[:, -1]  # (B, C, H, W)
        futures = torch.cat([batch.next_obs.unsqueeze(1), batch.next_cf], dim=1)  # (B, A, C,H,W)
        a = futures.shape[1]
        z = out.z_ctx  # (B, D) — reuse the context latent (encodes I_t)
        lvl = getattr(model.inverse, "feat_level", None)
        spatial = getattr(model.inverse, "spatial", False)
        f_t = model.encoder.features(it, lvl) if spatial else None

        total = 0.0
        for i in range(a):
            fut = futures[:, i]  # (B, C, H, W)
            # infer this action's code from the (I_t, future) pair, same path as forward()
            if spatial:
                a_pre = model.inverse(f_t, model.encoder.features(fut, lvl))
            else:
                a_pre = model.inverse(z, model.encoder(fut))
            a_q, _ = model.quantizer(a_pre)
            pred = model.head.predict(model.dynamics(z, a_q), it)
            target = fut - it if getattr(model.head, "delta", False) else fut
            total = total + F.mse_loss(pred, target)
        loss = total / a
        return loss, {"all_action_pred": loss.item()}
