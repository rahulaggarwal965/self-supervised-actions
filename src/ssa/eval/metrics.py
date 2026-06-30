import torch
import torch.nn.functional as F


@torch.no_grad()
def no_action_gap(model, batch) -> dict:
    """Prediction error with the inferred action vs. with a zero action.

    Returns ``{action_err, noaction_err, gap}`` where ``gap = noaction_err -
    action_err``; a positive gap means the action improves prediction. A
    diagnostic (works even when the margin loss is disabled)."""
    out = model(batch)
    action_err = F.mse_loss(out.pred, out.target)
    zero = torch.zeros_like(out.a_q)
    pred0 = model.head.predict(model.dynamics(out.z_ctx, zero))
    noaction_err = F.mse_loss(pred0, out.target)
    return {
        "action_err": float(action_err),
        "noaction_err": float(noaction_err),
        "gap": float(noaction_err - action_err),
    }
