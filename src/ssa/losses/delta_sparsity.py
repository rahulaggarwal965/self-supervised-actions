class DeltaSparsityLoss:
    """L1 penalty on the predicted frame-change (delta head only).

    The true change from one step is *sparse* — only the small agent region differs;
    the static distractors and background do not. But a coarse decoder tends to emit a
    low-amplitude change everywhere (the faint cyan "ghost" squares at distractor/old
    positions that survive into ``I_t + Δ``). Penalizing ``mean|Δ|`` pushes the decoder
    to predict exactly zero change wherever nothing moves, leaving only the genuine
    agent displacement — so the reconstructed next frame keeps the distractors clean.
    """

    def __call__(self, out, batch, model):
        if not getattr(model.head, "delta", False):
            loss = out.pred.new_zeros(())
            return loss, {"delta_l1": 0.0}
        loss = out.pred.abs().mean()
        return loss, {"delta_l1": loss.item()}
