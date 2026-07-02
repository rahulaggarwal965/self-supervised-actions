class AlphaSparsityLoss:
    """L1 penalty on the compositing head's alpha mask (CompositePixelDecoder only).

    The next frame is ``α·F + (1-α)·I_t``. Without pressure, ``α`` could go to 1 everywhere
    (ignoring ``I_t`` and repainting the whole scene, which reintroduces artifacts). Penalizing
    ``mean(α)`` keeps the mask localized to the region that actually changes (the moved agent),
    so the static scene is copied verbatim from ``I_t`` — clean distractors, no ghosts.
    """

    def __call__(self, out, batch, model):
        # recompute the observed-action prediction so head._alpha is the current forward mask
        model.head.predict(out.feat, batch.obs[:, -1])
        alpha = getattr(model.head, "_alpha", None)
        if alpha is None:
            zero = out.pred.new_zeros(())
            return zero, {"alpha_l1": 0.0}
        loss = alpha.mean()
        return loss, {"alpha_l1": loss.item()}
