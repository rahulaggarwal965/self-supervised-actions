# 0 — Synthetic Toy: does a VQ bottleneck discover the hidden action?

## Idea
Smallest faithful setting for self-supervised action discovery. A red agent
sprite moves L/R/U/D (the hidden action) over a plain background with static
distractors. We infer a discrete latent action through a VQ bottleneck and
predict the next frame, never seeing the true action during training.

## Hypothesis
With a tight VQ bottleneck, the discovered codes will partly align with the
four hidden actions. We expect to *observe* failure modes — the action being
ignored (history/identity shortcuts) and/or codebook collapse. Turning on the
`margin` (no-action counterfactual) and `usage` (entropy) losses should then
measurably raise NMI(code, true action) and codebook perplexity. That contrast
is the experiment.

## Design
- Model: `a = VQ(f(e(I_t), e(I_{t+1})))`, `Î_{t+1} = PixelDecoder(g(e(I_t), a))`.
- Loss (first run): pixel MSE + VQ only. Then add `margin`, then `usage`.
- Eval: NMI/ARI of codes vs true action; no-action prediction gap; codebook
  perplexity; counterfactual grid (apply each code to a fixed frame).

## Metrics (how to read the results)
The model never sees the true action, so we score the *discovered* codes against
the hidden L/R/U/D label after the fact. Each metric answers one question:

- **NMI(code, action)** — *"Does knowing the code tell you the action?"* Normalized
  mutual information between the code a transition was assigned and its true action.
  **0** = independent (the code says nothing about which way the agent moved), **1** =
  the code perfectly determines the action. This is the headline number for action
  discovery. Normalized, so it's comparable no matter how many codes are in play.
  *Example:* if code 3 fires for left-moves and only left-moves, it contributes high NMI.

- **ARI(code, action)** — *"Do the code-groups and action-groups agree?"* Adjusted
  Rand Index treats codes and actions as two clusterings of the same transitions and
  measures agreement, **adjusted so random labelings score ≈ 0** (1 = identical
  grouping). Complements NMI and is less fooled by having more codes (16) than
  actions (4).

- **codes used / perplexity** — *"Is the bottleneck actually being used?"* Perplexity is
  the *effective* number of codes (exp of the usage entropy): **1** = one code wins
  every transition (total **codebook collapse** — the bottleneck learned nothing),
  **16** (=K) = all codes used evenly. "codes used" is the raw count that ever fire.

- **no-action gap** — *"Does the action even matter for prediction?"* We predict the
  next step twice: with the real inferred code (`action_err`) and with a null / no-op
  code (`noaction_err`). **gap = noaction_err − action_err.** A gap ≈ 0 means the code
  is ignored — you predict just as well without it; a real positive gap means the code
  carries information the predictor needs. *Example:* in the fixed-start control,
  `action_err` 0.042 vs `noaction_err` 0.073 → using the action cuts error ~42%.

- **encoder z_std** — *"Did the representation collapse?"* Std of the encoder embedding
  across samples. **≈ 0** = the encoder outputs nearly the same vector for every frame
  (**representational collapse**: predicting the next latent becomes trivial and nothing
  is learned); **≈ 1** (what VICReg targets) = the representation genuinely varies. A
  collapse detector, not a quality score.

- **prediction MSE** (`val_mse` / `delta_mse` / `latent_mse`) — the raw next-step error
  the model trains on (full-frame, residual-frame, or latent target, per the head). Read
  it *together with the above*: a collapsed model reaches near-zero `latent_mse` by
  predicting a constant, so low error alone is not progress.

## Success target
NMI(code, true action) > 0.8 **and** a demonstrated no-action prediction gap;
codebook non-collapsed; counterfactual grid shows each code producing a
distinct, sensible move.

## Run
```bash
# baseline (pixel + vq)
uv run python train.py
# add anti-collapse terms once failures are observed (override loss group later)
```

## Results
Each run is logged as its own ordered subexperiment under `subexperiments/<n-name>/`
(a lab-notebook `README.md` with the exact command to reproduce, the going-in
hypothesis, metrics+figures, interpretation, and conclusion→next). The synthesis
across all of them is `subexperiments/RESULTS.md`.
