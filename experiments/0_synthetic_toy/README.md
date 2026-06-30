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
