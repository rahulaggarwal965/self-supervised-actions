# Exp 10 — Ablating the bundle: what actually solved it?

**Throughline:** [9 · bundle](../9-bundle/) → **ablation** → _next: a genuinely label-free mechanism (context bottleneck / same-state contrastive)_

## Reproduce

Each run changes **one lever** from the [Exp 8](../8-invariant-hires/) base (hires, K=16, `loss=vicreg`, NMI 0.36); all 5000 steps, seed 0, online:

```bash
# supervision only            (wandb abl-sup)
uv run python train.py model=minimal_invariant_hires loss=sup_only
# decorrelation only          (wandb abl-decorr)
uv run python train.py model=minimal_invariant_hires loss=decorr_only
# codebook shrink only        (wandb abl-k6)
uv run python train.py model=minimal_invariant_hires model.num_codes=6 loss=vicreg
# pure self-supervised bundle (wandb abl-ssl) — K=6 + decorrelation, NO labels
uv run python train.py model=minimal_invariant_hires model.num_codes=6 loss=decorr_only
```

## Results

| # | Config | K | added | **NMI(code,action)** | NMI(code,position) |
|---|---|---|---|---|---|
| 8 | base (hires) | 16 | — | 0.364 | 0.044 |
| — | K↓ only | 6 | — | 0.381 | — |
| — | decorrelation only | 16 | decorr | 0.295 | — |
| — | **pure SSL** (K↓ + decorr, **no labels**) | 6 | decorr | **0.262** | **0.017** |
| — | supervision only | 16 | 2.5% labels | **0.719** | — |
| 9 | **full bundle** | 6 | 2.5% labels + decorr | **0.942** | 0.012 |

## Interpretation — the 2.5% labels carried it; pure SSL is stuck

- **Supervision is the whole story.** 2.5% labels alone take 0.36 → **0.72**; with the small codebook the anchoring propagates cleanly to **0.94**. Everything above the Exp-8 baseline traces to the labels.
- **The label-free levers do not break the ceiling.** K↓ alone is ~neutral (0.38); decorrelation alone slightly *hurts* (0.30); and their combination **without labels is 0.26** — *below* the Exp-8 baseline. **Pure self-supervised action discovery is not achieved** (stuck at the ~0.26–0.36 hires ceiling).
- **Why decorrelation isn't enough (the key diagnostic).** In the pure-SSL run, decorrelation *did* strip position out of the code — `NMI(code,position)` fell to **0.017** — but `NMI(code,action)` stayed **0.26**. So the code became "not position" without becoming "the action." Removing position does not, by itself, make the code capture the action, because **the unsupervised objective never makes the action necessary** (the next latent is predictable from the current one; the action is a ~1% nudge — see [Exp 8](../8-invariant-hires/)). Supervision works precisely because it tells the code to *be* the action directly.

## Conclusion → next (the real, label-free problem)

Exp 9's NMI 0.94 is a legitimate **semi-supervised** result, but the project's novelty is *true* self-supervision, and this ablation shows we are not there. The bottleneck is not position-invariance (solved) and not the codebook or a decorrelation penalty — it is that **nothing in the label-free objective forces the code to carry the action**. The fix must make the action *necessary for prediction*:

1. **Context bottleneck** — throttle `z_t` into the dynamics (compress / noise / low-dim) so the next latent *cannot* be predicted from the current state alone, forcing the code to supply the displacement. Purest minimal-assumption route (works on plain `(s, s')` pairs).
2. **Same-state / different-action contrastive** (C-SWM-style) — `dynamics(z_t, code)` must match the true next-latent better than next-latents from the *same state under other actions*. Predicting from `z_t` alone cannot win → the action is necessary by construction. Label-free (the toy can branch a state into its possible next-states without naming the action); stronger data assumption (counterfactual/resettable env).

See [`research/`](../../../../research/) §B ("making the action necessary") and [RESULTS.md](../RESULTS.md).
