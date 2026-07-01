# Exp 12 — Same-state counterfactual contrastive (label-free): best pure-SSL, not yet solved

**Throughline:** [11 · forward-selection](../11-forward-selection/) → **counterfactual contrastive** → _next: tune (weight/temperature, prediction stability) or a stronger formulation_

## Reproduce

Trained 5000 steps on `bench`, seed 0, wandb online (`ssl-cf`), **no action labels**:

```bash
uv run python train.py model=minimal_invariant_hires model.num_codes=6 \
    loss=ssl_cf data.counterfactuals=true
```

`loss=ssl_cf` = vicreg + decorrelate-code-from-position + **CounterfactualContrastiveLoss**:
`dynamics(z_t, code)` must predict the observed next latent over the teacher encodings of
the *same state's* next frames under the **other actions** (`data.counterfactuals` rolls
these out; no action labels are used — only "which future actually happened").

## Hypothesis

Exp 11's forward-selection collapsed the codebook because its negatives were the model's
own predictions (a single code trivially wins). Real counterfactual futures as negatives
should make collapse impossible (one code can't predict four different futures) and force
the code to encode the action → label-free NMI ↑.

## Results

| Metric (val, random-position, label-free) | pure-SSL (Exp 10) | fwd-sel (Exp 11) | **Exp 12 (counterfactual)** |
|---|---|---|---|
| **NMI(code, action)** | 0.262 | 0.003 | **0.381** |
| ARI | 0.168 | ~0 | **0.252** |
| **NMI(code, position)** | 0.017 | — | **0.032** |
| codes used / perplexity | 5 | **2 / 1.18** (collapse) | **6 / 5.66** (balanced) |
| no-action gap | 3.8e-3 | 0.54 | 0.011 |
| cf / fwd accuracy | — | 0.85 | **cf_acc 1.0** |
| action_err (pred MSE) | 0.04 | — | **0.42** ⚠ |

Codes histogram: `[243, 115, 161, 255, 116, 134]` — all six used, no collapse.

## Interpretation — right mechanism, best pure-SSL result, still not the breakthrough

The structural guarantee worked: **collapse is gone** (6 balanced codes vs Exp 11's ~1),
position stays decoupled (`NMI(code,position)` 0.032), and this is the **best label-free
`NMI(code,action)` so far — 0.38** (vs 0.26 decorr-only, 0.36 Exp 8). Real counterfactual
negatives do force the code to carry *some* action information without collapsing.

But it is **not solved** (0.38 < 0.5 target < 0.62 control), and two signals say why:

- **`cf_acc` = 1.0 while `action_err` = 0.42.** The contrastive is *trivially satisfied* —
  the predicted next is always closer to the observed future than to the counterfactuals —
  yet the absolute prediction is poor (10× the usual ~0.04). The model distinguishes the
  observed future *coarsely* (enough to beat counterfactuals) without predicting it *well*
  or decomposing the action cleanly into codes. The contrastive + decorrelation appear to
  have degraded the latent-prediction representation.
- So the code carries *enough* action info to win the contrastive but not a *clean* 4-way
  decomposition — NMI 0.38, not 0.6+.

## Conclusion → next

The counterfactual contrastive is the **right, collapse-proof mechanism** and gives the
best label-free number to date, but as-configured it plateaus at 0.38. Likely levers:

1. **Tune the objective** — the contrastive weight/temperature vs the prediction term
   (right now `cf_acc` saturates at 1.0 while `action_err` blows up, i.e. it's over-weighted
   / too easy); a harder or better-calibrated contrastive (more negatives, margin, or a
   temperature schedule) may force a cleaner decomposition without wrecking prediction.
2. **Stabilize the representation** — the degraded `action_err` suggests the teacher/latent
   target needs protecting (e.g. stop-grad details, lower cf weight early, two-stage).
3. If label-free stalls here, **Stage-0 is met semi-supervised (Exp 9, 0.94)** and the honest
   framing is: pure-SSL discovers the action *partially* (0.38, position-free, no collapse);
   full label-free discovery on this toy is unresolved and is the standing research problem.

See [`research/`](../../../../research/) §B and [RESULTS.md](../RESULTS.md).
