# Exp 11 — Forward-selection (label-free): fixes the dynamics, collapses the codebook

**Throughline:** [10 · ablation](../10-ablation/) → **forward-selection (label-free)** → _next: prevent collapse — same-state-counterfactual negatives (real futures) or stronger anti-collapse_

## Reproduce

Trained 5000 steps on `bench`, seed 0, wandb online (`ssl-fwdsel`), **no action labels**:

```bash
uv run python train.py model=minimal_invariant_hires model.num_codes=6 loss=ssl
```

`loss=ssl` = vicreg + decorrelate-code-from-position + **ForwardSelectionLoss**. The
forward-selection term is a cross-entropy over codes: the code the inverse assigned must
make `dynamics(z_t, code)` predict the true next latent better than any other code. Goal:
force the dynamics to *use* the code (fix the action-agnostic forward model from Exp 9),
label-free.

## Hypothesis

Exp 10 showed the label-free objective never makes the action necessary (dynamics ignores
the code — 0.6% effect; pure-SSL NMI 0.26). Forcing the dynamics to select the right code
should make the code carry the prediction-relevant change; with decorrelation keeping
position out, that change should be the action → label-free NMI ↑ **and** a real gap.

## Results

| Metric (val, random-position, label-free) | pure-SSL (Exp 10) | **Exp 11 (+ forward-selection)** |
|---|---|---|
| **no-action gap** | 3.8e-3 | **0.541** (action_err 0.072 vs noaction 0.613) |
| NMI(code, action) | 0.262 | **0.003** |
| perplexity / codes used | 5.4 / — | **1.18 / 2** (`[0,0,0,886,138,0]`) |
| z_std | 1.02 | 1.03 |
| fwd_sel_acc | — | 0.85 |

## Interpretation — right idea, degenerate optimum

Two things happened, and they explain each other:

- **Forward-selection fixed the dynamics — decisively.** The no-action gap jumped from
  3.8e-3 to **0.54** (the action now cuts prediction error ~8.5×; the code-effect went from
  ~1% to dominant). The "world model ignores the action" problem from Exp 9 is solved: the
  dynamics genuinely acts on the code.
- **…but by collapsing the codebook to ~1 code.** `perplexity 1.18`, only 2 of 6 codes ever
  fire (one dominant). NMI(code, action) → 0 because a single code can't distinguish four
  actions. The code-contrastive has a **degenerate optimum**: assign (almost) everything to
  *one* code and make the dynamics lean on it hard — the unused codes predict badly, so the
  assigned code trivially "wins" (`fwd_sel_acc` 0.85 with ~1 code). The usage-entropy term
  (weight 0.1) was too weak to resist the forward-selection + prediction pull.

The lesson: forcing the dynamics to use *a* code is not the same as forcing it to use *the
right decomposition* of codes. The code-contrastive rewards "one strong code," not "four
action-codes," because its negatives are the model's *own* predictions under other codes —
which it can freely make bad.

## Conclusion → next

1. **Same-state / different-action contrastive (real negatives).** Replace the model's-own-
   predictions negatives with the **actual next-states under other actions** (the toy can
   roll them out): `dynamics(z_t, code)` must predict the observed next over the *real*
   counterfactual futures. One code **cannot** predict four genuinely-different futures, so
   collapse is structurally impossible — this is the robust version of Exp 11. (The env
   rollout is a data-generation change, still label-free.)
2. **Cheap probe first:** crank the usage-entropy weight (and/or lower the forward-selection
   weight / temperature) to see whether anti-collapse pressure alone lets forward-selection
   decompose into ≥4 action-codes.

See [`research/`](../../../../research/) §B and [RESULTS.md](../RESULTS.md).
