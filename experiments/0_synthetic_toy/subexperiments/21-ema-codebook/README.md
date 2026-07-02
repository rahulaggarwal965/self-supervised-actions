# Exp 21 — EMA codebook (stabilization attempt → negative result)

**Throughline:** [20 · transition baseline](../20-transition-baseline/) → **EMA codebook** → _negative: EMA transitions LATER and with more variance; the gradient codebook's fast adaptation is what makes the transition early. EMA is out._

## What this is

Subexp 19 identified the discovery as a sharp, seed-dependent VQ phase transition. The canonical
stabilizer is an **EMA codebook** (van den Oord): update codebook vectors by exponential moving
average toward assigned encodings instead of the hard gradient `codebook_loss`. Tested here as a
drop-in quantizer (`models/quantizer_ema.py`) against the baseline, `pixel_cf_allact`, `step=20`, `K=6`.

## Findings

**1. Per-batch dead-code reset COLLAPSES discovery.** With `reset_dead=True` (reset codes unused in a
batch to a random encoding), NMI **0.007 / 0.044 / 0.076** across seeds. With 6 codes for 4 actions,
~2 codes are unused each batch and get reset *every step* → the codebook never settles → NMI rises
(~0.66 by step 1000) then collapses. (Fix: `reset_dead=false`.)

**2. Pure EMA is stable but WORSE than the baseline** (4 seeds each, 8000 steps, unique names):

| | final NMI | transition step |
|---|---|---|
| baseline (gradient codebook) | **0.908 ± 0.022** (0.892–0.946) | **~1720–1940** (mean 2220) |
| EMA (`reset_dead=false`) | 0.805 ± 0.089 (0.683–0.892) | 2040–5640 (mean **4225**) |

EMA transitions **later** (mean 4225 vs 2220) and with **more variance** (sd 0.089 vs 0.022), and lower
mean NMI.

## Interpretation

The mechanism is the opposite of the hypothesis. The transition happens when the codebook aligns with
the drifting pre-quantization `a_pre`; the **gradient** codebook adapts *fast* (direct gradient), so it
aligns early. EMA's smoothing *slows* that alignment, delaying the transition and letting seed noise
matter more. Fast codebook adaptation is beneficial here — EMA removes it.

## Conclusion → next

**EMA does not stabilize the transition — it destabilizes it.** Negative result; the gradient codebook
is better. Combined with [subexp 20](../20-transition-baseline/) (the baseline is already robust: NMI
0.91±0.02, reliable ~1800-step transition), the transition fragility is much milder than feared for the
best config, and is not the pressing problem it seemed. A mechanistically-consistent lever if we revisit
this would be *faster* adaptation early (higher LR / LR-warmup), not smoothing.
