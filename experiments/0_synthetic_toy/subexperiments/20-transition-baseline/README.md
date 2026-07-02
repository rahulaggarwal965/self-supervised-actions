# Exp 20 — Transition-fragility baseline (the best config is actually robust)

**Throughline:** [19 · training dynamics](../19-training-dynamics/) → **seed sweep of the best config** → _`pixel_cf_allact` transitions early (~1800) and reliably; NMI 0.91±0.02 across seeds. The severe fragility was `pixel_clean`-specific, not fundamental._

## What this is

Subexp 19 found learning is a sharp, seed-dependent VQ phase transition and flagged fragility as the
main open. This quantifies it: a seed sweep of the **best discovery config** — pixel-delta head +
`pixel_cf_allact` (contrastive + all-action), additive dynamics, `step=20`, `K=6`, 8000 steps — measuring
final NMI and the transition step (first `cf_acc>0.95`) per seed. All on the bench, wandb online.

## Findings

**1. The transition is early and reliable.** 4-seed comparison run (`cmp-base-s0..3`):

| seed | 0 | 1 | 2 | 3 | summary |
|---|---|---|---|---|---|
| final NMI | 0.898 | 0.946 | 0.892 | 0.896 | **mean 0.908, sd 0.022** |
| transition step | 1720 | 1940 | 1720 | 3500 | mostly ~1800; all 4 transitioned |

Tight NMI (sd 0.022) and reliable, early transitions. Across **all 8 baseline seeds run** (this sweep +
an earlier batch), 7 landed ~0.89–0.95; one gave 0.666 (a partial transition) — so the occasional low
seed exists but is the exception.

**2. The severe fragility was config-specific.** Subexp 19's 5300-step transition was on `pixel_clean`
(which adds `delta_sparsity`); that term delays the transition. Without it, `pixel_cf_allact` transitions
~3× earlier and reliably. So the fragility is not fundamental to the method — it depends on the loss mix.

## Interpretation / conclusion

For the best config, **label-free discovery is robust** (NMI 0.91±0.02, reliable ~1800-step transition),
so the training-dynamics fragility is much milder than feared — the grokking jump still happens, but early
and consistently. The residual concern is the occasional low-NMI seed. [Subexp 21](../21-ema-codebook/)
then shows the natural stabilizer (EMA codebook) does **not** help (it transitions later, with more
variance) — the gradient codebook's fast adaptation is what gives the early transition. Net: stabilization
is not the pressing problem it appeared to be; the discovery result stands.
