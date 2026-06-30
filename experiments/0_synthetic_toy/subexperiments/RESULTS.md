# Results — Stage-0 Synthetic Toy

Two runs on the procedural sprite toy (one red agent moving L/R/U/D + static distractors), 5000 steps each, default `model/minimal` (K=16), on `bench` (alpa12 gpu1), logged live to wandb (project `ssa`). Figures/metrics below are regenerated from each fetched checkpoint on the held-out val set (seed+1).

| Run | wandb | loss |
|---|---|---|
| Baseline | [jzj52ot1](https://wandb.ai/rva_nsr/ssa/runs/jzj52ot1) | prediction + VQ |
| + margin + usage | [1saxa1eg](https://wandb.ai/rva_nsr/ssa/runs/1saxa1eg) | + no-action margin (m=0.002) + usage entropy (w=0.1) |

## Headline

| Metric (val) | Baseline | + margin + usage | Target |
|---|---|---|---|
| codes used / perplexity | 1 / **1.00** | 7 / **5.60** | non-collapsed |
| no-action gap | 3.2e-6 | **2.1e-4** | clearly > 0 |
| NMI(code, true action) | 0.00 | **0.003** | > 0.8 |
| ARI(code, true action) | 0.00 | −0.001 | high |
| val pixel MSE | 0.0083 | 0.0096 | — |

**The anti-collapse losses do their jobs, but the codes still are not the actions.** Usage broke the codebook collapse (1→7 codes) and the margin made the action weakly necessary (gap went positive, ~65×), but the discovered codes carry no information about the true L/R/U/D action (NMI ~0, far from the 0.8 target). Partial mechanism win; semantic action discovery **not** achieved on this toy as configured.

## Baseline (prediction + VQ): total failure — as predicted

One code carries every assignment (perplexity 1.0), and applying any of the 16 codes to a fixed frame yields identical predictions — the action is ignored.

![baseline codebook usage](codebook_usage.png)
![baseline counterfactual](counterfactual.png)

The toy's future is largely predictable without the action (`I_{t+1} ≈ I_t`; a blurry near-static prediction already gets low MSE), so the dynamics has no pressure to use `a_t`, and nothing stops VQ collapse.

## + margin + usage: collapse broken, action weakly necessary, but no semantic alignment

- **Collapse broken** — 7 of 16 codes now used (perplexity 5.6). The usage loss works.
- **Action weakly necessary** — no-action gap is positive (0.00021): predicting with the inferred action beats the zero-action prediction, where the baseline showed no difference. The margin loss works directionally, but the margin it can extract is tiny.
- **No action alignment** — NMI ≈ 0. The confusion matrix shows the used codes (dominated by codes 12–13) spread roughly **uniformly across all four actions** — no code is action-selective:

![ablation code-action confusion](full_losses/code_action_confusion.png)
![ablation codebook usage](full_losses/codebook_usage.png)

## Interpretation

Usage and margin are necessary but not sufficient here. Two things keep the codes from capturing the action:

1. **Weak action leverage.** Because `I_{t+1} ≈ I_t` on this toy, the action's marginal contribution to pixel MSE is tiny (the gap tops out at ~2e-4), so there is little gradient pressure for the codes to become action-selective rather than encoding scene/position. The inverse model `f(z_t, z_{t+1})` can route appearance information into the code instead of the controllable change.
2. **No pressure toward "change, not appearance."** Nothing yet forces `a_t` to describe the *delta* rather than the frame.

This matches the design doc's warnings, and points the next iterations at making `a_t` capture controllable change:

- **Predict the delta** (`ẑ_{t+1} = z_t + g(c_t, a_t)` / pixel delta) so the action explains change, not the static scene.
- **Raise the action's leverage** — a low-bandwidth context bottleneck `c_t = h(z_{t-k:t})`, and/or a harder future, so prediction genuinely needs `a_t`.
- **Stronger / relative margin** and a **contrastive action loss** (similar deltas → similar codes) to push semantic structure.
- Tune `m` and the usage weight; sweep K and history length.

Verdict vs. the Stage-0 success criterion (NMI > 0.8 + clear no-action gap): **not met yet** — the mechanism is half-working (anti-collapse + necessity), and the codes need change-focused structure before they become semantic. That is the next cycle.
