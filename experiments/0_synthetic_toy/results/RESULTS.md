# Results — Stage-0 Synthetic Toy

Procedural sprite toy (one red agent moving L/R/U/D + static distractors), K=16, 5000 steps/run on `bench`. Metrics/figures regenerated from each fetched checkpoint on the held-out val set (seed+1). Five experiments walking the design doc's levers.

## Headline

| Metric (val) | Baseline | + margin + usage | + delta | + latent | + latent + VICReg | Target |
|---|---|---|---|---|---|---|
| codes used / perplexity | 1 / 1.00 | 7 / 5.60 | 10 / 9.11 | 1 / 1.00 | **16 / 15.70** | non-collapsed |
| encoder `z_std` | — | — | — | 0.007 ⚠ | **1.01** ✓ | not ~0 |
| no-action gap | 3.2e-6 | 2.1e-4 | 7.6e-7 | 4.9e-3 (mirage) | **3.6e-3 (real)** | clearly > 0 |
| NMI(code, action) | 0.00 | 0.003 | 0.008 | 0.00 | **0.011** | > 0.8 |
| pred MSE | 0.0083 (px) | 0.0096 (px) | 0.0117 (Δ) | 0.0004 (lat†) | 0.037 (lat) | — |

† collapsed — trivially low. **Progress:** the latent + VICReg run is the first to (a) keep the representation healthy (`z_std≈1`), (b) use the whole codebook (16/16), and (c) make the action *genuinely* necessary (a real positive no-action gap, ~14% lower error with the action). **Still missing:** NMI ~0 — the codes are useful but encode the wrong thing.

## The arc

- **Baseline (pred + VQ):** total collapse (1 code), action ignored. The static future makes a blurry prediction MSE-cheap.
- **+ margin + usage:** usage breaks *codebook* collapse (7 codes), but the action is only weakly necessary and codes aren't action-selective (NMI ~0).
- **+ delta:** predicting `I_{t+1}−I_t` did **not** raise the action's leverage (gap back to ~0); the agent never moves across the counterfactual. *Hypothesis falsified.*
- **+ latent (V-JEPA):** predicting teacher latents **collapsed the representation** (`z_std≈0.007`); tiny MSE and a big gap were artifacts of a near-constant target.
- **+ latent + VICReg:** variance/covariance regularization fixes the collapse (`z_std≈1.01`), all 16 codes are used, and the no-action gap is now **real** (0.0036). But NMI stays ~0.01 — see below.

![latent+VICReg codebook usage (all 16 codes used)](vicreg/codebook_usage.png)
![latent+VICReg code-action confusion (used, but not action-aligned)](vicreg/code_action_confusion.png)

## Interpretation

Each lever fixed exactly what it targets, and the experiments isolate a clean chain of obstacles:

1. **Codebook collapse** → fixed by the usage loss.
2. **Action not necessary under pixel/delta MSE** → only the latent target (which penalizes the blur) makes the action matter.
3. **Representation collapse under latent prediction** → fixed by VICReg.
4. **Remaining obstacle: the codes encode *state*, not *action*.** With a healthy representation, the inverse model `f(z_t, z_{t+1})` still routes information that lowers latent MSE — most likely the agent's *position* — into the 16 codes, because nothing forces `a_t` to describe the *change* rather than the scene. The action helps prediction (real gap), but as a position/scene code, not a direction code, so NMI vs the true L/R/U/D stays ~0.

## Next levers (priority)

1. **Make the code a function of the latent *change*.** Feed the inverse model the latent delta `Δz = z_{t+1} − z_t` (dropping raw `z_t` from the code's input) so the code *cannot* encode absolute position — only the transition. This is the most direct attack on obstacle #4 and a small change to `InverseModel`.
2. **Contrastive action loss** — pull together codes for transitions with similar `Δz` and push apart dissimilar ones, structuring the codebook by transition type (direction) directly.
3. Sweep K and the VICReg/margin weights now that the pipeline is non-degenerate.

Verdict vs. Stage-0 (NMI > 0.8 + real no-action gap): **gap now real; NMI not yet.** The pipeline is finally non-degenerate (no collapse, action necessary, full codebook), so the remaining problem — codes encoding state instead of change — is now cleanly isolated and directly addressable.
