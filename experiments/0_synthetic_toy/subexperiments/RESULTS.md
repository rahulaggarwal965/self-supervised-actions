# Results — Stage-0 Synthetic Toy

Procedural sprite toy (one red agent moving L/R/U/D + static distractors), K=16, 5000 steps/run on `bench`. Metrics regenerated from each fetched checkpoint on the held-out val set (seed+1). This file is the **synthesis**; each subexperiment below has its own lab-notebook entry (config to reproduce · hypothesis · metrics+figures · interpretation · conclusion→next).

## The throughline (one lever at a time)

| # | Subexperiment | Lever added | Outcome | NMI |
|---|---|---|---|---|
| 0 | [baseline](0-baseline/) | pixel + VQ | codebook collapse, action ignored | 0.00 |
| 1 | [+margin+usage](1-full-losses/) | no-action margin + usage entropy | collapse fixed (7 codes), still not semantic | 0.003 |
| 2 | [+delta](2-delta/) | residual pixel prediction | gap→0; pixel target lets action be ignored | 0.008 |
| 3 | [+latent](3-latent/) | V-JEPA latent target | **representational collapse** (z_std≈0) | 0.00 |
| 4 | [+VICReg](4-vicreg/) | variance/covariance anti-collapse | mechanism healthy; obstacle now semantic | 0.011 |
| 5 | [+delta-code](5-delta-code/) | code from Δz only | codes encode **position**, not action (confirmed) | 0.013 |
| 6 | [fixed-start **control**](6-fixed-start/) | pin agent position | **action discovered** once position decoupled | **0.618** |
| 7 | [+invariant inverse](7-invariant/) | action from feature-diff + global avg pool | translation-invariant *head* insufficient — codes still track position | 0.027 |
| 8 | [+higher-res features](8-invariant-hires/) | read action from 16×16 (not 4×4) map | **resolution was a real obstacle** — NMI ~13×, position NMI drops | **0.364** |

(Exp 7 sanity: the same invariant model on the fixed-start setting reaches NMI **0.648** ≥ the control — the inverse is sound; the random-position gap is unchanged because the *encoder*, not just the head, carries position. Exp 8: at 16×16 the agent's 6-px move is resolvable; NMI(code,position) finally drops 0.067→0.044 while NMI(code,action) rises 0.027→0.364 — partial success, still < the 0.5 bar.)

## Headline

| Metric (val) | Baseline | +marg+use | +delta | +latent | +latent+VICReg | +delta-code | Target |
|---|---|---|---|---|---|---|---|
| codes / perplexity | 1 / 1.0 | 7 / 5.6 | 10 / 9.1 | 1 / 1.0 | 16 / 15.7 | 16 / 15.6 | non-collapsed |
| encoder `z_std` | — | — | — | 0.007 ⚠ | 1.01 ✓ | 1.02 ✓ | not ~0 |
| no-action gap | 3e-6 | 2e-4 | ~0 | 4.9e-3* | 3.6e-3 | 2.6e-3 | real, > 0 |
| NMI(code, action) | 0.00 | 0.003 | 0.008 | 0.00 | 0.011 | **0.013** | > 0.8 |

*mirage (collapsed target). **The mechanism is now fully healthy — no collapse, full codebook, a real no-action gap — yet NMI stays ~0.01.** Action discovery is not achieved; the obstacle is now purely *semantic*.

## The arc (what each lever fixed)

1. **Codebook collapse** → usage loss (1 → 7 codes).
2. **Action not necessary under pixel/delta MSE** → only the latent target makes it matter (delta pixel prediction did *not* help — hypothesis falsified).
3. **Representation collapse under latent prediction** → VICReg (`z_std` 0.007 → 1.01); first non-degenerate run, real gap, 16/16 codes.
4. **Code might encode absolute state** → fed the inverse model the latent change `Δz = z_{t+1}−z_t` only (`delta_input`). Result: NMI essentially unchanged (0.011 → 0.013).

![delta-code codebook usage (16/16 used)](5-delta-code/codebook_usage.png)
![delta-code code-action confusion (used, still not action-aligned)](5-delta-code/code_action_confusion.png)

## Interpretation — the remaining obstacle is semantic, not mechanistic

Every mechanistic failure has been fixed and confirmed: collapse (both kinds), and action necessity. But the codes still carry ~no information about the true L/R/U/D action, even when the code is a pure function of `Δz`. The most likely reason:

> **The encoder's latent space is not position-invariant**, so the latent change `Δz` entangles *where* the agent is with *which direction* it moved. The 16 VQ codes then capture the dominant (position-dependent) variance of `Δz` rather than the four clean directions — so NMI(code, action) ≈ 0 while the code still helps prediction (real gap).

In other words, "the action" is not a clean, low-dimensional, position-invariant quantity in this learned latent space, so a bottlenecked code over `Δz` does not recover it.

**This is now confirmed, not conjectured.** On the delta-code checkpoint, bucketing the agent's position into a 4×4 grid:

> `NMI(code, position-bucket) = 0.064`  vs  `NMI(code, action) = 0.013`

The codes align ~5× more strongly with *where the agent is* than with *which way it moved* (both are low — the codes are diffuse — but position clearly dominates action).

## Positive control: decouple position → the method discovers actions

To test the diagnosis directly, I pinned the agent's start position (same spot every sample, distractors kept so VICReg stays well-posed) and re-ran the full pipeline (latent + VICReg + delta-code). Position is now constant, so the only thing varying across transitions is the action.

| Metric (val) | random start | **fixed start (control)** |
|---|---|---|
| **NMI(code, action)** | 0.013 | **0.618** |
| ARI(code, action) | ~0 | **0.390** |
| no-action gap | 2.6e-3 | **0.030** (action → 42% lower error) |
| z_std / codes used | 1.02 / 16 | 1.02 / 16 |

NMI jumped **~46×** (0.013 → 0.62), and the confusion matrix shows each code is now action-selective (one action per code, ~4 codes per action):

![fixed-start code-action confusion](6-fixed-start/code_action_confusion.png)

**This confirms both the diagnosis and the method.** When position is decoupled from the action, the pipeline (latent prediction + VICReg anti-collapse + a delta-conditioned code) genuinely discovers the four actions. The failure on the random-position toy was the position confound, not a broken mechanism.

### Seeing the actions in pixels (decoder probe)

Latent-head runs have no decoder, so "what does each code *do*?" is invisible in pixel space. To make it visible without touching the trained model, I train a small post-hoc **decoder probe** `D(z) → pixels` on the *frozen* encoder (reconstructs next frames from their latents), then decode each code's predicted next-latent `D(dynamics(z_t, code_k))` into an actual frame. This is a faithful read-out of the learned dynamics, not a second model fit to the task.

![fixed-start decoded counterfactual: each code moves the agent a distinct direction](6-fixed-start/decoded_counterfactual.png)

From a single `I_t`, each of the 16 codes decodes to a distinct predicted next position of the red agent (reconstructions are soft — a 500-step probe on frozen latents — but the displacement direction is clearly code-dependent, with the ~4-codes-per-action grouping visible). This is the NMI 0.62 result rendered in pixels: the codes carry the action. (Logged live to wandb as `counterfactual/decoded` for every latent run.)

## Attempt at position-invariance (Exp 7) — head alone is not enough

The natural fix is to make action inference position-invariant. [Exp 7](7-invariant/) infers the action from the inter-frame **feature-map difference** through a global average pool, which is translation-invariant *by construction* (unit-tested exactly). It **did not** close the gap: on the random-position toy `NMI(code, action)` rose only 0.013 → 0.027, and `NMI(code, position)` stayed at **0.067 ≈ 0.064** — the codes still track position. The fixed-start sanity run reached NMI 0.65 (≥ control), so the inverse is sound.

The lesson: **invariance in the action head ≠ position-invariance in practice.** The pool is invariant to *clean spatial shifts of the feature map*, but the agent at different absolute positions does not produce shifted-copy feature maps in a generic 4×4-downsampled CNN, and the 6-px move is *sub-feature-cell*. Position lives in the **encoder**, so it must be fixed there.

[Exp 8](8-invariant-hires/) acted on the sub-cell half of that diagnosis: read the action from a **16×16** map instead of 4×4. This **worked, partially** — `NMI(code, action)` jumped 0.027 → **0.364** (~13×) and, for the first time, `NMI(code, position)` **dropped** (0.067 → 0.044). Resolution was a genuine obstacle. But pooling caps out below the 0.5 bar: global-average-pool discards the spatial *peak* that encodes the displacement, and the strided CNN remains only approximately shift-equivariant.

## Where this leaves us

The recipe **works once position is decoupled** (control: NMI 0.62); on the random-position toy, higher-resolution action features get us a third of the way (NMI 0.36) and confirm the resolution + position-entanglement diagnosis. A `research/` literature review surveys the approaches; the ranked next moves (cheapest first):

1. **Anti-aliased downsampling** (BlurPool / APS) so the feature-difference is a clean translated delta — directly attacks the residual aliasing Exp 8 leaves on the table.
2. **Explicit displacement readout** — soft-argmax cross-correlation / phase correlation between the two feature maps → a continuous, sub-pixel (Δx, Δy) that is translation-equivariant *by construction*; quantize that. The literature's predicted ceiling-raiser over pooling.
3. **Near-free objective/bottleneck levers** — shrink the codebook (K=16 → ~6, à la Genie), a **selectivity / decorrelate-code-from-position** loss (reuses our VICReg covariance machinery), or ~2.5% **action supervision** (free in our synthetic toy; LAOM reports +4.2×).
4. **Object-centric / agent-slot encoder** (C-SWM-style additive `z+Δz`, or slots → pick-agent-slot) — the principled, heavier build; position-invariance enforced by the encoder.

See [`research/`](../../../../research/) for the surveyed evidence behind each.

Verdict vs. Stage-0 (NMI > 0.8): **not yet met; mechanism validated, position-invariance partially addressed.** The control proves the method discovers actions (0.62) when position is decoupled; Exp 7→8 show that resolution matters (0.027→0.36) but a pooled head caps out, pointing to an explicit displacement readout or an object-centric encoder next.
