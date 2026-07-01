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
| 9 | [+bundle](9-bundle/) | K↓ (16→6) + 2.5% action supervision + decorrelate-from-position | **action discovered (semi-supervised)** — position NMI →0.012 | **0.942** |
| 10 | [ablation](10-ablation/) | isolate each bundle lever | **the 2.5% labels carried it**; pure-SSL = 0.26 (below base) | 0.26–0.94 |
| 11 | [+forward-selection](11-forward-selection/) | make dynamics *need* the code (label-free) | **fixes the dynamics (gap 3.8e-3→0.54) but collapses codebook to ~1** | 0.003 |

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

[Exp 9](9-bundle/) then combined three of the review's levers on the hires base — **codebook K=16→6, ~2.5% action supervision, and a decorrelate-code-from-position term**. This **met the Stage-0 target**: `NMI(code, action)` **0.942**, ARI **0.915**, with `NMI(code, position)` collapsed to **0.012** and a near-diagonal confusion matrix. The random-position toy is solved for *action discovery* — past both the >0.8 target and the 0.62 control.

The [ablation](10-ablation/) (Exp 10) then attributed that 0.94 cleanly: **the 2.5% labels carried it.** Supervision alone takes 0.36→0.72 (→0.94 with the small codebook); the label-free levers do not break the ceiling — K↓ alone ~0.38, decorrelation alone ~0.30, and the **pure-SSL** combination (K↓ + decorrelation, no labels) is **0.26**, *below* the Exp-8 baseline. Tellingly, that pure-SSL run drove `NMI(code,position)` to 0.017 (decorrelation *did* strip position) yet `NMI(code,action)` stayed 0.26 — removing position does not make the code capture the action.

## Where this leaves us

Stage-0's NMI > 0.8 is met **only semi-supervised** (Exp 9, 2.5% labels). The project's novelty — *true* self-supervision — is **not yet achieved**: pure-SSL is stuck at ~0.26–0.36. The ablation localizes why, and it is the same fundamental obstacle throughout: **nothing in the label-free objective forces the code to carry the action** (the next latent is predictable from the current state; the action is a ~1% nudge). Position-invariance is solved (Exp 8 resolution + decorrelation), and it wasn't sufficient.

The label-free fix must make the action *necessary for prediction*. [Exp 11](11-forward-selection/) tried the cheapest version — a **forward-selection** term (the assigned code must make the dynamics predict the true next best, contrasted against the model's *own* predictions under other codes). It **fixed the dynamics decisively** (no-action gap 3.8e-3 → **0.54**) but **collapsed the codebook to ~1 code** (NMI → 0): a single "apply the change" code trivially wins the code-contrastive, so the code space collapses instead of decomposing into 4 actions. Forcing the dynamics to use *a* code ≠ forcing the *right decomposition*.

That points to the robust version:

1. **Same-state / different-action contrastive (real negatives).** `dynamics(z_t, code)` must predict the observed next over the *actual* next-states under other actions (the toy rolls them out; still label-free). One code cannot predict four genuinely-different futures, so collapse is structurally impossible — the fix for Exp 11. **Leading next experiment.**
2. **Cheap probe first:** crank usage-entropy / lower forward-selection weight to see if anti-collapse pressure alone lets forward-selection decompose into ≥4 codes.
3. **Context bottleneck** — throttle `z_t` into the dynamics so the next can't be predicted from the current state alone. Complementary, minimal-assumption.

Other surveyed routes (anti-aliased downsampling, object-centric encoder) remain for harder stages. See [`research/`](../../../research/) §B.

Verdict vs. Stage-0 (NMI > 0.8): **met semi-supervised (0.94); pure self-supervised is the open problem (~0.26–0.36).** The arc: position control (Exp 6, 0.62) → resolution matters (Exp 8, 0.36) → 0.94 with 2.5% labels (Exp 9) → ablation: the labels carried it (Exp 10) → forward-selection makes the dynamics act on the code but collapses the codebook (Exp 11). The genuinely-label-free path is now a same-state-counterfactual contrastive that structurally prevents collapse.
