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
| 12 | [+counterfactual contrastive](12-counterfactual/) | predict observed next over real other-action futures (label-free) | **best pure-SSL, collapse-proof (6 codes), position-free — but plateaus** | 0.381 |
| 13 | [+delta target + projection](13-delta-contrastive/) | contrast the *change* in a projected subspace (label-free) | **label-free action discovery — best single run** | 0.785 |
| 14 | [stabilization](14-stabilization/) | seed sweep · two-stage · loss rebalance (label-free) | **robust ~0.56–0.70; 0.785 was high-variance; clean rep at predw10** | 0.70 |
| 15 | [additive dynamics](15-additive-dynamics/) | additive `z+T(code)` + decoder-free diagnosis (step=20) | **fault is the forward model (distinct-but-wrong latents), not the decoder; action swamped in latent** | 0.51–0.71 |
| 16 | [pixel contrastive](16-pixel-contrastive/) | predict + contrast in **pixel** space (high-signal) | **breakthrough — first action-conditional counterfactual (distinct directional moves)** | 0.82 |
| 17 | [all-action supervision](17-all-action-supervision/) | + per-code real-frame targets for every action | **clean, distinct, correct per-code moves — discovery + counterfactual solved** | **0.89–0.95** |
| 18 | [counterfactual fidelity](18-counterfactual-fidelity/) | finer decoder · L1 sparsity · full-frame · compositing | cosmetic; L1 is a knife-edge, full-frame drops distractors, compositing = structural fix | 0.63–0.95 |
| 19 | [training dynamics](19-training-dynamics/) | diagnose curves; fix eval-logging | **eval-logging bug hid results; learning is a sharp, seed-dependent VQ phase transition (grokking)** | — |

(Exps 15–19 move to a **larger action (`step=20`)** and a **pixel-delta head**; see the "step-20 + pixel-space" update at the bottom. Exp 7 sanity: the same invariant model on the fixed-start setting reaches NMI **0.648** ≥ the control — the inverse is sound; the random-position gap is unchanged because the *encoder*, not just the head, carries position. Exp 8: at 16×16 the agent's 6-px move is resolvable; NMI(code,position) finally drops 0.067→0.044 while NMI(code,action) rises 0.027→0.364 — partial success, still < the 0.5 bar.)

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

### The label-free hunt (Exp 11–13)

Exp 9 met the target only *semi-supervised*; the project's novelty is *true* self-supervision, and the ablation localized the obstacle: **nothing in the label-free objective forced the code to carry the action** (the next latent is predictable from the current state; the action is a ~1% nudge). Position-invariance (Exp 8) was necessary but not sufficient. The fix had to make the action *necessary for prediction* — the hunt across Exp 11–13:

The label-free fix must make the action *necessary for prediction*. [Exp 11](11-forward-selection/) tried the cheapest version — a **forward-selection** term (the assigned code must make the dynamics predict the true next best, contrasted against the model's *own* predictions under other codes). It **fixed the dynamics decisively** (no-action gap 3.8e-3 → **0.54**) but **collapsed the codebook to ~1 code** (NMI → 0): a single "apply the change" code trivially wins the code-contrastive, so the code space collapses instead of decomposing into 4 actions. Forcing the dynamics to use *a* code ≠ forcing the *right decomposition*.

[Exp 12](12-counterfactual/) built the robust version — a **same-state counterfactual contrastive**: `dynamics(z_t, code)` must predict the observed next over the *real* next-states under other actions (label-free; the toy rolls them out). This **structurally prevents collapse** (a single code can't predict four different futures) and delivered the **best pure-SSL result so far — NMI 0.381**, ARI 0.25, all 6 codes balanced, position decoupled (0.032). But it plateaus below the bar: `cf_acc` saturates at 1.0 while `action_err` blows up to 0.42, i.e. the contrastive is trivially satisfied (coarsely beating counterfactuals) without forcing a clean 4-way decomposition or predicting well. A usage-crank probe on the forward-selection variant behaved similarly (no collapse, NMI 0.32).

[Exp 13](13-delta-contrastive/) broke the plateau. Two changes to the counterfactual contrastive: **contrast the *delta*** (`pred − z_t` vs the counterfactual futures minus `z_t`, cancelling the shared static scene) in a **projection head** (so a strong contrastive shapes a projected subspace instead of dragging the raw prediction off-manifold), with a high weight (15). Result — **label-free NMI 0.785, ARI 0.75**, six codes balanced, position decoupled (0.11), near-diagonal confusion. The projection is essential (delta without it → 0.02); weight matters (w8→0.71, w15→0.785).

## Where this leaves us

**Label-free action discovery works** — with the important caveat that it's a *distribution*, not a point. The delta-target counterfactual contrastive discovers the four actions with **no labels**: best single run **NMI 0.785** (Exp 13), but the seed sweep (Exp 14) shows a wide spread (0.40–0.79, mean **~0.56**), so the honest number is **robust ~0.56–0.70, ≈ the 0.62 control**, not a stable 0.8. The best *balanced* run (`predw10`, prediction-weight rebalanced) gives **NMI 0.70, position-NMI 0.03, and a clean representation** (action_err 0.30) — the cleanest label-free result. The path that mattered: latent prediction + VICReg → position-invariant inverse at sufficient resolution (Exp 8) → make the action *necessary* via same-state counterfactual contrastive (Exp 12) → contrast the **delta** in a **projected** subspace (Exp 13) → rebalance losses for a clean forward model (Exp 14).

**The residual tension.** The action's true latent effect on this toy is small (a 6-px move), so there is an inherent conflict: a strong contrastive gives high code↔action NMI but a degraded forward model (poor counterfactual), while a strong prediction gives a clean forward model but a nearly action-agnostic dynamics (tiny no-action gap). The **inverse discovers the action** (the goal); a forward model that is simultaneously accurate *and* strongly action-conditional is the harder, partially-open problem — likely relaxed on a harder toy with a larger action-effect (the natural next stage).

Verdict vs. Stage-0 (NMI > 0.8): **met semi-supervised (0.94, Exp 9); label-free action discovery achieved and robust at ~0.56–0.70 (best 0.785), past the 0.62 control — the project's novelty demonstrated, though not a stable 0.8.** The arc: position control validates the mechanism (Exp 6, 0.62) → resolution matters (Exp 8, 0.36) → 2.5% labels solve but carry it (Exp 9/10) → forward-selection collapses (Exp 11) → counterfactual contrastive plateaus (Exp 12) → delta + projection breaks it label-free (Exp 13, best 0.785) → stabilization shows it's a high-variance ~0.6 mechanism with a clean-rep option (Exp 14). Next: variance reduction (compositional/inverse-cycle constraints), then a harder toy where the action-effect is large enough to resolve the accuracy-vs-conditionality tension.

## Update — step-20 + pixel-space (Exp 15–19): the counterfactual solved

The synthesis above documents the **label-free-latent** phase (Exp 0–14), which left the *forward model /
counterfactual* as the "partially-open" problem. Exp 15–19 close it, by acting on exactly the lever that
paragraph names — the action's effect was too small.

- **The action was low-variance in *latent* space, and that was the whole problem.** A decoder-free
  diagnosis ([Exp 15](15-additive-dynamics/)) showed the forward model produces *distinct-but-wrong*
  next-latents (swap-gap 93%, cover-err 269%): its prediction error (~8) dwarfs the action's latent
  footprint (~2.9). The decoder was blameless. Additive dynamics halved the error but couldn't win in a
  space where the signal is that small.
- **Predicting in pixel space, where a 20px move is high-signal, breaks it.** A pixel-delta head + a
  **same-state contrastive in pixel space** ([Exp 16](16-pixel-contrastive/)) gives the first genuinely
  action-conditional counterfactual — distinct, directionally-correct per-code moves, NMI **0.82**. (MSE
  alone still collapses to the mean; the contrastive is load-bearing.)
- **Adding per-code real-frame targets makes it clean** ([Exp 17](17-all-action-supervision/)): contrastive
  (discriminate *which* action) + all-action supervision (render *that* action's real move) → NMI
  **0.89–0.95**, 100%-consistent per-code directions. Discovery **and** a faithful counterfactual, label-free.
- Render **fidelity** ([Exp 18](18-counterfactual-fidelity/)) is a separate decoder-structure problem
  (compositing head is the fix), not a discovery problem.
- **Two process findings** ([Exp 19](19-training-dynamics/)): an eval-logging bug made every run *look*
  static on wandb (only mid-training figures were logged — now fixed); and learning is a **sharp,
  seed-dependent VQ phase transition** (grokking + codebook reorganization) — the real remaining fragility.

Full math + dynamics: [`docs/pipeline-and-losses.md`](../../../docs/pipeline-and-losses.md). **Remaining work
is stabilizing the transition** (EMA codebook / temperature annealing / warmup), not discovery or fidelity.
