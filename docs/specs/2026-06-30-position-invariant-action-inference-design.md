# Position-Invariant Action Inference — Design

**Date:** 2026-06-30
**Status:** approved (brainstorm), pre-implementation
**Experiment target:** `experiments/0_synthetic_toy` (Stage-0), new subexperiment `7-invariant`

## Context / problem

The Stage-0 toy walks one lever at a time toward self-supervised action discovery
(see `experiments/0_synthetic_toy/subexperiments/RESULTS.md`). By Exp 4 (+VICReg) the
mechanism is fully healthy — no collapse, full codebook (16/16), `z_std ≈ 1`, a real
no-action gap — yet `NMI(code, action) ≈ 0.01`. Exp 5 isolated the cause: the discovered
codes track **position**, not the action.

> `NMI(code, position) = 0.064` vs `NMI(code, action) = 0.013`

Exp 6 confirmed it with a positive control — pinning the agent's start position lifts
`NMI(code, action)` to **0.62** (ARI 0.39, a large real no-action gap). The decoded
counterfactuals make it visual: under random positions all codes decode to nearly the
same diffuse frame; under the fixed-start control each code moves the agent a distinct
direction.

**Root cause.** In a generic CNN encoder the global embedding `z` encodes absolute
position, so the latent change produced by "move left" depends on *where* the agent is.
A VQ code over that change therefore captures position-dependent variance rather than the
four clean directions.

## Goal & success criteria

Make "moved left" produce the **same code regardless of absolute position**, turning the
general (random-position) toy into the already-working control.

- **Primary:** `NMI(code, action)` on the random-position toy rises from ~0.013 toward the
  control — **target ≥ 0.5**, stretch the Stage-0 goal **> 0.8** — with a real no-action
  gap and a full codebook.
- **Diagnostic:** `NMI(code, position)` should *drop* (codes stop tracking position).
  Report both, as in Exp 5.
- **Mechanism intact:** `z_std ≈ 1`, 16/16 codes used (no regression on the fixes already
  banked).

## Approach (chosen)

**A — Translation-invariant action head.** Keep the entire working pipeline (latent
prediction + EMA teacher + VICReg + VQ) and change **only how the action is inferred**, so
the code is position-invariant by construction. Minimal blast radius; targets exactly the
diagnosed confound; keeps the discovery framing (we do not tell the model "actions are
translations").

Alternatives considered and deferred:
- **B — Object-centric slot encoder** (slot attention; action = relative change of the
  agent slot). The principled, general answer that scales to real scenes, but substantially
  more code and finicky to train. The natural follow-up if A plateaus.
- **C — Explicit flow / cross-correlation** between consecutive feature maps. Very direct
  for this toy, but hard-codes "actions are translations" and will not transfer to
  non-translational actions — less faithful to *discovering* the action.

## Architecture & components

The change is confined to the action-inference (inverse) path. Dynamics, the prediction
target, the EMA teacher, VICReg, and the VQ bottleneck are unchanged.

1. **Encoder exposes its spatial feature map.** `Encoder` already ends its conv stack in a
   `(C, H', W')` map before reducing to the global vector `z` (dim 256). Expose both:
   - `features(I) -> (B, C, H', W')` — the conv-stack output (translation-equivariant).
   - `forward(I) -> z` — the existing pooled/flattened global vector, **unchanged**, still
     used for the prediction target, dynamics input, and VICReg.

2. **Translation-invariant inverse model.** Infer the action from the inter-frame feature
   difference rather than from a concat/difference of global vectors:
   - `D = features(I_{t+1}) − features(I_t)`  →  `(B, C, H', W')`
   - small conv head (1–2 conv layers + nonlinearity) over `D`  →  motion feature map
   - **global average pool over the spatial dims**  →  `(B, C')` (translation-invariant)
   - linear  →  `a_pre` (action_dim)  →  VQ (unchanged) → code `a`

   Rationale: a left-move's *local* difference signature (old position fades, new position
   appears one step left) is identical wherever it occurs; the conv extracts that local
   signature and the global pool discards absolute location. This **supersedes** the Exp-5
   `delta_input` trick — it is the spatial analog done at the feature-map level.

3. **Dynamics + prediction: unchanged.** `dynamics(z_t, a) -> ẑ_{t+1}` predicting the
   EMA-teacher's global latent of `I_{t+1}` (`LatentHead`). `z_t` still carries position,
   `a` is now pure direction → dynamics composes them to predict the shifted next latent.

### Data flow

```
I_t, I_{t+1}
   ├─ encoder.features ─→ D = feat(I_{t+1}) − feat(I_t) ─→ conv ─→ GAP ─→ a_pre ─→ VQ ─→ a
   ├─ z_t   = encoder(I_t)            (position-carrying global latent)
   └─ z*    = teacher(I_{t+1})        (EMA-teacher target latent)
ẑ_{t+1} = dynamics(z_t, a)
loss = ‖ẑ_{t+1} − z*‖²  +  vq  +  margin  +  usage  +  vicreg(z_t)
```

## Configuration

New model config `config/model/minimal_invariant.yaml`: same as `minimal_latent`
(`LatentHead`, `teacher_momentum: 0.99`) but with the invariant inverse selected. The
invariant inverse is exposed either as a flag on `InverseModel` or a sibling component with
the same interface (`(z_t-or-features, ...) -> a_pre`); the implementation plan picks the
cleaner option. `delta_input` is not used with the invariant inverse (superseded).

Runs (5000 steps, seed 0, online wandb, on `bench`):
- **Main:** `model=minimal_invariant loss=vicreg` on the random-position toy (2 distractors).
- **Sanity:** the same model on the fixed-start setting (`+data.env.start=[29,29]`) — should
  remain ≥ the control.

Comparisons: Exp 5 random-start (NMI 0.013) and Exp 6 fixed-start control (0.62).

## Testing

- **Core property — translation invariance:** build the invariant inverse; feed a
  frame-pair `(I_t, I_{t+1})` and a spatially-shifted copy of the same pair; assert the
  inferred `a_pre` (and the selected code) are ~identical within tolerance. This is the
  defining property of the design.
- **Shapes / wiring:** `encoder.features(I)` returns a 4-D spatial map; the full model
  `forward` runs end-to-end and the VQ operates on the pooled vector; output fields match
  the existing `ModelOutput` contract.
- **Smoke:** a 1-step train on tiny synthetic data (existing test pattern).

## Files

- `src/ssa/models/encoder.py` — add `features()` (spatial map); keep `forward()` pooled.
- `src/ssa/models/inverse.py` (or a small new module) — translation-invariant inverse
  (conv + GAP) sharing the inverse interface.
- `experiments/0_synthetic_toy/config/model/minimal_invariant.yaml` — new model config.
- `tests/` — invariance property test + shape/smoke tests.
- After the run: `experiments/0_synthetic_toy/subexperiments/7-invariant/` (README +
  config.yaml + metrics.json + figures incl. decoded counterfactual), and a throughline
  row + synthesis update in `RESULTS.md`.

## Out of scope / future

- Object-centric / slot encoder (Approach B) — the next build if A plateaus below the >0.8
  target.
- Sweeps (K, VICReg/margin weights, longer training) toward >0.8.
- Harder toys (multiple agents, non-translational actions) — later stages.
