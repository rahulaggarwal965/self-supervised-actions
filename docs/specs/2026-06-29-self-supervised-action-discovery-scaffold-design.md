# Self-Supervised Action Discovery — Scaffold + First Experiment

**Date:** 2026-06-29
**Status:** Approved (design)
**Scope of this cycle:** Project scaffold (core package + experiment harness) validated end-to-end by the Stage-0 synthetic toy experiment. Research/papers pipeline is deferred to its own cycle.

## Goal

Build a self-supervised latent action discovery model: encode an image history, infer a discrete latent action `a` through a bottleneck, and predict the next observation from `(context, a)`. The central problem is forcing `a` to capture *controllable change* rather than being ignored or absorbing the entire future. See `ideas/initial_ideation.md` for the full problem framing and dataset roadmap.

This cycle delivers reusable infra plus the first faithful toy proof, following the "make it work on the smallest faithful setting, then get skeptical" research process.

## Key decisions (from brainstorming)

1. **Sequencing:** Scaffold + first toy experiment together, so abstractions are validated by real use. Research pipeline deferred.
2. **Toy env:** Custom procedural 2D sprites — full control over the ground-truth hidden action set and clean evaluation.
3. **Prediction target:** Pluggable behind one interface. Run pixel-prediction first (directly renderable counterfactuals, no teacher needed); swap to latent-teacher prediction later without rearchitecting.
4. **Loss scope:** Bare minimum first — pixel-prediction + VQ only. `margin` (no-action counterfactual) and `usage` (entropy anti-collapse) are implemented and tested but **off by default**, switched on reactively when we observe action-ignoring / code collapse. Contrastive/cycle/delta/object-centric deferred to ablations. All losses live in a pluggable, ablatable registry.
5. **Framework:** Raw PyTorch + a thin reusable `Trainer` in the core package (no Lightning/Accelerate). Single local GPU + single-GPU `bench` dispatch; no near-term distributed need.

## Repo layout

```
self-supervised-actions/
├── pyproject.toml · uv.lock · .python-version · ruff config
├── src/ssa/                    # CORE PACKAGE — changes go through a branch + review
│   ├── models/                 # encoder, latent_action (inverse f), quantizer (VQ), dynamics (g), heads
│   ├── losses/                 # registry + pure loss fns (prediction, vq; margin/usage/contrastive/cycle available, off)
│   ├── data/                   # base dataset/batch contracts (TransitionBatch protocol)
│   ├── train/                  # thin Trainer (loop, AMP, ckpt, seed, wandb) + eval callbacks
│   ├── eval/                   # clustering (NMI/ARI), counterfactual render, probes (for later robot data)
│   └── utils/                  # seed, logging, viz, config glue
├── experiments/
│   └── 0_synthetic_toy/        # SANDBOX — commit freely
│       ├── README.md           # idea · hypothesis · design  (written BEFORE running)
│       ├── config/             # Hydra: config.yaml + model/ data/ loss/ groups
│       ├── env.py              # procedural 2D sprite env + Dataset (experiment-local for now)
│       ├── train.py · eval.py  # Hydra entrypoints
│       └── results/            # RESULTS.md + figures  (written AFTER running)
├── research/                   # research/papers/ skeleton + README (pipeline = its own later cycle)
├── docs/specs/                 # design specs
└── tests/                      # pytest over the core package
```

**Boundary rule:** the package defines contracts and reusable mechanism; experiments provide concrete instances and configs. The toy env starts experiment-local and is *promoted* into `ssa/data/` only once a second experiment needs it — avoids putting unstable code through review prematurely.

## Core package — modules & boundaries

`LatentActionModel` wires five config-selected, swappable parts:

- **`encoder`** — small CNN student `e_s`; optional EMA teacher `e_t` (only needed for the latent target head).
- **`latent_action`** (inverse `f`) — maps `(z_t, z_{t+1}) → pre-quant action`.
- **`quantizer`** — VQ codebook; returns quantized action + aux dict (codes, commit/codebook loss terms, usage stats / perplexity).
- **`dynamics`** (`g`) — `(context, a) → prediction feature`.
- **`head`** — pluggable prediction target. Common interface: `.target(batch)`, `.predict(feat)`, `.loss(pred, target)`.
  - `PixelDecoder`: target = next-frame pixels; prediction = decoded image (directly renderable).
  - `LatentHead`: target = `sg(e_t(I_{t+1}))`; prediction = predicted latent.

`forward` returns predictions + an aux dict of tensors (codes, pre/post-quant, perplexity, …) that losses consume.

**`losses/`** — registry of pure functions `(outputs, batch) → (scalar, logdict)`. Config lists active terms + weights. First run: `prediction` + `vq`. Implemented-but-off: `margin`, `usage`. Deferred stubs: `contrastive`, `cycle`.

**`train/Trainer`** — generic: model + dataloaders + optimizer + weighted loss list + eval callbacks + wandb logger. Knows nothing about latent actions. Owns loop / AMP / checkpoint / seed / logging.

**`eval/`** — `clustering` (NMI/ARI vs ground-truth actions), `counterfactual` (apply each code to a fixed state, decode → grid figure), `probes` (deferred, for robot data).

Each unit is independently testable: VQ math, loss shapes/signs, one Trainer step, env determinism.

## Experiment infra — Hydra + Wandb + bench

- **Hydra** composes configs from groups (`model/`, `data/`, `loss/`); `multirun` drives sweeps (codebook size K ∈ {8,16,…,256}, history length, etc.). Per-run output dir.
- **Wandb** — project `ssa`, runs grouped/tagged by experiment id; assumes `WANDB_API_KEY`/login (offline fallback). Logs losses, codebook perplexity/usage, NMI/ARI, counterfactual figures.
- **bench** — `bench submit --name toy -- uv run python experiments/0_synthetic_toy/train.py ...`; checkpoints/figures fetched back. Wandb key must exist on the remote.
- **Experiment lifecycle enforced by structure**: `README.md` (idea/hypothesis/design) committed *before* runs; `results/RESULTS.md` + figures *after*. A short template ships in the scaffold.

## First experiment — `0_synthetic_toy`

- **Env** (`env.py`): plain background, one controllable agent-sprite + a few distractor shapes, ~64×64. Hidden action set = small discrete vocabulary (move L/R/U/D, optionally rotate/push). Emits `(I_{t-k:t}, I_{t+1}, true_action)` on the fly; fixed-seed eval set. `true_action` used **only** in eval, never training.
- **Model (first run):** `a_t = VQ(f(e_s(I_t), e_s(I_{t+1})))`, `Î_{t+1} = head(g(e_s(I_t), a_t))` with `head = PixelDecoder`. Loss = pixel-prediction + VQ only.
- **Hypothesis (README):** with a tight VQ bottleneck, codes will *partly* align with hidden actions, and we expect to *observe* action-ignoring / code collapse — then enabling `margin` + `usage` measurably improves NMI and codebook perplexity. That contrast is the experiment's story.
- **Success target:** NMI(code, true_action) > 0.8 **and** a demonstrated no-action prediction gap; codebook non-collapsed (healthy perplexity); counterfactual grid shows each code producing a distinct, sensible transition.

## Tooling & workflow

- **uv + pyproject** (`ssa` package, dev dependency group). Core deps: `torch`, `hydra-core`, `wandb`, `numpy`, `pillow`, `matplotlib`, `scikit-learn` (NMI/ARI + k-means baselines), `einops`, `tqdm`. Dev: `pytest`, `ruff`.
- **git**: `git init` as part of the scaffold. Convention (`CONTRIBUTING.md`): core-package edits → feature branch + review before landing on `main`; experiments commit directly. Wire a remote (for real MRs) when ready.
- **Testing**: pytest over the package; TDD for the core mechanism (quantizer, losses, trainer step, env determinism).
- **Docs**: concise. Package modules carry short docstrings; experiments carry README + RESULTS; no sprawling prose.

## Out of scope (this cycle)

- Research/papers pipeline + wiki (own cycle).
- Real datasets (BAIR, RoboNet, BridgeData, etc.) and their probes.
- Contrastive / cycle / delta / object-centric losses beyond stubs.
- Latent-teacher head wired into a full run (interface built; first run is pixel).
- Multi-GPU / distributed training.
```
