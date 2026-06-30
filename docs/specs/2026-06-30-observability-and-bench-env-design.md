# Experiment Observability + bench env-forwarding

**Date:** 2026-06-30
**Status:** Approved (design)
**Goal:** Make the synthetic-toy training run fully observable on Weights & Biases (rich metrics + visualizations + held-out val performance for debugging), default to online logging for real runs, and enable live online wandb on remote `bench` jobs by adding env-var forwarding to the `bench` dispatcher. Then launch the first real (baseline) run and break the oversized PR #1 into reviewable chunks.

Spans two repos: `self-supervised-actions` (instrumentation + config) and `gpu-dispatch` (`bench --env`).

## Decisions (from brainstorming)

1. **Observability = "debugging dashboard"** (metrics + the four visualizations below + held-out val perf).
2. **wandb default `online`** for real runs; smoke/verification runs pass `wandb.mode=offline`; tests use `NoopLogger` (`disabled`).
3. **bench gets `--env` forwarding** so remote jobs get `WANDB_API_KEY` for live online logging (user pre-authorized gpu-dispatch changes).
4. **Instrument first, then run.**
5. **PR #1 → 4 stacked scaffold PRs + a 5th PR for the instrumentation.**
6. The only external secret the run needs is wandb (`RVA_NSR_WANDB_API_KEY` → `WANDB_API_KEY`). git push = SSH; `gh` = `GH_TOKEN`.

## Part A — Instrumentation architecture (`ssa` package + experiment)

Generic mechanism in the package; experiment wires it. The experiment stays wandb-agnostic (it hands the logger plain matplotlib figures).

**Package (`src/ssa/`, reviewed):**
- `train/logging.py` — add `log_figures(figures: dict[str, Figure], step: int)` to the `Logger` protocol and both impls. `WandbLogger` wraps each figure as `wandb.Image`, logs it, and closes the figure (no leak). `NoopLogger` no-ops.
- `eval/metrics.py` (new) — `no_action_gap(model, batch) -> dict` returning `{action_err, noaction_err, gap}` where `action_err = MSE(g(c, a), target)`, `noaction_err = MSE(g(c, 0), target)`, `gap = noaction_err - action_err`. Model-agnostic; works even though the margin *loss* is off.
- `eval/figures.py` (new) — pure figure builders returning `matplotlib.figure.Figure`:
  - `reconstruction_panel(model, batch, n=8)` — `n` rows of `[I_t, true I_{t+1}, predicted I_{t+1}]`.
  - `counterfactual_figure(model, obs)` — labeled wrapper over the existing `counterfactual_grid` (I_t + one decoded panel per code).
  - `code_action_confusion(codes, labels, num_codes)` — code×action count heatmap.
  - `codebook_usage_bar(codes, num_codes)` — per-code usage bar.
- `train/trainer.py` — two changes: (1) include `train/grad_norm` in `train_step` logs (capture the value from `clip_grad_norm_` when `grad_clip` is set); (2) change the eval-hook contract to `eval_fn(model, step) -> (scalars: dict, figures: dict[str, Figure])`; the Trainer logs scalars via `logger.log` and figures via `logger.log_figures`.

**Experiment (`experiments/0_synthetic_toy/`):**
- `train.py` — rewrite `make_eval_fn` to, over the held-out val set (`eval_ds`, seed+1): accumulate `codes`/`labels` and compute val scalars; compute `no_action_gap` on a batch; and build the four figures on a *fixed* viz batch (first `n` val samples, so panels are comparable across steps). Returns `(scalars, figures)`.

## Part B — Dashboard contents

- **Train scalars** (every `log_every`): `loss/total`, `prediction/pred`, `vq/vq`, `vq/perplexity`, `train/grad_norm`.
- **Val scalars** (every `eval_every`, held-out set): `val/mse`, `val/nmi`, `val/ari`, `val/perplexity`, `val/noaction_gap` (+ `val/action_err`, `val/noaction_err`).
- **Val figures** (every `eval_every`, fixed viz batch): `recon/panel`, `counterfactual/grid`, `codes/confusion`, `codes/usage`.

These target the experiment's thesis directly: perplexity/usage → collapse; NMI/ARI + confusion → action alignment; no-action gap → whether the action matters.

## Part C — wandb online/offline

- `config/config.yaml`: default `wandb.mode: online`.
- Smoke/verification commands pass `wandb.mode=offline`; tests construct `NoopLogger` (`disabled`).
- Documented in the experiment README.

## Part D — bench `--env` forwarding (`gpu-dispatch`)

Add a repeatable `--env NAME` flag to `bench submit`/`bench run`:
- At submit time, read `NAME` from the local environment (error clearly if unset).
- Transmit the value to the remote **without exposing it on the command line or in the job record**: write the `NAME=VALUE` pairs to a `0600` env file, rsync it into the remote job dir, and have the remote launch wrapper `set -a; . <envfile>; set +a; rm -f <envfile>` before running the user command. The env file must be excluded from the fetched `--out` globs.
- Follow gpu-dispatch's existing structure/spec+plan conventions; add tests (the value is set in the job env, never appears in the stored command/job metadata, file is 0600 and removed).

This is implemented as its own change in `gpu-dispatch` (its own commit/tests there), tracked by this spec.

## Part E — Rollout sequence

1. Build `bench --env` in gpu-dispatch (TDD + two-stage review).
2. Build the instrumentation in `ssa` + experiment (TDD + two-stage review); smoke-verify offline.
3. Launch the **baseline** run (prediction + vq only) on `bench`, online, in the background. Run through the zsh chain so the secret is sourced from `RVA_NSR_WANDB_API_KEY` and never appears in my output, with `bench --env WANDB_API_KEY` forwarding it to the remote:
   `zsh -ic 'WANDB_API_KEY=$RVA_NSR_WANDB_API_KEY bench submit --name toy-baseline --env WANDB_API_KEY --out "outputs/**" -- uv run python experiments/0_synthetic_toy/train.py wandb.mode=online'`
   then `bench wait <id>` as a background task. The baseline is expected to *show* action-ignoring / collapse — that contrast is the experiment.
4. Close PR #1 → open 4 stacked scaffold PRs (infra → models → losses+trainer+eval → experiment+docs) + a 5th PR for the instrumentation.
5. Monitor the run; write `experiments/0_synthetic_toy/results/RESULTS.md` with metrics + figures.

## Out of scope (this cycle)

- `loss/full.yaml` (margin + usage ablation) — the *next* cycle, after observing the baseline failures.
- "Heavier diagnostics" (nearest-neighbor action retrievals, embedding projections, per-code montages).
- Any wandb logging beyond the four figures + listed scalars.
- Multi-GPU/distributed; non-toy datasets.
