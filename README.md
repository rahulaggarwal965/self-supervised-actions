# self-supervised-actions

Self-supervised latent action discovery from video. Encode an image history,
infer a discrete latent action through a bottleneck, and predict the next
observation — forcing the action to capture controllable change.

## Layout
- `src/ssa/` — core package (MR-reviewed): models, losses, training, eval.
- `experiments/` — sandbox experiments; each has a README (idea/hypothesis),
  Hydra config, and a `results/` doc with figures.
- `research/` — papers wiki/database (pipeline is a later cycle).

## Setup
```bash
uv sync
uv run pytest
```

## Run the first experiment
```bash
uv run python experiments/0_synthetic_toy/train.py
```

See `ideas/initial_ideation.md` for the full research framing and dataset roadmap.
