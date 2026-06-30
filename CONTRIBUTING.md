# Contributing

## Branching
- **Core package (`src/ssa/`)** changes land via a feature branch and review
  before merging to `main`.
- **Experiments (`experiments/`)** may be committed directly.

## Style
- Run `uv run ruff check .` and `uv run ruff format .` before committing.
- Tests: `uv run pytest`. New core-package code is test-driven.
- Commit messages: imperative mood, no tool/AI attribution.

## Promotion
Experiment code (e.g. the toy env) is promoted into `src/ssa/` only once a
second experiment needs it and it has stabilized.
