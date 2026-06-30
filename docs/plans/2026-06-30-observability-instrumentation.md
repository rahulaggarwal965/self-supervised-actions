# Experiment Observability Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a wandb "debugging dashboard" to the synthetic-toy experiment — rich train + held-out-val scalars and four diagnostic figures logged at eval cadence — and default wandb to online for real runs.

**Architecture:** Generic, wandb-agnostic mechanism in the `ssa` package (a `log_figures` logger method; pure metric + matplotlib-figure builders in `ssa/eval`; a Trainer that logs grad-norm and routes an eval callback's `(scalars, figures)`); the experiment's `train.py` wires it over the existing held-out val set. The experiment hands the logger plain matplotlib figures; only `WandbLogger` knows about `wandb.Image`.

**Tech Stack:** PyTorch, matplotlib (Agg), wandb, Hydra, pytest, ruff.

**Conventions:** TDD; commit per task; plain commit messages (imperative, no AI/Co-Authored-By attribution). Tests are package code → `tests/`. NOTE: this sandbox blocks network/syscalls — run every Bash command (incl. git, pytest, training runs) with `dangerouslyDisableSandbox: true`. `ModelOutput` fields: `pred, target, a_pre, a_q, z_ctx, feat, vq`. `vq` dict keys: `codes, dist, commit_loss, codebook_loss, perplexity`.

---

## Task 1: `Logger.log_figures`

**Files:**
- Modify: `src/ssa/train/logging.py`
- Test: `tests/test_logging.py`

- [ ] **Step 1: Write the failing test** — add to `tests/test_logging.py`:
```python
def test_noop_logger_accepts_log_figures():
    from ssa.train.logging import NoopLogger

    logger = NoopLogger()
    logger.log_figures({"panel": object()}, step=0)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logging.py -v`
Expected: FAIL (`NoopLogger has no attribute log_figures`).

- [ ] **Step 3: Implement** — update `src/ssa/train/logging.py` to its full new form:
```python
from typing import Protocol


class Logger(Protocol):
    def log(self, metrics: dict, step: int) -> None: ...
    def log_figures(self, figures: dict, step: int) -> None: ...
    def finish(self) -> None: ...


class NoopLogger:
    """Logger that drops everything; used for tests and disabled runs."""

    def log(self, metrics: dict, step: int) -> None:
        pass

    def log_figures(self, figures: dict, step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger:
    """Thin wrapper over ``wandb.init``/``run.log``."""

    def __init__(self, project: str, config: dict | None = None, **kwargs) -> None:
        import wandb

        self.run = wandb.init(project=project, config=config, **kwargs)

    def log(self, metrics: dict, step: int) -> None:
        self.run.log(metrics, step=step)

    def log_figures(self, figures: dict, step: int) -> None:
        import matplotlib.pyplot as plt
        import wandb

        self.run.log({k: wandb.Image(v) for k, v in figures.items()}, step=step)
        for fig in figures.values():
            plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logging.py -v`
Expected: PASS. Then `uv run pytest -q` → still green.

- [ ] **Step 5: Commit**
```bash
git add src/ssa/train/logging.py tests/test_logging.py
git commit -m "feat: add log_figures to logger"
```

---

## Task 2: `no_action_gap` metric

**Files:**
- Create: `src/ssa/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test** — `tests/test_metrics.py`:
```python
import torch

from ssa.data.batch import TransitionBatch
from ssa.eval.metrics import no_action_gap
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import PixelDecoder
from ssa.models.inverse import InverseModel
from ssa.models.model import LatentActionModel
from ssa.models.quantizer import VectorQuantizer


def _model():
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )


def test_no_action_gap_returns_three_floats_with_gap_identity():
    batch = TransitionBatch(
        obs=torch.rand(4, 1, 3, 64, 64),
        next_obs=torch.rand(4, 3, 64, 64),
        action=torch.zeros(4, dtype=torch.long),
    )
    m = no_action_gap(_model(), batch)
    assert set(m) == {"action_err", "noaction_err", "gap"}
    assert all(isinstance(v, float) for v in m.values())
    assert abs(m["gap"] - (m["noaction_err"] - m["action_err"])) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/ssa/eval/metrics.py`:
```python
import torch
import torch.nn.functional as F


@torch.no_grad()
def no_action_gap(model, batch) -> dict:
    """Prediction error with the inferred action vs. with a zero action.

    Returns ``{action_err, noaction_err, gap}`` where ``gap = noaction_err -
    action_err``; a positive gap means the action improves prediction. A
    diagnostic (works even when the margin loss is disabled)."""
    out = model(batch)
    action_err = F.mse_loss(out.pred, out.target)
    zero = torch.zeros_like(out.a_q)
    pred0 = model.head.predict(model.dynamics(out.z_ctx, zero))
    noaction_err = F.mse_loss(pred0, out.target)
    return {
        "action_err": float(action_err),
        "noaction_err": float(noaction_err),
        "gap": float(noaction_err - action_err),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/ssa/eval/metrics.py tests/test_metrics.py
git commit -m "feat: add no-action prediction gap metric"
```

---

## Task 3: Diagnostic figure builders

**Files:**
- Create: `src/ssa/eval/figures.py`
- Test: `tests/test_figures.py`

- [ ] **Step 1: Write the failing test** — `tests/test_figures.py`:
```python
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
import torch  # noqa: E402

from ssa.data.batch import TransitionBatch  # noqa: E402
from ssa.eval.figures import (  # noqa: E402
    code_action_confusion,
    codebook_usage_bar,
    counterfactual_figure,
    reconstruction_panel,
)
from ssa.models.dynamics import Dynamics  # noqa: E402
from ssa.models.encoder import Encoder  # noqa: E402
from ssa.models.heads import PixelDecoder  # noqa: E402
from ssa.models.inverse import InverseModel  # noqa: E402
from ssa.models.model import LatentActionModel  # noqa: E402
from ssa.models.quantizer import VectorQuantizer  # noqa: E402


def _model(num_codes=6):
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=num_codes, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )


def _batch(b=4):
    return TransitionBatch(
        obs=torch.rand(b, 1, 3, 64, 64),
        next_obs=torch.rand(b, 3, 64, 64),
        action=torch.zeros(b, dtype=torch.long),
    )


def test_reconstruction_panel_returns_figure():
    assert isinstance(reconstruction_panel(_model(), _batch(), n=3), Figure)


def test_counterfactual_figure_returns_figure():
    fig = counterfactual_figure(_model(num_codes=6), _batch().obs[0])
    assert isinstance(fig, Figure)


def test_confusion_and_usage_return_figures():
    codes = [0, 0, 1, 2, 2, 2]
    labels = [0, 0, 1, 1, 2, 2]
    assert isinstance(code_action_confusion(codes, labels, num_codes=6), Figure)
    assert isinstance(codebook_usage_bar(codes, num_codes=6), Figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_figures.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/ssa/eval/figures.py`:
```python
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from ssa.eval.counterfactual import counterfactual_grid


def _show(ax, img: torch.Tensor) -> None:
    ax.imshow(img.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    ax.axis("off")


@torch.no_grad()
def reconstruction_panel(model, batch, n: int = 8) -> Figure:
    """Rows of [I_t, true I_{t+1}, predicted I_{t+1}] (PixelDecoder head only)."""
    out = model(batch)
    n = min(n, batch.obs.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n), squeeze=False)
    titles = ["I_t", "true I_{t+1}", "pred"]
    for i in range(n):
        imgs = [batch.obs[i, -1], batch.next_obs[i], out.pred[i]]
        for j, img in enumerate(imgs):
            _show(axes[i][j], img)
            if i == 0:
                axes[i][j].set_title(titles[j])
    fig.tight_layout()
    return fig


@torch.no_grad()
def counterfactual_figure(model, obs) -> Figure:
    """I_t plus one decoded panel per codebook entry (PixelDecoder head only).

    ``obs``: ``(T, C, H, W)`` history for one example."""
    grid = counterfactual_grid(model, obs)  # (K, C, H, W)
    k = grid.shape[0]
    fig, axes = plt.subplots(1, k + 1, figsize=(2 * (k + 1), 2), squeeze=False)
    row = axes[0]
    _show(row[0], obs[-1])
    row[0].set_title("I_t")
    for i in range(k):
        _show(row[i + 1], grid[i])
        row[i + 1].set_title(f"code {i}")
    fig.tight_layout()
    return fig


def code_action_confusion(codes, labels, num_codes: int) -> Figure:
    """Heatmap of discovered code (rows) vs. ground-truth action (cols)."""
    codes = np.asarray([int(c) for c in codes])
    labels = np.asarray([int(x) for x in labels])
    num_actions = int(labels.max()) + 1 if labels.size else 1
    mat = np.zeros((num_codes, num_actions), dtype=int)
    for c, x in zip(codes, labels):
        mat[c, x] += 1
    fig, ax = plt.subplots(figsize=(2 + 0.5 * num_actions, 2 + 0.35 * num_codes))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xlabel("true action")
    ax.set_ylabel("code")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def codebook_usage_bar(codes, num_codes: int) -> Figure:
    """Per-code usage count over the eval set (collapse diagnostic)."""
    counts = np.bincount(np.asarray([int(c) for c in codes]), minlength=num_codes)
    fig, ax = plt.subplots(figsize=(2 + 0.3 * num_codes, 2))
    ax.bar(range(num_codes), counts[:num_codes])
    ax.set_xlabel("code")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_figures.py -v`
Expected: PASS (4 tests). Then `uv run pytest -q` → still green.

- [ ] **Step 5: Commit**
```bash
git add src/ssa/eval/figures.py tests/test_figures.py
git commit -m "feat: add diagnostic figure builders"
```

---

## Task 4: Trainer — grad-norm logging + `(scalars, figures)` eval contract

**Files:**
- Modify: `src/ssa/train/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Update the existing fit test + add a grad-norm test** in `tests/test_trainer.py`.

Replace `test_fit_runs_loop_with_eval_logging_and_restores_train_mode` with this version (eval_fn now returns `(scalars, figures)`; `RecordingLogger` gains `log_figures`):
```python
def test_fit_runs_loop_with_eval_logging_and_restores_train_mode():
    torch.manual_seed(0)
    model = _model()
    batch = TransitionBatch(
        obs=torch.rand(2, 1, 3, 64, 64),
        next_obs=torch.rand(2, 3, 64, 64),
        action=torch.zeros(2, dtype=torch.long),
    )
    loader = [batch, batch]
    terms = [LossTerm("prediction", PredictionLoss(), 1.0), LossTerm("vq", VQLoss(), 1.0)]

    calls = {"n": 0}

    def eval_fn(m, step):
        calls["n"] += 1
        return {"eval/dummy": 0.0}, {}

    class RecordingLogger:
        def __init__(self):
            self.steps = []
            self.figure_calls = 0

        def log(self, metrics, step):
            self.steps.append(step)

        def log_figures(self, figures, step):
            self.figure_calls += 1

        def finish(self):
            pass

    logger = RecordingLogger()
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        terms,
        logger=logger,
        grad_clip=1.0,
    )
    trainer.fit(loader, max_steps=6, eval_every=2, eval_fn=eval_fn, log_every=2)

    assert calls["n"] == 2
    assert logger.figure_calls == 2
    assert model.training is True
    assert 0 in logger.steps
```

Add a grad-norm test:
```python
def test_train_step_logs_grad_norm_when_clipping():
    torch.manual_seed(0)
    model = _model()
    batch = TransitionBatch(
        obs=torch.rand(4, 1, 3, 64, 64),
        next_obs=torch.rand(4, 3, 64, 64),
        action=torch.zeros(4, dtype=torch.long),
    )
    terms = [LossTerm("prediction", PredictionLoss(), 1.0), LossTerm("vq", VQLoss(), 1.0)]
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), terms, grad_clip=1.0)
    logs = trainer.train_step(batch)
    assert "train/grad_norm" in logs and logs["train/grad_norm"] >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: FAIL (`eval_fn` return is a tuple now → old code logged a dict; `train/grad_norm` missing; `log_figures` attribute).

- [ ] **Step 3: Implement** — in `src/ssa/train/trainer.py`, update `train_step` (grad-norm) and `fit` (eval contract). The two methods become:
```python
    def train_step(self, batch) -> dict:
        batch = batch.to(self.device)
        out = self.model(batch)
        logs: dict = {}
        total = 0.0
        for term in self.loss_terms:
            val, log = term.fn(out, batch, self.model)
            total = total + term.weight * val
            logs.update({f"{term.name}/{k}": v for k, v in log.items()})
        self.optimizer.zero_grad()
        total.backward()
        if self.grad_clip:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            logs["train/grad_norm"] = float(norm)
        self.optimizer.step()
        self.model.update_teacher()
        logs["loss/total"] = total.item()
        return logs

    def fit(self, loader, max_steps, eval_every=0, eval_fn=None, log_every=10) -> None:
        self.model.train()
        step = 0
        while step < max_steps:
            for batch in loader:
                logs = self.train_step(batch)
                if self.logger and step % log_every == 0:
                    self.logger.log(logs, step)
                if eval_fn and eval_every and step > 0 and step % eval_every == 0:
                    self.model.eval()
                    scalars, figures = eval_fn(self.model, step)
                    if self.logger:
                        self.logger.log(scalars, step)
                        self.logger.log_figures(figures, step)
                    self.model.train()
                step += 1
                if step >= max_steps:
                    break
```
(`__init__` and `save` are unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: PASS (overfit test + updated fit test + grad-norm test). Then `uv run pytest -q` → all green.

- [ ] **Step 5: Commit**
```bash
git add src/ssa/train/trainer.py tests/test_trainer.py
git commit -m "feat: log grad norm and route eval figures in Trainer"
```

---

## Task 5: Wire the dashboard into the experiment + default online

**Files:**
- Modify: `experiments/0_synthetic_toy/train.py`
- Modify: `experiments/0_synthetic_toy/config/config.yaml`

No unit test (experiment wiring); verification is an offline smoke run.

- [ ] **Step 1: Default wandb to online** — in `experiments/0_synthetic_toy/config/config.yaml`, change the `wandb.mode` line:
```yaml
wandb:
  project: ssa
  mode: online   # real runs log live; smoke/verification runs pass wandb.mode=offline; tests use disabled
```

- [ ] **Step 2: Rewrite the eval callback + force Agg + thread num_codes** in `experiments/0_synthetic_toy/train.py`.

(a) At the very top of the file (before any matplotlib/pyplot import is triggered transitively), add:
```python
import matplotlib

matplotlib.use("Agg")
```
(b) Add the new eval imports near the other `ssa` imports:
```python
from ssa.eval.figures import (
    code_action_confusion,
    codebook_usage_bar,
    counterfactual_figure,
    reconstruction_panel,
)
from ssa.eval.metrics import no_action_gap
```
(c) Replace `make_eval_fn` with the dashboard version:
```python
def make_eval_fn(eval_loader, device, num_codes, viz_n=8):
    import numpy as np

    viz_batch = next(iter(eval_loader))  # fixed batch so figures are comparable across steps

    def eval_fn(model, step):
        codes, labels = [], []
        mse_sum, count = 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                b = batch.to(device)
                out = model(b)
                codes += out.vq["codes"].tolist()
                labels += b.action.tolist()
                mse_sum += torch.nn.functional.mse_loss(out.pred, out.target, reduction="sum").item()
                count += out.pred.numel()
        cluster = nmi_ari(codes, labels)
        counts = np.bincount(np.asarray(codes), minlength=num_codes).astype(float)
        probs = counts / counts.sum()
        perplexity = float(np.exp(-(probs * np.log(probs + 1e-10)).sum()))
        vb = viz_batch.to(device)
        gap = no_action_gap(model, vb)
        scalars = {
            "val/nmi": cluster["nmi"],
            "val/ari": cluster["ari"],
            "val/mse": mse_sum / count,
            "val/perplexity": perplexity,
            "val/action_err": gap["action_err"],
            "val/noaction_err": gap["noaction_err"],
            "val/noaction_gap": gap["gap"],
        }
        figures = {
            "recon/panel": reconstruction_panel(model, vb, n=viz_n),
            "counterfactual/grid": counterfactual_figure(model, vb.obs[0]),
            "codes/confusion": code_action_confusion(codes, labels, num_codes),
            "codes/usage": codebook_usage_bar(codes, num_codes),
        }
        return scalars, figures

    return eval_fn
```
(d) Update the `make_eval_fn(...)` call in `main` to pass `num_codes`:
```python
    trainer.fit(
        train_loader, max_steps=cfg.train.max_steps, eval_every=cfg.train.eval_every,
        eval_fn=make_eval_fn(eval_loader, device, cfg.model.num_codes),
        log_every=cfg.train.log_every,
    )
```

- [ ] **Step 3: Offline smoke-verify the dashboard end-to-end**

Run (sandbox disabled):
```bash
WANDB_MODE=offline uv run python experiments/0_synthetic_toy/train.py \
  train.max_steps=10 train.eval_every=5 data.train_size=256 data.eval_size=128 \
  data.num_workers=0 train.device=cpu wandb.mode=offline
```
Expected: exit 0. At step 5 the eval runs and `WandbLogger.log_figures` logs the four figures into the offline `wandb/` dir without error — this is what validates `wandb.Image(fig)` works. Confirm a `wandb/` run dir was created (it logs scalars + media). Then run `uv run pytest -q` → all green, and `uv run ruff check . && uv run ruff format --check .` → clean. Clean up `model.pt`, `outputs/`, `wandb/` afterward (all gitignored).

If the run errors inside `log_figures` (some wandb versions reject a raw Figure), change `WandbLogger.log_figures` to log the figures directly (`self.run.log(figures, step=step)` — wandb auto-converts matplotlib Figures) and re-verify; report the change.

- [ ] **Step 4: Commit**
```bash
git add experiments/0_synthetic_toy/train.py experiments/0_synthetic_toy/config/config.yaml
git commit -m "feat(exp): wandb debugging dashboard + default online"
```

---

## Final verification

- [ ] `uv run pytest -q` → all pass.
- [ ] `uv run ruff check . && uv run ruff format --check .` → clean.
- [ ] Offline smoke run completes and writes a `wandb/` dir with the four figures (validates the image path). Artifacts cleaned up.

## Self-Review Notes
- **Spec coverage:** `log_figures` (Task 1), `no_action_gap` (Task 2), the four figures (Task 3), grad-norm + `(scalars, figures)` eval contract (Task 4), experiment wiring + val scalars + figures + default online (Task 5). Matches spec Parts A–C.
- **Contract change:** the eval-hook signature changes from `-> dict` to `-> (scalars, figures)`; the only in-repo caller is the experiment `make_eval_fn` (Task 5) and the Trainer fit test (updated in Task 4). No other callers.
- **wandb.Image(fig):** validated by the offline smoke run (Task 5 step 3); fallback noted if a wandb version rejects raw Figures.
- **Out of scope:** `loss/full.yaml` (margin/usage ablation), heavier diagnostics — deferred per spec.
