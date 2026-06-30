# Scaffold + Synthetic Toy Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `ssa` core package + Hydra/Wandb experiment harness, validated end-to-end by a Stage-0 synthetic-toy latent-action experiment (pixel-prediction + VQ).

**Architecture:** `LatentActionModel` wires five swappable parts — encoder, inverse model `f`, VQ quantizer, dynamics `g`, and a pluggable prediction head (`PixelDecoder` now, `LatentHead` later). Losses are small callables composed by config into a weighted list consumed by a generic `Trainer`. The first run enables only `prediction` + `vq`; `margin` and `usage` are implemented and tested but off, ready to switch on when we observe action-ignoring or code collapse. Experiments live outside the package, compose `ssa` via Hydra `instantiate`, and log to Wandb.

**Tech Stack:** Python 3.12, uv, PyTorch, Hydra, Wandb, scikit-learn (NMI/ARI), matplotlib, pytest, ruff.

**Conventions:**
- Tensors: images `(B, C, H, W)`, `C=3`, `H=W=64`, float in `[0,1]`. Feature vectors `(B, dim)`.
- `ModelOutput` field order is fixed: `pred, target, a_pre, a_q, z_ctx, feat, vq`. Do not reorder.
- Loss callables share one signature: `__call__(out, batch, model) -> (Tensor, dict)`.
- Commit messages: imperative mood, no tool/AI attribution.

---

## Task 1: Project scaffold + smoke test

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/ssa/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import ssa

    assert ssa.__version__ == "0.1.0"
```

- [ ] **Step 2: Create project files**

`pyproject.toml`:
```toml
[project]
name = "ssa"
version = "0.1.0"
description = "Self-supervised action discovery"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.5",
    "numpy>=1.26",
    "hydra-core>=1.3",
    "wandb>=0.17",
    "matplotlib>=3.8",
    "scikit-learn>=1.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ssa"]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests", "experiments"]
```

`.python-version`:
```
3.12
```

`.gitignore`:
```
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
wandb/
outputs/
multirun/
*.pt
```

`src/ssa/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Sync and run the test**

Run: `uv sync && uv run pytest tests/test_smoke.py -v`
Expected: PASS (package builds, installs editable, imports).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .python-version .gitignore src/ssa/__init__.py tests/test_smoke.py uv.lock
git commit -m "chore: scaffold ssa package with uv + pytest"
```

---

## Task 2: Seed utility

**Files:**
- Create: `src/ssa/utils/__init__.py`, `src/ssa/utils/seed.py`, `tests/test_seed.py`

- [ ] **Step 1: Write the failing test**

`tests/test_seed.py`:
```python
import torch

from ssa.utils.seed import set_seed


def test_set_seed_makes_torch_deterministic():
    set_seed(0)
    a = torch.rand(5)
    set_seed(0)
    b = torch.rand(5)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: ssa.utils.seed`.

- [ ] **Step 3: Implement**

`src/ssa/utils/__init__.py`:
```python
```

`src/ssa/utils/seed.py`:
```python
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ssa/utils tests/test_seed.py
git commit -m "feat: add seed utility"
```

---

## Task 3: Vector quantizer (VQ bottleneck)

**Files:**
- Create: `src/ssa/models/__init__.py`, `src/ssa/models/quantizer.py`, `tests/test_quantizer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_quantizer.py`:
```python
import torch

from ssa.models.quantizer import VectorQuantizer


def test_quantizer_shapes_and_codes():
    vq = VectorQuantizer(num_codes=8, dim=4)
    z = torch.randn(5, 4)
    z_q, info = vq(z)
    assert z_q.shape == (5, 4)
    assert info["codes"].shape == (5,)
    assert info["dist"].shape == (5, 8)
    assert int(info["codes"].max()) < 8 and int(info["codes"].min()) >= 0


def test_straight_through_gradient_flows_to_input():
    vq = VectorQuantizer(num_codes=8, dim=4)
    z = torch.randn(5, 4, requires_grad=True)
    z_q, _ = vq(z)
    z_q.sum().backward()
    assert z.grad is not None and torch.any(z.grad != 0)


def test_perplexity_in_range():
    vq = VectorQuantizer(num_codes=8, dim=4)
    _, info = vq(torch.randn(32, 4))
    assert 1.0 <= float(info["perplexity"]) <= 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quantizer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/models/__init__.py`:
```python
```

`src/ssa/models/quantizer.py`:
```python
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    """VQ bottleneck with straight-through estimator.

    Returns the quantized vector and an info dict with commit/codebook losses,
    batch perplexity, hard code ids, and squared distances to every code (the
    usage loss consumes ``dist``).
    """

    def __init__(self, num_codes: int, dim: int) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.codebook = nn.Embedding(num_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: Tensor) -> tuple[Tensor, dict]:
        w = self.codebook.weight
        dist = z.pow(2).sum(1, keepdim=True) - 2 * z @ w.t() + w.pow(2).sum(1)
        codes = dist.argmin(dim=1)
        z_q = self.codebook(codes)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()  # straight-through
        probs = F.softmax(-dist, dim=1).mean(0)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        info = {
            "codes": codes,
            "dist": dist,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
        }
        return z_q, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_quantizer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/__init__.py src/ssa/models/quantizer.py tests/test_quantizer.py
git commit -m "feat: add VQ bottleneck quantizer"
```

---

## Task 4: Encoder

**Files:**
- Create: `src/ssa/models/encoder.py`, `tests/test_encoder.py`

- [ ] **Step 1: Write the failing test**

`tests/test_encoder.py`:
```python
import torch

from ssa.models.encoder import Encoder


def test_encoder_maps_frame_to_vector():
    enc = Encoder(in_ch=3, dim=128)
    z = enc(torch.rand(2, 3, 64, 64))
    assert z.shape == (2, 128)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/models/encoder.py`:
```python
from torch import Tensor, nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    """Small CNN mapping one ``(C, H, W)`` frame (H=W=64) to a ``(dim,)`` vector."""

    def __init__(self, in_ch: int = 3, dim: int = 256, widths=(32, 64, 128, 256)) -> None:
        super().__init__()
        chans = (in_ch, *widths)
        self.conv = nn.Sequential(*[_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.proj = nn.Linear(widths[-1] * 4 * 4, dim)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(x).flatten(1)
        return self.proj(h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_encoder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/encoder.py tests/test_encoder.py
git commit -m "feat: add CNN encoder"
```

---

## Task 5: Inverse model (latent action `f`)

**Files:**
- Create: `src/ssa/models/inverse.py`, `tests/test_inverse.py`

- [ ] **Step 1: Write the failing test**

`tests/test_inverse.py`:
```python
import torch

from ssa.models.inverse import InverseModel


def test_inverse_outputs_action_vector():
    f = InverseModel(dim=32, action_dim=8)
    a = f(torch.randn(4, 32), torch.randn(4, 32))
    assert a.shape == (4, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inverse.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/models/inverse.py`:
```python
import torch
from torch import Tensor, nn


class InverseModel(nn.Module):
    """Infers a pre-quantization action vector from ``(z_t, z_{t+1})``."""

    def __init__(self, dim: int = 256, action_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(), nn.Linear(hidden, action_dim)
        )
        self.action_dim = action_dim

    def forward(self, z_t: Tensor, z_tp1: Tensor) -> Tensor:
        return self.net(torch.cat([z_t, z_tp1], dim=-1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inverse.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/inverse.py tests/test_inverse.py
git commit -m "feat: add inverse latent-action model"
```

---

## Task 6: Dynamics (`g`)

**Files:**
- Create: `src/ssa/models/dynamics.py`, `tests/test_dynamics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dynamics.py`:
```python
import torch

from ssa.models.dynamics import Dynamics


def test_dynamics_predicts_feature():
    g = Dynamics(dim=32, action_dim=8)
    feat = g(torch.randn(4, 32), torch.randn(4, 8))
    assert feat.shape == (4, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dynamics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/models/dynamics.py`:
```python
import torch
from torch import Tensor, nn


class Dynamics(nn.Module):
    """Predicts the next-frame feature from ``(context, action)``."""

    def __init__(self, dim: int = 256, action_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + action_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )

    def forward(self, context: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([context, action], dim=-1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dynamics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/dynamics.py tests/test_dynamics.py
git commit -m "feat: add dynamics model"
```

---

## Task 7: Prediction heads (pluggable target interface)

**Files:**
- Create: `src/ssa/models/heads.py`, `tests/test_heads.py`

- [ ] **Step 1: Write the failing test**

`tests/test_heads.py`:
```python
from types import SimpleNamespace

import torch

from ssa.models.encoder import Encoder
from ssa.models.heads import LatentHead, PixelDecoder


def _batch():
    # heads only read batch.next_obs, so a lightweight stand-in keeps this test
    # independent of TransitionBatch (Task 8).
    return SimpleNamespace(next_obs=torch.rand(2, 3, 64, 64))


def test_pixel_decoder_predicts_image_and_targets_next_frame():
    head = PixelDecoder(dim=32)
    img = head.predict(torch.randn(2, 32))
    assert img.shape == (2, 3, 64, 64)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    batch = _batch()
    assert torch.equal(head.target(batch, model=None), batch.next_obs)


def test_latent_head_targets_teacher_encoding():
    class FakeModel:
        teacher = Encoder(in_ch=3, dim=16)

    head = LatentHead()
    feat = torch.randn(2, 16)
    assert torch.equal(head.predict(feat), feat)
    target = head.target(_batch(), FakeModel())
    assert target.shape == (2, 16)
    assert not target.requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_heads.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/models/heads.py`:
```python
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class PredictionHead(nn.Module, ABC):
    """Maps a dynamics feature to a prediction and exposes the regression target.

    ``target`` may consult the model (e.g. for an EMA teacher encoder).
    """

    @abstractmethod
    def predict(self, feat: Tensor) -> Tensor: ...

    @abstractmethod
    def target(self, batch, model) -> Tensor: ...


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class PixelDecoder(PredictionHead):
    """Decodes a feature vector to a ``(C, H, W)`` image; target is the next frame."""

    def __init__(self, dim: int = 256, out_ch: int = 3, base: int = 256) -> None:
        super().__init__()
        self.base = base
        self.fc = nn.Linear(dim, base * 4 * 4)
        self.net = nn.Sequential(_up(base, 128), _up(128, 64), _up(64, 32), _up(32, 32))
        self.out = nn.Conv2d(32, out_ch, 3, padding=1)

    def predict(self, feat: Tensor) -> Tensor:
        h = self.fc(feat).view(-1, self.base, 4, 4)
        h = self.net(h)  # 4 -> 8 -> 16 -> 32 -> 64
        return self.out(h).sigmoid()

    def target(self, batch, model) -> Tensor:
        return batch.next_obs


class LatentHead(PredictionHead):
    """Predicts the EMA-teacher encoding of the next frame (stop-grad target)."""

    def predict(self, feat: Tensor) -> Tensor:
        return feat

    def target(self, batch, model) -> Tensor:
        with torch.no_grad():
            return model.teacher(batch.next_obs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_heads.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/heads.py tests/test_heads.py
git commit -m "feat: add pluggable prediction heads (pixel, latent)"
```

---

## Task 8: TransitionBatch + collate

**Files:**
- Create: `src/ssa/data/__init__.py`, `src/ssa/data/batch.py`, `tests/test_batch.py`

- [ ] **Step 1: Write the failing test**

`tests/test_batch.py`:
```python
import torch

from ssa.data.batch import TransitionBatch, transition_collate


def test_collate_stacks_samples():
    samples = [
        {
            "obs": torch.rand(1, 3, 64, 64),
            "next_obs": torch.rand(3, 64, 64),
            "action": torch.tensor(i, dtype=torch.long),
        }
        for i in range(4)
    ]
    batch = transition_collate(samples)
    assert isinstance(batch, TransitionBatch)
    assert batch.obs.shape == (4, 1, 3, 64, 64)
    assert batch.next_obs.shape == (4, 3, 64, 64)
    assert batch.action.tolist() == [0, 1, 2, 3]


def test_to_moves_tensors():
    batch = transition_collate(
        [{"obs": torch.rand(1, 3, 64, 64), "next_obs": torch.rand(3, 64, 64),
          "action": torch.tensor(0)}]
    )
    assert batch.to("cpu").obs.device.type == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_batch.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/data/__init__.py`:
```python
```

`src/ssa/data/batch.py`:
```python
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TransitionBatch:
    """A batch of transitions. ``action`` holds ground-truth labels for eval only."""

    obs: Tensor  # (B, T, C, H, W)
    next_obs: Tensor  # (B, C, H, W)
    action: Tensor | None = None  # (B,)

    def to(self, device) -> "TransitionBatch":
        return TransitionBatch(
            self.obs.to(device),
            self.next_obs.to(device),
            None if self.action is None else self.action.to(device),
        )


def transition_collate(samples: list[dict]) -> TransitionBatch:
    return TransitionBatch(
        obs=torch.stack([s["obs"] for s in samples]),
        next_obs=torch.stack([s["next_obs"] for s in samples]),
        action=torch.stack([s["action"] for s in samples]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_batch.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/data tests/test_batch.py
git commit -m "feat: add TransitionBatch and collate"
```

---

## Task 9: LatentActionModel

**Files:**
- Create: `src/ssa/models/model.py`, `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import torch

from ssa.data.batch import TransitionBatch
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import LatentHead, PixelDecoder
from ssa.models.inverse import InverseModel
from ssa.models.model import LatentActionModel
from ssa.models.quantizer import VectorQuantizer


def _model(head):
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=head,
        teacher_momentum=None if isinstance(head, PixelDecoder) else 0.99,
    )


def _batch(b=2):
    return TransitionBatch(
        obs=torch.rand(b, 1, 3, 64, 64),
        next_obs=torch.rand(b, 3, 64, 64),
        action=torch.zeros(b, dtype=torch.long),
    )


def test_forward_pixel_head_shapes():
    out = _model(PixelDecoder(dim=32))(_batch())
    assert out.pred.shape == (2, 3, 64, 64)
    assert out.target.shape == (2, 3, 64, 64)
    assert out.a_q.shape == (2, 8)
    assert out.vq["codes"].shape == (2,)
    assert out.z_ctx.shape == (2, 32)


def test_forward_latent_head_uses_teacher():
    model = _model(LatentHead())
    out = model(_batch())
    assert out.pred.shape == (2, 32)
    assert out.target.shape == (2, 32)


def test_update_teacher_changes_teacher_toward_student():
    model = _model(LatentHead())
    before = model.teacher.proj.weight.clone()
    with torch.no_grad():
        model.encoder.proj.weight.add_(1.0)
    model.update_teacher()
    assert not torch.equal(before, model.teacher.proj.weight)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/models/model.py`:
```python
import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ModelOutput:
    pred: Tensor
    target: Tensor
    a_pre: Tensor
    a_q: Tensor
    z_ctx: Tensor
    feat: Tensor
    vq: dict


class LatentActionModel(nn.Module):
    """Wires encoder, inverse model, quantizer, dynamics, and a prediction head.

    When ``teacher_momentum`` is set (needed by ``LatentHead``), an EMA copy of
    the encoder is kept and updated via ``update_teacher()``.
    """

    def __init__(self, encoder, inverse, quantizer, dynamics, head, teacher_momentum=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.inverse = inverse
        self.quantizer = quantizer
        self.dynamics = dynamics
        self.head = head
        self.teacher_momentum = teacher_momentum
        if teacher_momentum is not None:
            self.teacher = copy.deepcopy(encoder)
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self) -> None:
        if self.teacher_momentum is None:
            return
        m = self.teacher_momentum
        for tp, sp in zip(self.teacher.parameters(), self.encoder.parameters(), strict=True):
            tp.mul_(m).add_(sp, alpha=1 - m)

    def forward(self, batch) -> ModelOutput:
        z_ctx = self.encoder(batch.obs[:, -1])
        z_tp1 = self.encoder(batch.next_obs)
        a_pre = self.inverse(z_ctx, z_tp1)
        a_q, vq = self.quantizer(a_pre)
        feat = self.dynamics(z_ctx, a_q)
        pred = self.head.predict(feat)
        target = self.head.target(batch, self)
        return ModelOutput(pred, target, a_pre, a_q, z_ctx, feat, vq)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/model.py tests/test_model.py
git commit -m "feat: add LatentActionModel wiring all components"
```

---

## Task 10: Losses (prediction, vq, margin, usage) + LossTerm

**Files:**
- Create: `src/ssa/losses/__init__.py`, `src/ssa/losses/base.py`, `src/ssa/losses/prediction.py`, `src/ssa/losses/vq.py`, `src/ssa/losses/margin.py`, `src/ssa/losses/usage.py`, `tests/test_losses.py`

- [ ] **Step 1: Write the failing test**

`tests/test_losses.py`:
```python
import torch

from ssa.data.batch import TransitionBatch
from ssa.losses.margin import MarginLoss
from ssa.losses.prediction import PredictionLoss
from ssa.losses.usage import UsageLoss
from ssa.losses.vq import VQLoss
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


def _batch(b=4):
    return TransitionBatch(
        obs=torch.rand(b, 1, 3, 64, 64),
        next_obs=torch.rand(b, 3, 64, 64),
        action=torch.zeros(b, dtype=torch.long),
    )


def test_all_losses_return_scalar_and_log():
    model, batch = _model(), _batch()
    out = model(batch)
    for loss in (PredictionLoss(), VQLoss(), MarginLoss(), UsageLoss()):
        val, log = loss(out, batch, model)
        assert val.ndim == 0
        assert isinstance(log, dict)


def test_margin_is_nonnegative():
    model, batch = _model(), _batch()
    val, _ = MarginLoss(m=0.1)(model(batch), batch, model)
    assert float(val) >= 0.0


def test_usage_reports_entropies():
    model, batch = _model(), _batch()
    _, log = UsageLoss()(model(batch), batch, model)
    assert "h_sample" in log and "h_batch" in log
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/losses/__init__.py`:
```python
```

`src/ssa/losses/base.py`:
```python
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class LossTerm:
    """A named, weighted loss callable: ``fn(out, batch, model) -> (Tensor, dict)``."""

    name: str
    fn: Callable
    weight: float = 1.0
```

`src/ssa/losses/prediction.py`:
```python
import torch.nn.functional as F


class PredictionLoss:
    """MSE between the head's prediction and its target (pixels or latents)."""

    def __call__(self, out, batch, model):
        loss = F.mse_loss(out.pred, out.target)
        return loss, {"pred": loss.item()}
```

`src/ssa/losses/vq.py`:
```python
class VQLoss:
    """Codebook + commitment loss from the quantizer info dict."""

    def __init__(self, beta: float = 0.25) -> None:
        self.beta = beta

    def __call__(self, out, batch, model):
        loss = out.vq["codebook_loss"] + self.beta * out.vq["commit_loss"]
        return loss, {"vq": loss.item(), "perplexity": out.vq["perplexity"].item()}
```

`src/ssa/losses/margin.py`:
```python
import torch
import torch.nn.functional as F


class MarginLoss:
    """No-action counterfactual: prediction with the inferred action must beat
    prediction with a zero action by margin ``m`` (off by default)."""

    def __init__(self, m: float = 0.1) -> None:
        self.m = m

    def __call__(self, out, batch, model):
        zero = torch.zeros_like(out.a_q)
        pred0 = model.head.predict(model.dynamics(out.z_ctx, zero))
        err = F.mse_loss(out.pred, out.target)
        err0 = F.mse_loss(pred0, out.target)
        loss = F.relu(self.m + err - err0)
        return loss, {"margin": loss.item(), "err": err.item(), "err0": err0.item()}
```

`src/ssa/losses/usage.py`:
```python
import torch.nn.functional as F


class UsageLoss:
    """Low per-sample code entropy (decisive) + high batch entropy (all codes
    used). Off by default."""

    def __init__(self, sample_weight: float = 1.0, batch_weight: float = 1.0) -> None:
        self.sample_weight = sample_weight
        self.batch_weight = batch_weight

    def __call__(self, out, batch, model):
        probs = F.softmax(-out.vq["dist"], dim=1)
        h_sample = -(probs * (probs + 1e-10).log()).sum(1).mean()
        p_bar = probs.mean(0)
        h_batch = -(p_bar * (p_bar + 1e-10).log()).sum()
        loss = self.sample_weight * h_sample - self.batch_weight * h_batch
        return loss, {"h_sample": h_sample.item(), "h_batch": h_batch.item()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_losses.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/losses tests/test_losses.py
git commit -m "feat: add loss terms (prediction, vq, margin, usage)"
```

---

## Task 11: Logger (Noop + Wandb)

**Files:**
- Create: `src/ssa/train/__init__.py`, `src/ssa/train/logging.py`, `tests/test_logging.py`

- [ ] **Step 1: Write the failing test**

`tests/test_logging.py`:
```python
from ssa.train.logging import NoopLogger


def test_noop_logger_accepts_calls():
    logger = NoopLogger()
    logger.log({"loss": 1.0}, step=0)
    logger.finish()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logging.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/train/__init__.py`:
```python
```

`src/ssa/train/logging.py`:
```python
from typing import Protocol


class Logger(Protocol):
    def log(self, metrics: dict, step: int) -> None: ...
    def finish(self) -> None: ...


class NoopLogger:
    """Logger that drops everything; used for tests and disabled runs."""

    def log(self, metrics: dict, step: int) -> None:
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

    def finish(self) -> None:
        self.run.finish()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ssa/train/__init__.py src/ssa/train/logging.py tests/test_logging.py
git commit -m "feat: add logger (noop + wandb)"
```

---

## Task 12: Trainer

**Files:**
- Create: `src/ssa/train/trainer.py`, `tests/test_trainer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_trainer.py`:
```python
import torch

from ssa.data.batch import TransitionBatch
from ssa.losses.base import LossTerm
from ssa.losses.prediction import PredictionLoss
from ssa.losses.vq import VQLoss
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import PixelDecoder
from ssa.models.inverse import InverseModel
from ssa.models.model import LatentActionModel
from ssa.models.quantizer import VectorQuantizer
from ssa.train.trainer import Trainer


def _model():
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )


def test_train_step_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    model = _model()
    batch = TransitionBatch(
        obs=torch.rand(4, 1, 3, 64, 64),
        next_obs=torch.rand(4, 3, 64, 64),
        action=torch.zeros(4, dtype=torch.long),
    )
    terms = [LossTerm("prediction", PredictionLoss(), 1.0), LossTerm("vq", VQLoss(), 1.0)]
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), terms)
    first = trainer.train_step(batch)["loss/total"]
    for _ in range(40):
        last = trainer.train_step(batch)["loss/total"]
    assert last < first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/train/trainer.py`:
```python
import torch


class Trainer:
    """Minimal training loop: weighted multi-term loss, periodic eval + logging,
    checkpointing. Agnostic to model internals."""

    def __init__(self, model, optimizer, loss_terms, device="cpu", logger=None, grad_clip=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_terms = loss_terms
        self.device = device
        self.logger = logger
        self.grad_clip = grad_clip

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
                    metrics = eval_fn(self.model, step)
                    if self.logger:
                        self.logger.log(metrics, step)
                    self.model.train()
                step += 1
                if step >= max_steps:
                    break

    def save(self, path) -> None:
        torch.save({"model": self.model.state_dict()}, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ssa/train/trainer.py tests/test_trainer.py
git commit -m "feat: add thin Trainer"
```

---

## Task 13: Eval — clustering + counterfactual

**Files:**
- Create: `src/ssa/eval/__init__.py`, `src/ssa/eval/clustering.py`, `src/ssa/eval/counterfactual.py`, `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

`tests/test_eval.py`:
```python
import torch

from ssa.eval.clustering import nmi_ari
from ssa.eval.counterfactual import counterfactual_grid
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import PixelDecoder
from ssa.models.inverse import InverseModel
from ssa.models.model import LatentActionModel
from ssa.models.quantizer import VectorQuantizer


def test_nmi_ari_perfect_for_consistent_relabeling():
    labels = [0, 0, 1, 1, 2, 2]
    codes = [5, 5, 3, 3, 7, 7]  # different ids, same partition
    m = nmi_ari(codes, labels)
    assert m["nmi"] > 0.99 and m["ari"] > 0.99


def test_counterfactual_grid_has_one_image_per_code():
    model = LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=6, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )
    grid = counterfactual_grid(model, torch.rand(1, 3, 64, 64))
    assert grid.shape == (6, 3, 64, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/ssa/eval/__init__.py`:
```python
```

`src/ssa/eval/clustering.py`:
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def nmi_ari(codes, labels) -> dict:
    """NMI and ARI between discovered code ids and ground-truth action labels."""
    codes = [int(c) for c in codes]
    labels = [int(x) for x in labels]
    return {
        "nmi": float(normalized_mutual_info_score(labels, codes)),
        "ari": float(adjusted_rand_score(labels, codes)),
    }
```

`src/ssa/eval/counterfactual.py`:
```python
import torch


@torch.no_grad()
def counterfactual_grid(model, obs):
    """Apply every codebook entry to a single context frame and decode.

    ``obs``: ``(T, C, H, W)`` history for one example. Returns ``(K, C, H, W)``
    predictions (meaningful only for a ``PixelDecoder`` head).
    """
    device = next(model.parameters()).device
    z = model.encoder(obs[-1:].to(device))  # (1, dim)
    codebook = model.quantizer.codebook.weight  # (K, action_dim)
    preds = [model.head.predict(model.dynamics(z, codebook[k : k + 1])) for k in range(len(codebook))]
    return torch.cat(preds, dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/eval tests/test_eval.py
git commit -m "feat: add eval (clustering NMI/ARI, counterfactual grid)"
```

---

## Task 14: Toy environment + dataset

**Files:**
- Create: `experiments/__init__.py` (empty, so pytest can collect), `experiments/0_synthetic_toy/env.py`, `experiments/0_synthetic_toy/test_env.py`

Note: pytest is configured to also collect `experiments/`. The test file sits next to `env.py` and imports it directly.

- [ ] **Step 1: Write the failing test**

`experiments/0_synthetic_toy/test_env.py`:
```python
import numpy as np

from env import ToyActionEnv, ToyDataset


def test_dataset_is_deterministic_for_a_seed():
    a = ToyDataset(size=8, seed=0, n_distractors=0)
    b = ToyDataset(size=8, seed=0, n_distractors=0)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.next, b.next)
    assert np.array_equal(a.act, b.act)


def test_shapes():
    ds = ToyDataset(size=4, seed=0, history=1, n_distractors=0)
    sample = ds[0]
    assert sample["obs"].shape == (1, 3, 64, 64)
    assert sample["next_obs"].shape == (3, 64, 64)
    assert sample["action"].dtype.__str__() == "torch.int64"


def test_action_moves_agent_in_expected_direction():
    # No distractors: the only red blob is the agent. Check centroid shift sign.
    env = ToyActionEnv(seed=0, n_distractors=0, step=6)
    for _ in range(20):
        obs, nxt, a = env.sample()
        cur_x, cur_y = _red_centroid(obs[-1])
        nxt_x, nxt_y = _red_centroid(nxt)
        dx, dy = env.actions[a]
        if dx != 0:
            assert np.sign(nxt_x - cur_x) == np.sign(dx) or nxt_x == cur_x  # clamped at wall
        if dy != 0:
            assert np.sign(nxt_y - cur_y) == np.sign(dy) or nxt_y == cur_y


def _red_centroid(img):
    mask = (img[0] > 0.9) & (img[1] < 0.1) & (img[2] < 0.1)
    ys, xs = np.nonzero(mask)
    return xs.mean(), ys.mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest experiments/0_synthetic_toy/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: env`.

- [ ] **Step 3: Implement**

`experiments/__init__.py`:
```python
```

`experiments/0_synthetic_toy/env.py`:
```python
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ToyActionEnv:
    """Procedural 2D sprite world. One red agent square moves by a fixed step in
    one of four directions (the hidden action); optional static distractor
    squares never move. Renders to ``(3, size, size)`` float frames in [0,1]."""

    def __init__(self, size=64, agent=6, step=6, n_distractors=2, history=1, seed=0):
        self.size = size
        self.agent = agent
        self.step = step
        self.n_distractors = n_distractors
        self.history = history
        self.rng = np.random.default_rng(seed)
        self.actions = np.array([(-step, 0), (step, 0), (0, -step), (0, step)], dtype=int)

    def _box(self, img, x, y, s, color):
        x0 = int(np.clip(x, 0, self.size - s))
        y0 = int(np.clip(y, 0, self.size - s))
        for c in range(3):
            img[c, y0 : y0 + s, x0 : x0 + s] = color[c]

    def _render(self, pos, distractors):
        img = np.ones((3, self.size, self.size), np.float32)
        for (dx, dy), color in distractors:
            self._box(img, dx, dy, 5, color)
        self._box(img, pos[0], pos[1], self.agent, (1.0, 0.0, 0.0))
        return img

    def sample(self):
        lo, hi = self.step, self.size - self.agent - self.step
        pos = self.rng.integers(lo, hi, size=2)
        distractors = [
            (
                tuple(int(v) for v in self.rng.integers(0, self.size - 5, size=2)),
                tuple(float(v) for v in self.rng.random(3)),
            )
            for _ in range(self.n_distractors)
        ]
        a = int(self.rng.integers(0, len(self.actions)))
        next_pos = np.clip(pos + self.actions[a], 0, self.size - self.agent)
        cur = self._render(pos, distractors)
        obs = np.stack([cur] * self.history, axis=0)  # (T, 3, H, W)
        nxt = self._render(next_pos, distractors)
        return obs, nxt, a


class ToyDataset(Dataset):
    """Pre-generates ``size`` transitions deterministically from a seed."""

    def __init__(self, size=4096, seed=0, **env_kwargs):
        env = ToyActionEnv(seed=seed, **env_kwargs)
        obs, nxt, act = [], [], []
        for _ in range(size):
            o, n, a = env.sample()
            obs.append(o)
            nxt.append(n)
            act.append(a)
        self.obs = np.stack(obs)
        self.next = np.stack(nxt)
        self.act = np.asarray(act)

    def __len__(self) -> int:
        return len(self.act)

    def __getitem__(self, i: int) -> dict:
        return {
            "obs": torch.from_numpy(self.obs[i]),
            "next_obs": torch.from_numpy(self.next[i]),
            "action": torch.tensor(int(self.act[i]), dtype=torch.long),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest experiments/0_synthetic_toy/test_env.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/__init__.py experiments/0_synthetic_toy/env.py experiments/0_synthetic_toy/test_env.py
git commit -m "feat(exp): add synthetic toy env + dataset"
```

---

## Task 15: Hydra configs + train entrypoint

**Files:**
- Create: `experiments/0_synthetic_toy/config/config.yaml`, `config/model/minimal.yaml`, `config/data/toy.yaml`, `config/loss/baseline.yaml`, `config/train/default.yaml`, `experiments/0_synthetic_toy/train.py`

- [ ] **Step 1: Create the config files**

`experiments/0_synthetic_toy/config/config.yaml`:
```yaml
defaults:
  - model: minimal
  - data: toy
  - loss: baseline
  - train: default
  - _self_

seed: 0
experiment: 0_synthetic_toy
wandb:
  project: ssa
  mode: offline   # set to "online" when ready, or "disabled" to skip wandb
```

`experiments/0_synthetic_toy/config/model/minimal.yaml`:
```yaml
dim: 256
action_dim: 64
num_codes: 16
teacher_momentum: null
encoder:
  _target_: ssa.models.encoder.Encoder
  in_ch: 3
  dim: ${model.dim}
inverse:
  _target_: ssa.models.inverse.InverseModel
  dim: ${model.dim}
  action_dim: ${model.action_dim}
quantizer:
  _target_: ssa.models.quantizer.VectorQuantizer
  num_codes: ${model.num_codes}
  dim: ${model.action_dim}
dynamics:
  _target_: ssa.models.dynamics.Dynamics
  dim: ${model.dim}
  action_dim: ${model.action_dim}
head:
  _target_: ssa.models.heads.PixelDecoder
  dim: ${model.dim}
```

`experiments/0_synthetic_toy/config/data/toy.yaml`:
```yaml
train_size: 8192
eval_size: 1024
seed: 0
batch_size: 128
num_workers: 4
env:
  size: 64
  agent: 6
  step: 6
  n_distractors: 2
  history: 1
```

`experiments/0_synthetic_toy/config/loss/baseline.yaml`:
```yaml
terms:
  - name: prediction
    weight: 1.0
    fn:
      _target_: ssa.losses.prediction.PredictionLoss
  - name: vq
    weight: 1.0
    fn:
      _target_: ssa.losses.vq.VQLoss
      beta: 0.25
```

`experiments/0_synthetic_toy/config/train/default.yaml`:
```yaml
max_steps: 5000
lr: 0.0003
eval_every: 500
log_every: 20
grad_clip: 1.0
device: cuda
```

- [ ] **Step 2: Create the train entrypoint**

`experiments/0_synthetic_toy/train.py`:
```python
import pathlib
import sys

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import ToyDataset  # noqa: E402

from ssa.data.batch import transition_collate
from ssa.eval.clustering import nmi_ari
from ssa.losses.base import LossTerm
from ssa.train.logging import NoopLogger, WandbLogger
from ssa.train.trainer import Trainer
from ssa.utils.seed import set_seed


def build_model(cfg):
    from ssa.models.model import LatentActionModel

    return LatentActionModel(
        encoder=instantiate(cfg.model.encoder),
        inverse=instantiate(cfg.model.inverse),
        quantizer=instantiate(cfg.model.quantizer),
        dynamics=instantiate(cfg.model.dynamics),
        head=instantiate(cfg.model.head),
        teacher_momentum=cfg.model.teacher_momentum,
    )


def make_eval_fn(eval_loader, device):
    def eval_fn(model, step):
        codes, labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                out = model(batch.to(device))
                codes += out.vq["codes"].tolist()
                labels += batch.action.tolist()
        return {f"eval/{k}": v for k, v in nmi_ari(codes, labels).items()}

    return eval_fn


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    env_kwargs = OmegaConf.to_container(cfg.data.env, resolve=True)

    train_ds = ToyDataset(size=cfg.data.train_size, seed=cfg.data.seed, **env_kwargs)
    eval_ds = ToyDataset(size=cfg.data.eval_size, seed=cfg.data.seed + 1, **env_kwargs)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=transition_collate,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.data.batch_size, collate_fn=transition_collate
    )

    model = build_model(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    terms = [LossTerm(t.name, instantiate(t.fn), t.weight) for t in cfg.loss.terms]

    logger = (
        NoopLogger()
        if cfg.wandb.mode == "disabled"
        else WandbLogger(
            project=cfg.wandb.project, mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    )

    trainer = Trainer(model, optim, terms, device=device, logger=logger,
                      grad_clip=cfg.train.grad_clip)
    trainer.fit(
        train_loader, max_steps=cfg.train.max_steps, eval_every=cfg.train.eval_every,
        eval_fn=make_eval_fn(eval_loader, device), log_every=cfg.train.log_every,
    )
    trainer.save("model.pt")
    logger.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify a short run executes end-to-end**

Run:
```bash
WANDB_MODE=offline uv run python experiments/0_synthetic_toy/train.py \
  train.max_steps=10 train.eval_every=5 data.train_size=256 data.eval_size=128 \
  data.num_workers=0 train.device=cpu wandb.mode=disabled
```
Expected: completes without error; prints Hydra run dir; `model.pt` written in that run dir (under `outputs/...`). No assertion on metrics — this only verifies wiring.

- [ ] **Step 4: Commit**

```bash
git add experiments/0_synthetic_toy/config experiments/0_synthetic_toy/train.py
git commit -m "feat(exp): add hydra configs + train entrypoint"
```

---

## Task 16: Eval entrypoint (figures + metrics)

**Files:**
- Create: `experiments/0_synthetic_toy/eval.py`, `experiments/0_synthetic_toy/results/.gitkeep`

- [ ] **Step 1: Create the eval script**

`experiments/0_synthetic_toy/eval.py`:
```python
import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import ToyDataset  # noqa: E402
from train import build_model  # noqa: E402

from ssa.data.batch import transition_collate
from ssa.eval.clustering import nmi_ari
from ssa.eval.counterfactual import counterfactual_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model.pt")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="config")

    model = build_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = OmegaConf.to_container(cfg.data.env, resolve=True)
    ds = ToyDataset(size=cfg.data.eval_size, seed=cfg.data.seed + 1, **env_kwargs)
    loader = DataLoader(ds, batch_size=128, collate_fn=transition_collate)

    codes, labels = [], []
    with torch.no_grad():
        for b in loader:
            o = model(b)
            codes += o.vq["codes"].tolist()
            labels += b.action.tolist()
    metrics = nmi_ari(codes, labels)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    sample = ds[0]
    grid = counterfactual_grid(model, sample["obs"])
    k = grid.shape[0]
    fig, axes = plt.subplots(1, k + 1, figsize=(2 * (k + 1), 2))
    axes[0].imshow(sample["obs"][-1].permute(1, 2, 0))
    axes[0].set_title("I_t")
    for i in range(k):
        axes[i + 1].imshow(grid[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i + 1].set_title(f"code {i}")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out / "counterfactual.png", bbox_inches="tight", dpi=120)
    print(metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify against the checkpoint from Task 15**

Run (from a directory containing the `model.pt` produced in Task 15, e.g. the Hydra output dir, or copy it to the experiment dir first):
```bash
cd experiments/0_synthetic_toy && uv run python eval.py --ckpt <path-to-model.pt> --out results
```
Expected: writes `results/metrics.json` and `results/counterfactual.png`; prints the metrics dict. (Metrics will be poor — it's a 10-step smoke checkpoint. We only verify the script runs.)

- [ ] **Step 3: Commit**

```bash
git add experiments/0_synthetic_toy/eval.py experiments/0_synthetic_toy/results/.gitkeep
git commit -m "feat(exp): add eval script (metrics + counterfactual figure)"
```

---

## Task 17: Experiment README, repo docs, research skeleton

**Files:**
- Create: `experiments/0_synthetic_toy/README.md`, `README.md`, `CONTRIBUTING.md`, `research/README.md`, `research/papers/.gitkeep`

- [ ] **Step 1: Write the experiment README (idea/hypothesis/design)**

`experiments/0_synthetic_toy/README.md`:
```markdown
# 0 — Synthetic Toy: does a VQ bottleneck discover the hidden action?

## Idea
Smallest faithful setting for self-supervised action discovery. A red agent
sprite moves L/R/U/D (the hidden action) over a plain background with static
distractors. We infer a discrete latent action through a VQ bottleneck and
predict the next frame, never seeing the true action during training.

## Hypothesis
With a tight VQ bottleneck, the discovered codes will partly align with the
four hidden actions. We expect to *observe* failure modes — the action being
ignored (history/identity shortcuts) and/or codebook collapse. Turning on the
`margin` (no-action counterfactual) and `usage` (entropy) losses should then
measurably raise NMI(code, true action) and codebook perplexity. That contrast
is the experiment.

## Design
- Model: `a = VQ(f(e(I_t), e(I_{t+1})))`, `Î_{t+1} = PixelDecoder(g(e(I_t), a))`.
- Loss (first run): pixel MSE + VQ only. Then add `margin`, then `usage`.
- Eval: NMI/ARI of codes vs true action; no-action prediction gap; codebook
  perplexity; counterfactual grid (apply each code to a fixed frame).

## Success target
NMI(code, true action) > 0.8 **and** a demonstrated no-action prediction gap;
codebook non-collapsed; counterfactual grid shows each code producing a
distinct, sensible move.

## Run
```bash
# baseline (pixel + vq)
uv run python train.py
# add anti-collapse terms once failures are observed (override loss group later)
```

## Results
See `results/RESULTS.md` (written after runs).
```

- [ ] **Step 2: Write repo README**

`README.md`:
```markdown
# self-supervised-actions

Self-supervised latent action discovery from video. Encode an image history,
infer a discrete latent action through a bottleneck, and predict the next
observation — forcing the action to capture controllable change.

## Layout
- `src/ssa/` — core package (MR-reviewed): models, losses, training, eval.
- `experiments/` — sandbox experiments; each has a README (idea/hypothesis),
  Hydra config, and a `results/` doc with figures.
- `research/` — papers wiki/database (pipeline is a later cycle).
- `docs/specs`, `docs/plans` — design specs and implementation plans.

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
```

- [ ] **Step 3: Write CONTRIBUTING**

`CONTRIBUTING.md`:
```markdown
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
```

- [ ] **Step 4: Write research skeleton**

`research/README.md`:
```markdown
# Research

Wiki/database of applicable literature for this project.

`papers/` will hold one subfolder per paper (downloaded source/arXiv, processed
notes, and a structured summary). The ingestion/wiki pipeline is its own build
cycle — see future spec. Seed reading list: Genie (latent action model), LAPO
(latent action policies), V-JEPA / V-JEPA 2, Temporal-DINO.
```

`research/papers/.gitkeep`:
```
```

- [ ] **Step 5: Commit**

```bash
git add experiments/0_synthetic_toy/README.md README.md CONTRIBUTING.md research/
git commit -m "docs: add experiment README, repo README, contributing, research skeleton"
```

---

## Final verification

- [ ] **Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean (fix anything flagged, then re-commit).

- [ ] **Confirm a short training run still works**

Run:
```bash
uv run python experiments/0_synthetic_toy/train.py \
  train.max_steps=20 data.train_size=256 data.eval_size=128 \
  data.num_workers=0 train.device=cpu wandb.mode=disabled
```
Expected: completes, writes `model.pt`.

---

## Self-Review Notes (for the plan author)

- **Spec coverage:** repo layout (Task 1, 17), core modules — encoder/inverse/quantizer/dynamics/heads/model (Tasks 3–9), losses incl. off-by-default margin/usage (Task 10), Trainer (Task 12), eval clustering+counterfactual (Task 13), data contracts (Task 8), toy env (Task 14), Hydra+Wandb+entrypoints (Tasks 15–16), research skeleton + CONTRIBUTING + README (Task 17). Probes (`eval/probes.py`), contrastive/cycle losses, and the latent-head full run are explicitly **out of scope** per the spec and are not tasked.
- **Dependency note:** `pillow`, `einops`, `tqdm` from the spec's wishlist are omitted (unused) to keep deps lean; add when first needed.
- **Ordering:** every task's tests depend only on earlier tasks. Task 7 (heads) uses a `SimpleNamespace` stand-in for the batch, so it does not depend on `TransitionBatch` (Task 8).
- **Interface consistency:** `ModelOutput(pred, target, a_pre, a_q, z_ctx, feat, vq)`, loss signature `(out, batch, model)`, quantizer info keys `codes/dist/commit_loss/codebook_loss/perplexity`, and `ToyDataset` attrs `obs/next/act` are used consistently across tasks.
```
