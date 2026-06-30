# Position-Invariant Action Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the discovered VQ code position-invariant by inferring the action from a spatially-pooled inter-frame feature difference, so the random-position toy reaches the fixed-start control's action discovery (NMI 0.013 → ≥ 0.5).

**Architecture:** Keep the whole working pipeline (latent prediction + EMA teacher + VICReg + VQ). Change only the action-inference path: expose the encoder's conv feature map; a new `InvariantInverseModel` takes `features(I_{t+1}) − features(I_t)`, applies a small conv, global-average-pools over space (discarding absolute position), and projects to `a_pre`. Dynamics, prediction, VICReg, and the teacher are untouched.

**Tech Stack:** PyTorch, Hydra config groups, pytest, ruff. See `docs/specs/2026-06-30-position-invariant-action-inference-design.md`.

---

## File Structure

- `src/ssa/models/encoder.py` — add `features()` (conv map) + `project()`; `forward` composes them; expose `feat_ch`.
- `src/ssa/models/inverse.py` — add `InvariantInverseModel` (conv + global average pool); mark `InverseModel.spatial = False`.
- `src/ssa/models/model.py` — `forward` branches on `inverse.spatial` to feed feature maps vs. global vectors.
- `experiments/0_synthetic_toy/config/model/minimal_invariant.yaml` — new model config.
- `tests/test_encoder.py`, `tests/test_invariant_inverse.py`, `tests/test_model.py` — tests.

The encoder always emits `widths[-1]=256` conv channels at a `4×4` spatial map (64→32→16→8→4). The invariant inverse therefore uses `feat_ch=256`.

---

### Task 1: Encoder exposes its spatial feature map

**Files:**
- Modify: `src/ssa/models/encoder.py`
- Test: `tests/test_encoder.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_encoder.py`:

```python
def test_encoder_exposes_spatial_features():
    enc = Encoder(dim=32)
    x = torch.randn(2, 3, 64, 64)
    f = enc.features(x)
    assert f.shape == (2, enc.feat_ch, 4, 4)  # conv map before pooling
    assert enc.feat_ch == 256
    # forward is unchanged and equals project(features(x))
    assert torch.allclose(enc(x), enc.project(f))
```

(Ensure `import torch` is present at the top of the file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_encoder.py::test_encoder_exposes_spatial_features -v`
Expected: FAIL (`Encoder` has no attribute `features` / `feat_ch` / `project`).

- [ ] **Step 3: Implement**

Replace the body of `Encoder` in `src/ssa/models/encoder.py` with:

```python
class Encoder(nn.Module):
    """Small CNN mapping one ``(C, H, W)`` frame (H=W=64) to a ``(dim,)`` vector.

    ``features`` exposes the conv feature map *before* pooling (translation-
    equivariant, ``(feat_ch, 4, 4)``); ``forward`` projects it to the global
    vector. The feature map is what position-invariant action inference reads.
    """

    def __init__(self, in_ch: int = 3, dim: int = 256, widths=(32, 64, 128, 256)) -> None:
        super().__init__()
        chans = (in_ch, *widths)
        self.conv = nn.Sequential(*[_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.proj = nn.Linear(widths[-1] * 4 * 4, dim)
        self.dim = dim
        self.feat_ch = widths[-1]

    def features(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def project(self, f: Tensor) -> Tensor:
        return self.proj(f.flatten(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.project(self.features(x))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_encoder.py -v`
Expected: PASS (existing encoder tests still pass too).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/encoder.py tests/test_encoder.py
git commit -m "feat(encoder): expose spatial feature map (features/project)"
```

---

### Task 2: InvariantInverseModel (conv + global average pool)

**Files:**
- Modify: `src/ssa/models/inverse.py`
- Test: `tests/test_invariant_inverse.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_invariant_inverse.py`:

```python
import torch

from ssa.models.inverse import InvariantInverseModel


def test_invariant_inverse_outputs_action_vector():
    f = InvariantInverseModel(feat_ch=16, action_dim=8)
    f.eval()
    a = f(torch.randn(4, 16, 4, 4), torch.randn(4, 16, 4, 4))
    assert a.shape == (4, 8)


def test_invariant_inverse_is_translation_invariant():
    # conv uses circular padding + global average pool, so a spatial roll of
    # BOTH frames leaves the inferred action unchanged: the code depends on the
    # local change pattern, not on where in the frame it happened.
    torch.manual_seed(0)
    f = InvariantInverseModel(feat_ch=16, action_dim=8)
    f.eval()
    f_t, f_tp1 = torch.randn(4, 16, 4, 4), torch.randn(4, 16, 4, 4)
    a = f(f_t, f_tp1)
    rolled_t = torch.roll(f_t, shifts=(1, 2), dims=(2, 3))
    rolled_tp1 = torch.roll(f_tp1, shifts=(1, 2), dims=(2, 3))
    assert torch.allclose(a, f(rolled_t, rolled_tp1), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_invariant_inverse.py -v`
Expected: FAIL (`cannot import name 'InvariantInverseModel'`).

- [ ] **Step 3: Implement**

Add to `src/ssa/models/inverse.py` (and set `spatial = False` on the existing `InverseModel` class body — add the line `spatial = False` just inside `class InverseModel(nn.Module):`, above `__init__`):

```python
class InvariantInverseModel(nn.Module):
    """Position-invariant action inference.

    Infers the pre-quantization action from the inter-frame *feature-map*
    difference ``features(I_{t+1}) - features(I_t)`` via a small conv followed by
    a global average pool over space. The conv uses circular padding so it is
    shift-equivariant; the spatial mean is then shift-invariant. The result: the
    code depends on the *local* change pattern (a left-move's signature), not on
    the agent's absolute position.
    """

    spatial = True

    def __init__(self, feat_ch: int = 256, action_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.proj = nn.Linear(hidden, action_dim)
        self.action_dim = action_dim

    def forward(self, f_t: Tensor, f_tp1: Tensor) -> Tensor:
        d = f_tp1 - f_t  # (B, feat_ch, H', W') — the motion signature
        h = self.net(d).mean(dim=(2, 3))  # global average pool -> (B, hidden)
        return self.proj(h)
```

Note: `GroupNorm` and `SiLU` are pointwise/per-spatial-statistic, so they preserve shift-equivariance under circular roll; combined with `mean` over space the output is shift-invariant, which the property test verifies exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_invariant_inverse.py tests/test_inverse.py -v`
Expected: PASS (both files).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/inverse.py tests/test_invariant_inverse.py
git commit -m "feat(inverse): position-invariant inverse (feature diff + GAP)"
```

---

### Task 3: Wire the invariant inverse into the model

**Files:**
- Modify: `src/ssa/models/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py` (uses the existing `_batch` helper; add the import):

```python
from ssa.models.inverse import InvariantInverseModel  # add near the other imports


def test_forward_invariant_inverse_runs():
    enc = Encoder(dim=32)
    model = LatentActionModel(
        encoder=enc,
        inverse=InvariantInverseModel(feat_ch=enc.feat_ch, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=LatentHead(),
        teacher_momentum=0.99,
    )
    out = model(_batch())
    assert out.a_q.shape == (2, 8)
    assert out.z_ctx.shape == (2, 32)
    assert out.vq["codes"].shape == (2,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py::test_forward_invariant_inverse_runs -v`
Expected: FAIL — the current `forward` calls `self.inverse(z_ctx, z_tp1)` with global vectors, but `InvariantInverseModel.forward` expects 4-D feature maps (shape/dim error).

- [ ] **Step 3: Implement**

Replace `LatentActionModel.forward` in `src/ssa/models/model.py` with:

```python
    def forward(self, batch) -> ModelOutput:
        f_ctx = self.encoder.features(batch.obs[:, -1])
        z_ctx = self.encoder.project(f_ctx)
        if getattr(self.inverse, "spatial", False):
            # position-invariant: action from the inter-frame feature-map diff
            a_pre = self.inverse(f_ctx, self.encoder.features(batch.next_obs))
        else:
            a_pre = self.inverse(z_ctx, self.encoder(batch.next_obs))
        a_q, vq = self.quantizer(a_pre)
        feat = self.dynamics(z_ctx, a_q)
        pred = self.head.predict(feat)
        target = self.head.target(batch, self)
        return ModelOutput(pred, target, a_pre, a_q, z_ctx, feat, vq)
```

This computes the context feature map once and derives `z_ctx` from it (no double conv). The non-spatial path is behaviorally identical to before (`z_ctx = encoder(obs[:,-1])`, `a_pre = inverse(z_ctx, encoder(next_obs))`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS (existing pixel/latent-head model tests still pass — the non-spatial branch is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/ssa/models/model.py tests/test_model.py
git commit -m "feat(model): route feature maps to the position-invariant inverse"
```

---

### Task 4: `minimal_invariant` model config

**Files:**
- Create: `experiments/0_synthetic_toy/config/model/minimal_invariant.yaml`

- [ ] **Step 1: Create the config**

Create `experiments/0_synthetic_toy/config/model/minimal_invariant.yaml`:

```yaml
# Position-invariant variant: same V-JEPA latent setup as minimal_latent, but the
# action is inferred by InvariantInverseModel from the inter-frame feature-map
# difference (conv + global average pool), so the code is invariant to the agent's
# absolute position. feat_ch must match the encoder's last conv width (256).
# Select with `model=minimal_invariant`.
dim: 256
action_dim: 64
num_codes: 16
teacher_momentum: 0.99
encoder:
  _target_: ssa.models.encoder.Encoder
  in_ch: 3
  dim: ${model.dim}
inverse:
  _target_: ssa.models.inverse.InvariantInverseModel
  feat_ch: 256
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
  _target_: ssa.models.heads.LatentHead
```

- [ ] **Step 2: Verify the config builds a runnable model (CPU smoke)**

Run from the experiment dir:

```bash
cd experiments/0_synthetic_toy && MPLBACKEND=Agg uv run python train.py \
  model=minimal_invariant loss=vicreg \
  data.train_size=64 data.eval_size=64 data.batch_size=32 \
  train.max_steps=4 train.eval_every=4 train.log_every=2 train.probe_steps=3 \
  train.device=cpu wandb.mode=disabled
```

Expected: exits 0 (model instantiates via Hydra, trains 4 steps, runs eval + the decoder-probe counterfactual since the head is `LatentHead`).

- [ ] **Step 3: Run the full suite + lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add experiments/0_synthetic_toy/config/model/minimal_invariant.yaml
git commit -m "feat(exp): minimal_invariant model config"
```

---

### Task 5: Run the experiment + write subexperiment `7-invariant`

This is the experiment phase (not TDD). Driven from the main session after Tasks 1–4 land.

- [ ] **Step 1: Launch the main + sanity runs on `bench` (online)**

Main (random-position toy, the failing setting) and a sanity run on the fixed-start setting, via the `--env`-forwarding launcher (forwards `WANDB_API_KEY` + `TORCHINDUCTOR_CACHE_DIR`):

```
bench submit --name invariant --out 'outputs/**' --env WANDB_API_KEY --env TORCHINDUCTOR_CACHE_DIR \
  -- uv run python experiments/0_synthetic_toy/train.py model=minimal_invariant loss=vicreg wandb.mode=online
bench submit --name invariant-fixed --out 'outputs/**' --env WANDB_API_KEY --env TORCHINDUCTOR_CACHE_DIR \
  -- uv run python experiments/0_synthetic_toy/train.py model=minimal_invariant loss=vicreg '+data.env.start=[29,29]' wandb.mode=online
```

Spread across free benches (alpa12, alpamayoDT1, alpa11, ix-host).

- [ ] **Step 2: Fetch metrics + figures**

After completion, read each run's `wandb-summary.json` (nmi, ari, perplexity, z_std, action_err, noaction_err) and scp the decoded counterfactual + code/usage figures from the remote wandb media.

- [ ] **Step 3: Diagnostic — NMI(code, position)**

Recompute `NMI(code, position-bucket)` on the main run (4×4 position grid, as in Exp 5) to confirm it *drops* relative to Exp 5's 0.064. Use the existing `nmi_ari` helper against a 4×4-bucketed agent position.

- [ ] **Step 4: Write `subexperiments/7-invariant/`**

Create `experiments/0_synthetic_toy/subexperiments/7-invariant/` with `README.md` (reproduce command, hypothesis, metrics+figures, interpretation, conclusion→next), `config.yaml` (resolved, as in the other subexperiments), `metrics.json`, and the figures. Add the throughline row + synthesis update to `subexperiments/RESULTS.md`. Verdict against the success bar (NMI ≥ 0.5 primary, > 0.8 stretch; NMI(code, position) dropped).

- [ ] **Step 5: Commit**

```bash
git add experiments/0_synthetic_toy/subexperiments
git commit -m "exp: position-invariant inverse results (subexperiment 7)"
```

---

## Self-Review

- **Spec coverage:** encoder `features()` (Task 1), invariant inverse conv+GAP (Task 2), model wiring / inverse-path-only change (Task 3), `minimal_invariant` config (Task 4), experiment + diagnostics + subexperiment doc (Task 5), translation-invariance property test (Task 2), success criteria + NMI(code,position) diagnostic (Task 5). All covered.
- **Type consistency:** `Encoder.features/project/feat_ch`, `InvariantInverseModel(feat_ch, action_dim, hidden)` with `spatial = True`, `InverseModel.spatial = False`, `model.forward` branch on `getattr(inverse, "spatial", False)` — consistent across tasks.
- **No placeholders:** every code/test step shows full code and exact commands.
