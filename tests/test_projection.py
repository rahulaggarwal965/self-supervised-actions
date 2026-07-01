from types import SimpleNamespace

import torch

from ssa.losses.counterfactual import CounterfactualContrastiveLoss
from ssa.models.projection import ProjectionHead


def test_projection_head_shape():
    assert ProjectionHead(dim=32, out_dim=8)(torch.randn(4, 32)).shape == (4, 8)


def test_cf_loss_uses_projection_when_present():
    b, m, d = 6, 3, 32
    out = SimpleNamespace(pred=torch.randn(b, d), target=torch.randn(b, d))
    batch = SimpleNamespace(next_cf=torch.rand(b, m, 3, 8, 8))
    teacher = lambda x: x.reshape(x.shape[0], -1)[:, :d]  # noqa: E731
    model = SimpleNamespace(teacher=teacher, projection=ProjectionHead(dim=d, out_dim=8))
    val, log = CounterfactualContrastiveLoss()(out, batch, model)
    assert val.ndim == 0 and "cf_acc" in log
