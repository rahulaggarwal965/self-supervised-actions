from types import SimpleNamespace

import torch
from torch import nn

from ssa.losses.coverage import FutureCoverageLoss


def test_future_coverage_returns_scalar_and_diagnostic():
    b, m, d, k, a = 8, 3, 16, 6, 8
    w = torch.randn(a, d)
    model = SimpleNamespace(
        quantizer=SimpleNamespace(codebook=nn.Embedding(k, a)),
        dynamics=lambda z, act: z + act @ w,
        head=SimpleNamespace(predict=lambda f: f),
        teacher=lambda x: x.reshape(x.shape[0], -1)[:, :d],
    )
    out = SimpleNamespace(z_ctx=torch.randn(b, d), target=torch.randn(b, d))
    batch = SimpleNamespace(next_cf=torch.rand(b, m, 3, 8, 8))
    val, log = FutureCoverageLoss()(out, batch, model)
    assert val.ndim == 0
    assert {"future_cov", "cov_distinct"} <= set(log)
