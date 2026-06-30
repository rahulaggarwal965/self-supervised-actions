from types import SimpleNamespace

import torch

from ssa.losses.vicreg import VICRegLoss


def _out(z):
    return SimpleNamespace(z_ctx=z)


def test_vicreg_penalizes_collapsed_embeddings():
    loss = VICRegLoss()
    collapsed_val, collapsed_log = loss(_out(torch.zeros(16, 8)), None, None)
    spread_val, _ = loss(_out(torch.randn(256, 8) * 3.0), None, None)
    # a collapsed (zero-variance) embedding is penalised far more than a spread one
    assert float(collapsed_val) > float(spread_val)
    assert collapsed_log["z_std"] < 0.5


def test_vicreg_returns_scalar_and_log():
    val, log = VICRegLoss()(_out(torch.randn(32, 8)), None, None)
    assert val.ndim == 0
    assert {"vicreg", "var", "cov", "z_std"} <= set(log)
