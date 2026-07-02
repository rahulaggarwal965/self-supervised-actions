from types import SimpleNamespace

import torch
from torch import nn

from ssa.losses.forward_selection import ForwardSelectionLoss


def _model(k=6, a=8, d=16):
    # a toy model whose dynamics genuinely depends on the code: z + code @ W
    w = torch.randn(a, d)
    cb = nn.Embedding(k, a)
    return SimpleNamespace(
        quantizer=SimpleNamespace(codebook=cb),
        dynamics=lambda z, act: z + act @ w,
        head=SimpleNamespace(predict=lambda f: f),
    )


def test_forward_selection_returns_scalar_and_acc():
    torch.manual_seed(0)
    m = _model()
    z = torch.randn(12, 16)
    codes = torch.randint(0, 6, (12,))
    target = torch.randn(12, 16)
    out = SimpleNamespace(z_ctx=z, target=target, vq={"codes": codes})
    val, log = ForwardSelectionLoss()(out, None, m)
    assert val.ndim == 0
    assert {"fwd_sel", "fwd_sel_acc"} <= set(log)
    assert 0.0 <= log["fwd_sel_acc"] <= 1.0


def test_forward_selection_low_when_assigned_code_predicts_best():
    torch.manual_seed(0)
    m = _model()
    z = torch.randn(12, 16)
    codes = torch.randint(0, 6, (12,))
    # make the target exactly the assigned code's own prediction -> that code wins
    with torch.no_grad():
        proto = m.quantizer.codebook.weight[codes]  # (12, 8)
        target = m.dynamics(z, proto)
    out = SimpleNamespace(z_ctx=z, target=target, vq={"codes": codes})
    val, log = ForwardSelectionLoss(temperature=0.1)(out, None, m)
    assert log["fwd_sel_acc"] == 1.0  # assigned code is the closest for every sample
    # and the loss is much lower than the uniform-baseline cross-entropy (ln 6)
    import math

    assert float(val) < math.log(6)
