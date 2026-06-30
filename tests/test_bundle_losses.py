from types import SimpleNamespace

import torch
from torch import nn

from ssa.losses.decorrelation import DecorrelationLoss
from ssa.losses.supervision import ActionSupervisionLoss


def _model(num_codes=6, action_dim=8):
    cb = nn.Embedding(num_codes, action_dim)
    return SimpleNamespace(quantizer=SimpleNamespace(codebook=cb))


def test_action_supervision_labels_a_fraction_and_returns_scalar():
    model = _model()
    out = SimpleNamespace(a_pre=torch.randn(40, 8))
    batch = SimpleNamespace(action=torch.randint(0, 4, (40,)))
    val, log = ActionSupervisionLoss(label_frac=0.025)(out, batch, model)
    assert val.ndim == 0
    assert log["n_labeled"] == 1  # round(0.025 * 40)
    # a larger fraction labels more samples
    _, log_half = ActionSupervisionLoss(label_frac=0.5)(out, batch, model)
    assert log_half["n_labeled"] == 20


def test_action_supervision_is_zero_when_a_pre_matches_action_prototype():
    model = _model()
    actions = torch.randint(0, 4, (40,))
    a_pre = torch.randn(40, 8)
    # make the (single) labeled sample sit exactly on its action's codebook row
    a_pre[0] = model.quantizer.codebook.weight[actions[0]].detach()
    out = SimpleNamespace(a_pre=a_pre)
    batch = SimpleNamespace(action=actions)
    val, _ = ActionSupervisionLoss(label_frac=0.025)(out, batch, model)
    assert float(val) < 1e-6


def test_decorrelation_penalizes_correlated_more_than_independent():
    torch.manual_seed(0)
    z = torch.randn(256, 16)
    # a_pre correlated with z (copies its first 8 dims) vs independent
    corr = SimpleNamespace(a_pre=z[:, :8].clone(), z_ctx=z)
    indep = SimpleNamespace(a_pre=torch.randn(256, 8), z_ctx=z)
    loss = DecorrelationLoss()
    corr_val, log = loss(corr, None, None)
    indep_val, _ = loss(indep, None, None)
    assert float(corr_val) > float(indep_val)
    assert corr_val.ndim == 0 and "decorr" in log
