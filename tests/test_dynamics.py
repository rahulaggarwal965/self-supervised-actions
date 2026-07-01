import torch

from ssa.models.dynamics import Dynamics


def test_dynamics_predicts_feature():
    g = Dynamics(dim=32, action_dim=8)
    feat = g(torch.randn(4, 32), torch.randn(4, 8))
    assert feat.shape == (4, 32)


def test_dynamics_residual_adds_to_context():
    torch.manual_seed(0)
    g = Dynamics(dim=32, action_dim=8, residual=True)
    ctx, act = torch.randn(4, 32), torch.randn(4, 8)
    delta = g.net(torch.cat([ctx, act], dim=-1))
    assert torch.allclose(g(ctx, act), ctx + delta)  # additive transition
