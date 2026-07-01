import torch

from ssa.models.dynamics import Dynamics


def test_dynamics_predicts_feature():
    g = Dynamics(dim=32, action_dim=8)
    feat = g(torch.randn(4, 32), torch.randn(4, 8))
    assert feat.shape == (4, 32)
