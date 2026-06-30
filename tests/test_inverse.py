import torch

from ssa.models.inverse import InverseModel


def test_inverse_outputs_action_vector():
    f = InverseModel(dim=32, action_dim=8)
    a = f(torch.randn(4, 32), torch.randn(4, 32))
    assert a.shape == (4, 8)
