import torch

from ssa.models.inverse import InverseModel


def test_inverse_outputs_action_vector():
    f = InverseModel(dim=32, action_dim=8)
    a = f(torch.randn(4, 32), torch.randn(4, 32))
    assert a.shape == (4, 8)


def test_inverse_delta_input_depends_only_on_the_change():
    f = InverseModel(dim=32, action_dim=8, delta_input=True)
    z_t, z_tp1, shift = torch.randn(4, 32), torch.randn(4, 32), torch.randn(4, 32)
    a = f(z_t, z_tp1)
    assert a.shape == (4, 8)
    # action is a function of z_{t+1} - z_t only, so a constant shift leaves it unchanged
    assert torch.allclose(a, f(z_t + shift, z_tp1 + shift), atol=1e-5)
