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
