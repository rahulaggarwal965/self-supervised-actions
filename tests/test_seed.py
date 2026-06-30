import torch

from ssa.utils.seed import set_seed


def test_set_seed_makes_torch_deterministic():
    set_seed(0)
    a = torch.rand(5)
    set_seed(0)
    b = torch.rand(5)
    assert torch.equal(a, b)
