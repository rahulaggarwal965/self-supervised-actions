import torch

from ssa.models.quantizer import VectorQuantizer


def test_quantizer_shapes_and_codes():
    vq = VectorQuantizer(num_codes=8, dim=4)
    z = torch.randn(5, 4)
    z_q, info = vq(z)
    assert z_q.shape == (5, 4)
    assert info["codes"].shape == (5,)
    assert info["dist"].shape == (5, 8)
    assert int(info["codes"].max()) < 8 and int(info["codes"].min()) >= 0


def test_straight_through_gradient_flows_to_input():
    vq = VectorQuantizer(num_codes=8, dim=4)
    z = torch.randn(5, 4, requires_grad=True)
    z_q, _ = vq(z)
    z_q.sum().backward()
    assert z.grad is not None and torch.any(z.grad != 0)


def test_perplexity_in_range():
    vq = VectorQuantizer(num_codes=8, dim=4)
    _, info = vq(torch.randn(32, 4))
    assert 1.0 <= float(info["perplexity"]) <= 8.0


def test_codes_select_nearest_codebook_rows():
    torch.manual_seed(0)
    vq = VectorQuantizer(num_codes=8, dim=4)
    z = torch.randn(5, 4)
    z_q, info = vq(z)
    assert torch.allclose(z_q, vq.codebook(info["codes"]), atol=1e-5)


def test_perplexity_collapses_to_one_for_identical_inputs():
    vq = VectorQuantizer(num_codes=8, dim=4)
    z = torch.ones(16, 4)  # identical inputs -> all map to the same code
    _, info = vq(z)
    assert abs(float(info["perplexity"]) - 1.0) < 1e-4
