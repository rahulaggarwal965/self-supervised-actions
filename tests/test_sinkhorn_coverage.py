import torch

from ssa.losses.sinkhorn_coverage import SinkhornCoverageLoss


def test_sinkhorn_plan_marginals():
    loss = SinkhornCoverageLoss(eps=0.05, iters=30)
    cost = torch.rand(3, 6, 4)
    plan = loss._sinkhorn(cost)
    assert torch.allclose(plan.sum(dim=(1, 2)), torch.ones(3), atol=1e-3)
    assert torch.allclose(plan.sum(dim=2), torch.full((3, 6), 1 / 6), atol=1e-2)
    assert torch.allclose(plan.sum(dim=1), torch.full((3, 4), 1 / 4), atol=1e-2)


def test_sinkhorn_prefers_aligned_predictions():
    """Transported cost is lower when predictions sit on the real futures."""
    loss = SinkhornCoverageLoss(eps=0.05, iters=30)
    futures = torch.randn(2, 4, 8)
    aligned = torch.cat([futures, futures[:, :2]], dim=1)  # 6 preds covering the 4 futures
    far = torch.randn(2, 6, 8) * 5.0
    c_aligned = (aligned.unsqueeze(2) - futures.unsqueeze(1)).pow(2).mean(-1)
    c_far = (far.unsqueeze(2) - futures.unsqueeze(1)).pow(2).mean(-1)
    assert (loss._sinkhorn(c_aligned) * c_aligned).sum() < (loss._sinkhorn(c_far) * c_far).sum()
