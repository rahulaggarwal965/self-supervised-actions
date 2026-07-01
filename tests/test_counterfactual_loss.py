import math
from types import SimpleNamespace

import torch

from ssa.losses.counterfactual import CounterfactualContrastiveLoss


def _teacher(x):
    # deterministic "encoder": flatten and take the first 16 dims
    return x.reshape(x.shape[0], -1)[:, :16]


def test_counterfactual_returns_scalar_and_acc():
    b, m = 8, 3
    out = SimpleNamespace(pred=torch.randn(b, 16), target=torch.randn(b, 16))
    batch = SimpleNamespace(next_cf=torch.rand(b, m, 3, 8, 8))
    model = SimpleNamespace(teacher=_teacher)
    val, log = CounterfactualContrastiveLoss()(out, batch, model)
    assert val.ndim == 0
    assert {"cf_contrastive", "cf_acc"} <= set(log)


def test_counterfactual_low_when_pred_matches_observed_over_counterfactuals():
    b, m = 8, 3
    observed = torch.rand(b, 3, 8, 8)
    cf = torch.rand(b, m, 3, 8, 8)
    # the prediction equals the teacher encoding of the OBSERVED next -> positive wins
    pred = _teacher(observed)
    out = SimpleNamespace(pred=pred, target=_teacher(observed))
    batch = SimpleNamespace(next_cf=cf)
    model = SimpleNamespace(teacher=_teacher)
    val, log = CounterfactualContrastiveLoss(temperature=0.1)(out, batch, model)
    assert log["cf_acc"] == 1.0  # observed future is the closest for every sample
    assert float(val) < math.log(1 + m)  # below the uniform-over-(1+M) baseline


def test_counterfactual_hinge_mode_low_when_pred_matches_observed():
    b, m = 8, 3
    observed = torch.rand(b, 3, 8, 8)
    cf = torch.rand(b, m, 3, 8, 8)
    pred = _teacher(observed)  # exactly the observed encoding -> H_pos = 0
    out = SimpleNamespace(pred=pred, target=_teacher(observed))
    batch = SimpleNamespace(next_cf=cf)
    model = SimpleNamespace(teacher=_teacher)
    val, log = CounterfactualContrastiveLoss(mode="hinge", margin=1.0)(out, batch, model)
    assert log["cf_acc"] == 1.0  # observed energy below every counterfactual energy
    assert float(val) >= 0.0


def test_counterfactual_delta_mode_runs():
    b, m, d = 8, 3, 16
    out = SimpleNamespace(pred=torch.randn(b, d), target=torch.randn(b, d), z_ctx=torch.randn(b, d))
    batch = SimpleNamespace(next_cf=torch.rand(b, m, 3, 8, 8))
    model = SimpleNamespace(teacher=_teacher)
    val, log = CounterfactualContrastiveLoss(delta=True)(out, batch, model)
    assert val.ndim == 0 and "cf_acc" in log
