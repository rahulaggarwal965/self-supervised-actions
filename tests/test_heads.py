from types import SimpleNamespace

import torch

from ssa.models.encoder import Encoder
from ssa.models.heads import LatentHead, PixelDecoder


def _batch():
    # heads only read batch.next_obs, so a lightweight stand-in keeps this test
    # independent of TransitionBatch (Task 8).
    return SimpleNamespace(next_obs=torch.rand(2, 3, 64, 64))


def test_pixel_decoder_predicts_image_and_targets_next_frame():
    head = PixelDecoder(dim=32)
    img = head.predict(torch.randn(2, 32))
    assert img.shape == (2, 3, 64, 64)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    batch = _batch()
    assert torch.equal(head.target(batch, model=None), batch.next_obs)


def test_latent_head_targets_teacher_encoding():
    class FakeModel:
        teacher = Encoder(in_ch=3, dim=16)

    head = LatentHead()
    feat = torch.randn(2, 16)
    assert torch.equal(head.predict(feat), feat)
    target = head.target(_batch(), FakeModel())
    assert target.shape == (2, 16)
    assert not target.requires_grad
