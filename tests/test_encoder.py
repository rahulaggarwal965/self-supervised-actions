import torch

from ssa.models.encoder import Encoder


def test_encoder_maps_frame_to_vector():
    enc = Encoder(in_ch=3, dim=128)
    z = enc(torch.rand(2, 3, 64, 64))
    assert z.shape == (2, 128)


def test_encoder_exposes_spatial_features():
    enc = Encoder(dim=32)
    x = torch.randn(2, 3, 64, 64)
    f = enc.features(x)
    assert f.shape == (2, enc.feat_ch, 4, 4)  # conv map before pooling
    assert enc.feat_ch == 256
    # forward is unchanged and equals project(features(x))
    assert torch.allclose(enc(x), enc.project(f))
