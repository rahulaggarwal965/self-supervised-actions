import torch

from ssa.models.encoder import Encoder


def test_encoder_maps_frame_to_vector():
    enc = Encoder(in_ch=3, dim=128)
    z = enc(torch.rand(2, 3, 64, 64))
    assert z.shape == (2, 128)
