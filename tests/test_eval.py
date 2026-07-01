import torch

from ssa.eval.clustering import nmi_ari
from ssa.eval.counterfactual import counterfactual_grid
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import PixelDecoder
from ssa.models.inverse import InverseModel
from ssa.models.latent_action import LatentActionModel
from ssa.models.quantizer import VectorQuantizer


def test_nmi_ari_perfect_for_consistent_relabeling():
    labels = [0, 0, 1, 1, 2, 2]
    codes = [5, 5, 3, 3, 7, 7]  # different ids, same partition
    m = nmi_ari(codes, labels)
    assert m["nmi"] > 0.99 and m["ari"] > 0.99


def test_counterfactual_grid_has_one_image_per_code():
    model = LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=6, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )
    grid = counterfactual_grid(model, torch.rand(1, 3, 64, 64))
    assert grid.shape == (6, 3, 64, 64)
