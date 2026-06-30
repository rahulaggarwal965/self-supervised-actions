import torch

from ssa.data.batch import TransitionBatch
from ssa.eval.metrics import no_action_gap
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import PixelDecoder
from ssa.models.inverse import InverseModel
from ssa.models.model import LatentActionModel
from ssa.models.quantizer import VectorQuantizer


def _model():
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )


def test_no_action_gap_returns_three_floats_with_gap_identity():
    batch = TransitionBatch(
        obs=torch.rand(4, 1, 3, 64, 64),
        next_obs=torch.rand(4, 3, 64, 64),
        action=torch.zeros(4, dtype=torch.long),
    )
    m = no_action_gap(_model(), batch)
    assert set(m) == {"action_err", "noaction_err", "gap"}
    assert all(isinstance(v, float) for v in m.values())
    assert abs(m["gap"] - (m["noaction_err"] - m["action_err"])) < 1e-6
