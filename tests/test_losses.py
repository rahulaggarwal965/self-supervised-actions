import torch

from ssa.data.batch import TransitionBatch
from ssa.losses.margin import MarginLoss
from ssa.losses.prediction import PredictionLoss
from ssa.losses.usage import UsageLoss
from ssa.losses.vq import VQLoss
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import PixelDecoder
from ssa.models.inverse import InverseModel
from ssa.models.latent_action import LatentActionModel
from ssa.models.quantizer import VectorQuantizer


def _model():
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )


def _batch(b=4):
    return TransitionBatch(
        obs=torch.rand(b, 1, 3, 64, 64),
        next_obs=torch.rand(b, 3, 64, 64),
        action=torch.zeros(b, dtype=torch.long),
    )


def test_all_losses_return_scalar_and_log():
    model, batch = _model(), _batch()
    out = model(batch)
    for loss in (PredictionLoss(), VQLoss(), MarginLoss(), UsageLoss()):
        val, log = loss(out, batch, model)
        assert val.ndim == 0
        assert isinstance(log, dict)


def test_margin_is_nonnegative():
    model, batch = _model(), _batch()
    val, _ = MarginLoss(m=0.1)(model(batch), batch, model)
    assert float(val) >= 0.0


def test_usage_reports_entropies():
    model, batch = _model(), _batch()
    _, log = UsageLoss()(model(batch), batch, model)
    assert "h_sample" in log and "h_batch" in log
