import matplotlib

matplotlib.use("Agg")
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ssa.data.batch import TransitionBatch  # noqa: E402
from ssa.eval.decoder_probe import (  # noqa: E402
    decoded_action_delta_figure,
    decoded_counterfactual_figure,
    train_decoder_probe,
)
from ssa.models.dynamics import Dynamics  # noqa: E402
from ssa.models.encoder import Encoder  # noqa: E402
from ssa.models.heads import PixelDecoder  # noqa: E402
from ssa.models.inverse import InverseModel  # noqa: E402
from ssa.models.latent_action import LatentActionModel  # noqa: E402
from ssa.models.quantizer import VectorQuantizer  # noqa: E402


def _model(num_codes=6):
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=num_codes, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )


def _batch(b=8):
    return TransitionBatch(
        obs=torch.rand(b, 1, 3, 64, 64),
        next_obs=torch.rand(b, 3, 64, 64),
        action=torch.zeros(b, dtype=torch.long),
    )


def test_decoder_probe_trains_and_decodes_counterfactual():
    model = _model(num_codes=6)
    probe = train_decoder_probe(model, [_batch()], steps=3)
    assert isinstance(probe, PixelDecoder)
    fig = decoded_counterfactual_figure(model, probe, _batch().obs[0])
    assert isinstance(fig, Figure)
    delta = decoded_action_delta_figure(model, probe, _batch().obs[0])
    assert isinstance(delta, Figure)
