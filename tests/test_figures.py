import matplotlib

matplotlib.use("Agg")
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ssa.data.batch import TransitionBatch  # noqa: E402
from ssa.eval.figures import (  # noqa: E402
    code_action_confusion,
    codebook_usage_bar,
    counterfactual_figure,
    reconstruction_panel,
)
from ssa.models.dynamics import Dynamics  # noqa: E402
from ssa.models.encoder import Encoder  # noqa: E402
from ssa.models.heads import PixelDecoder  # noqa: E402
from ssa.models.inverse import InverseModel  # noqa: E402
from ssa.models.model import LatentActionModel  # noqa: E402
from ssa.models.quantizer import VectorQuantizer  # noqa: E402


def _model(num_codes=6):
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=num_codes, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )


def _batch(b=4):
    return TransitionBatch(
        obs=torch.rand(b, 1, 3, 64, 64),
        next_obs=torch.rand(b, 3, 64, 64),
        action=torch.zeros(b, dtype=torch.long),
    )


def test_reconstruction_panel_returns_figure():
    assert isinstance(reconstruction_panel(_model(), _batch(), n=3), Figure)


def test_counterfactual_figure_returns_figure():
    fig = counterfactual_figure(_model(num_codes=6), _batch().obs[0])
    assert isinstance(fig, Figure)


def test_confusion_and_usage_return_figures():
    codes = [0, 0, 1, 2, 2, 2]
    labels = [0, 0, 1, 1, 2, 2]
    assert isinstance(code_action_confusion(codes, labels, num_codes=6), Figure)
    assert isinstance(codebook_usage_bar(codes, num_codes=6), Figure)
