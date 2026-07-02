import torch

from ssa.data.batch import TransitionBatch
from ssa.losses.base import LossTerm
from ssa.losses.prediction import PredictionLoss
from ssa.losses.vq import VQLoss
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import PixelDecoder
from ssa.models.inverse import InverseModel
from ssa.models.model import LatentActionModel
from ssa.models.quantizer import VectorQuantizer
from ssa.train.trainer import Trainer


def _model():
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=PixelDecoder(dim=32),
    )


def test_train_step_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    model = _model()
    batch = TransitionBatch(
        obs=torch.rand(4, 1, 3, 64, 64),
        next_obs=torch.rand(4, 3, 64, 64),
        action=torch.zeros(4, dtype=torch.long),
    )
    terms = [LossTerm("prediction", PredictionLoss(), 1.0), LossTerm("vq", VQLoss(), 1.0)]
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), terms)
    first = trainer.train_step(batch)["loss/total"]
    for _ in range(40):
        last = trainer.train_step(batch)["loss/total"]
    assert last < first


def test_fit_runs_loop_with_eval_logging_and_restores_train_mode():
    torch.manual_seed(0)
    model = _model()
    batch = TransitionBatch(
        obs=torch.rand(2, 1, 3, 64, 64),
        next_obs=torch.rand(2, 3, 64, 64),
        action=torch.zeros(2, dtype=torch.long),
    )
    loader = [batch, batch]
    terms = [LossTerm("prediction", PredictionLoss(), 1.0), LossTerm("vq", VQLoss(), 1.0)]

    calls = {"n": 0}

    def eval_fn(m, step):
        calls["n"] += 1
        return {"eval/dummy": 0.0}, {}

    class RecordingLogger:
        def __init__(self):
            self.steps = []
            self.figure_calls = 0

        def log(self, metrics, step):
            self.steps.append(step)

        def log_figures(self, figures, step):
            self.figure_calls += 1

        def finish(self):
            pass

    logger = RecordingLogger()
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        terms,
        logger=logger,
        grad_clip=1.0,
    )
    trainer.fit(loader, max_steps=6, eval_every=2, eval_fn=eval_fn, log_every=2)

    # periodic evals at steps 2 and 4, plus one final eval at step 6 (the converged
    # model) so the dashboard reflects the model we keep — 3 evals total.
    assert calls["n"] == 3
    assert logger.figure_calls == 3
    assert model.training is True
    assert 0 in logger.steps


def test_train_step_logs_grad_norm_when_clipping():
    torch.manual_seed(0)
    model = _model()
    batch = TransitionBatch(
        obs=torch.rand(4, 1, 3, 64, 64),
        next_obs=torch.rand(4, 3, 64, 64),
        action=torch.zeros(4, dtype=torch.long),
    )
    terms = [LossTerm("prediction", PredictionLoss(), 1.0), LossTerm("vq", VQLoss(), 1.0)]
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), terms, grad_clip=1.0)
    logs = trainer.train_step(batch)
    assert "train/grad_norm" in logs and logs["train/grad_norm"] >= 0.0
