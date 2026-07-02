import torch

from ssa.data.batch import TransitionBatch
from ssa.models.dynamics import Dynamics
from ssa.models.encoder import Encoder
from ssa.models.heads import LatentHead, PixelDecoder
from ssa.models.inverse import InvariantInverseModel, InverseModel
from ssa.models.latent_action import LatentActionModel
from ssa.models.quantizer import VectorQuantizer


def _model(head):
    return LatentActionModel(
        encoder=Encoder(dim=32),
        inverse=InverseModel(dim=32, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=head,
        teacher_momentum=None if isinstance(head, PixelDecoder) else 0.99,
    )


def _batch(b=2):
    return TransitionBatch(
        obs=torch.rand(b, 1, 3, 64, 64),
        next_obs=torch.rand(b, 3, 64, 64),
        action=torch.zeros(b, dtype=torch.long),
    )


def test_forward_pixel_head_shapes():
    out = _model(PixelDecoder(dim=32))(_batch())
    assert out.pred.shape == (2, 3, 64, 64)
    assert out.target.shape == (2, 3, 64, 64)
    assert out.a_q.shape == (2, 8)
    assert out.vq["codes"].shape == (2,)
    assert out.z_ctx.shape == (2, 32)


def test_forward_latent_head_uses_teacher():
    model = _model(LatentHead())
    out = model(_batch())
    assert out.pred.shape == (2, 32)
    assert out.target.shape == (2, 32)


def test_update_teacher_changes_teacher_toward_student():
    model = _model(LatentHead())
    before = model.teacher.proj.weight.clone()
    with torch.no_grad():
        model.encoder.proj.weight.add_(1.0)
    model.update_teacher()
    assert not torch.equal(before, model.teacher.proj.weight)


def test_teacher_params_are_frozen():
    model = _model(LatentHead())
    assert all(not p.requires_grad for p in model.teacher.parameters())


def test_forward_invariant_inverse_runs():
    enc = Encoder(dim=32)
    model = LatentActionModel(
        encoder=enc,
        inverse=InvariantInverseModel(feat_ch=enc.feat_ch, action_dim=8),
        quantizer=VectorQuantizer(num_codes=4, dim=8),
        dynamics=Dynamics(dim=32, action_dim=8),
        head=LatentHead(),
        teacher_momentum=0.99,
    )
    out = model(_batch())
    assert out.a_q.shape == (2, 8)
    assert out.z_ctx.shape == (2, 32)
    assert out.vq["codes"].shape == (2,)
