import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ModelOutput:
    pred: Tensor
    target: Tensor
    a_pre: Tensor
    a_q: Tensor
    z_ctx: Tensor
    feat: Tensor
    vq: dict


class LatentActionModel(nn.Module):
    """Wires encoder, inverse model, quantizer, dynamics, and a prediction head.

    When ``teacher_momentum`` is set (needed by ``LatentHead``), an EMA copy of
    the encoder is kept and updated via ``update_teacher()``.
    """

    def __init__(self, encoder, inverse, quantizer, dynamics, head, teacher_momentum=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.inverse = inverse
        self.quantizer = quantizer
        self.dynamics = dynamics
        self.head = head
        self.teacher_momentum = teacher_momentum
        if teacher_momentum is not None:
            self.teacher = copy.deepcopy(encoder)
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self) -> None:
        if self.teacher_momentum is None:
            return
        m = self.teacher_momentum
        for tp, sp in zip(self.teacher.parameters(), self.encoder.parameters(), strict=True):
            tp.mul_(m).add_(sp, alpha=1 - m)

    def forward(self, batch) -> ModelOutput:
        f_ctx = self.encoder.features(batch.obs[:, -1])
        z_ctx = self.encoder.project(f_ctx)
        if getattr(self.inverse, "spatial", False):
            # position-invariant: action from the inter-frame feature-map diff,
            # read at the inverse's chosen encoder level (None = final map)
            lvl = getattr(self.inverse, "feat_level", None)
            f_t = f_ctx if lvl is None else self.encoder.features(batch.obs[:, -1], lvl)
            a_pre = self.inverse(f_t, self.encoder.features(batch.next_obs, lvl))
        else:
            a_pre = self.inverse(z_ctx, self.encoder(batch.next_obs))
        a_q, vq = self.quantizer(a_pre)
        feat = self.dynamics(z_ctx, a_q)
        pred = self.head.predict(feat)
        target = self.head.target(batch, self)
        return ModelOutput(pred, target, a_pre, a_q, z_ctx, feat, vq)
