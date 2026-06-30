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
        z_ctx = self.encoder(batch.obs[:, -1])
        z_tp1 = self.encoder(batch.next_obs)
        a_pre = self.inverse(z_ctx, z_tp1)
        a_q, vq = self.quantizer(a_pre)
        feat = self.dynamics(z_ctx, a_q)
        pred = self.head.predict(feat)
        target = self.head.target(batch, self)
        return ModelOutput(pred, target, a_pre, a_q, z_ctx, feat, vq)
