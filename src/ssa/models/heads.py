from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class PredictionHead(nn.Module, ABC):
    """Maps a dynamics feature to a prediction and exposes the regression target.

    ``target`` may consult the model (e.g. for an EMA teacher encoder).
    """

    @abstractmethod
    def predict(self, feat: Tensor) -> Tensor: ...

    @abstractmethod
    def target(self, batch, model) -> Tensor: ...


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class PixelDecoder(PredictionHead):
    """Decodes a feature vector to a ``(C, H, W)`` image.

    With ``delta=False`` (default) it predicts the next frame directly (sigmoid,
    target = ``I_{t+1}``). With ``delta=True`` it predicts the residual
    ``I_{t+1} - I_t`` (tanh, since the delta lives in [-1, 1]); this forces the
    action to explain the change rather than the static scene.
    """

    def __init__(
        self, dim: int = 256, out_ch: int = 3, base: int = 256, delta: bool = False
    ) -> None:
        super().__init__()
        self.base = base
        self.delta = delta
        self.fc = nn.Linear(dim, base * 4 * 4)
        self.net = nn.Sequential(_up(base, 128), _up(128, 64), _up(64, 32), _up(32, 32))
        self.out = nn.Conv2d(32, out_ch, 3, padding=1)

    def predict(self, feat: Tensor) -> Tensor:
        h = self.fc(feat).view(-1, self.base, 4, 4)
        h = self.net(h)  # 4 -> 8 -> 16 -> 32 -> 64
        out = self.out(h)
        return out.tanh() if self.delta else out.sigmoid()

    def target(self, batch, model) -> Tensor:
        if self.delta:
            return batch.next_obs - batch.obs[:, -1]
        return batch.next_obs


class LatentHead(PredictionHead):
    """Predicts the EMA-teacher encoding of the next frame (stop-grad target)."""

    def predict(self, feat: Tensor) -> Tensor:
        return feat

    def target(self, batch, model) -> Tensor:
        with torch.no_grad():
            return model.teacher(batch.next_obs)
