from torch import Tensor, nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    """Small CNN mapping one ``(C, H, W)`` frame (H=W=64) to a ``(dim,)`` vector.

    ``features`` exposes the conv feature map *before* pooling (translation-
    equivariant); ``forward`` projects it to the global vector. The feature map
    is what position-invariant action inference reads. Passing ``level`` returns
    an earlier (higher-resolution) map after that many conv blocks — e.g.
    ``level=2`` gives a ``(64, 16, 16)`` map, finer than the ``(256, 4, 4)``
    default, so small motions are resolvable rather than sub-cell.
    """

    def __init__(self, in_ch: int = 3, dim: int = 256, widths=(32, 64, 128, 256)) -> None:
        super().__init__()
        chans = (in_ch, *widths)
        self.conv = nn.Sequential(*[_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.proj = nn.Linear(widths[-1] * 4 * 4, dim)
        self.dim = dim
        self.feat_ch = widths[-1]

    def features(self, x: Tensor, level: int | None = None) -> Tensor:
        return self.conv(x) if level is None else self.conv[:level](x)

    def project(self, f: Tensor) -> Tensor:
        return self.proj(f.flatten(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.project(self.features(x))
