from torch import Tensor, nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    """Small CNN mapping one ``(C, H, W)`` frame (H=W=64) to a ``(dim,)`` vector."""

    def __init__(self, in_ch: int = 3, dim: int = 256, widths=(32, 64, 128, 256)) -> None:
        super().__init__()
        chans = (in_ch, *widths)
        self.conv = nn.Sequential(*[_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.proj = nn.Linear(widths[-1] * 4 * 4, dim)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(x).flatten(1)
        return self.proj(h)
