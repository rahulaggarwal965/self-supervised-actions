from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class LossTerm:
    """A named, weighted loss callable: ``fn(out, batch, model) -> (Tensor, dict)``."""

    name: str
    fn: Callable
    weight: float = 1.0
