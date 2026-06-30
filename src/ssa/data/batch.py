from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TransitionBatch:
    """A batch of transitions. ``action`` holds ground-truth labels for eval only."""

    obs: Tensor  # (B, T, C, H, W)
    next_obs: Tensor  # (B, C, H, W)
    action: Tensor | None = None  # (B,)

    def to(self, device) -> "TransitionBatch":
        return TransitionBatch(
            self.obs.to(device),
            self.next_obs.to(device),
            None if self.action is None else self.action.to(device),
        )


def transition_collate(samples: list[dict]) -> TransitionBatch:
    return TransitionBatch(
        obs=torch.stack([s["obs"] for s in samples]),
        next_obs=torch.stack([s["next_obs"] for s in samples]),
        action=torch.stack([s["action"] for s in samples]),
    )
