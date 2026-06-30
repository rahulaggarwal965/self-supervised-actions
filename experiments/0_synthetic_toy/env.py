from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ToyActionEnv:
    """Procedural 2D sprite world. One red agent square moves by a fixed step in
    one of four directions (the hidden action); optional static distractor
    squares never move. Renders to ``(3, size, size)`` float frames in [0,1]."""

    def __init__(self, size=64, agent=6, step=6, n_distractors=2, history=1, seed=0):
        self.size = size
        self.agent = agent
        self.step = step
        self.n_distractors = n_distractors
        self.history = history
        self.rng = np.random.default_rng(seed)
        self.actions = np.array([(-step, 0), (step, 0), (0, -step), (0, step)], dtype=int)

    def _box(self, img, x, y, s, color):
        x0 = int(np.clip(x, 0, self.size - s))
        y0 = int(np.clip(y, 0, self.size - s))
        for c in range(3):
            img[c, y0 : y0 + s, x0 : x0 + s] = color[c]

    def _render(self, pos, distractors):
        img = np.ones((3, self.size, self.size), np.float32)
        for (dx, dy), color in distractors:
            self._box(img, dx, dy, 5, color)
        self._box(img, pos[0], pos[1], self.agent, (1.0, 0.0, 0.0))
        return img

    def sample(self):
        lo, hi = self.step, self.size - self.agent - self.step
        pos = self.rng.integers(lo, hi, size=2)
        distractors = [
            (
                tuple(int(v) for v in self.rng.integers(0, self.size - 5, size=2)),
                tuple(float(v) for v in self.rng.random(3)),
            )
            for _ in range(self.n_distractors)
        ]
        a = int(self.rng.integers(0, len(self.actions)))
        next_pos = np.clip(pos + self.actions[a], 0, self.size - self.agent)
        cur = self._render(pos, distractors)
        obs = np.stack([cur] * self.history, axis=0)  # (T, 3, H, W)
        nxt = self._render(next_pos, distractors)
        return obs, nxt, a


class ToyDataset(Dataset):
    """Pre-generates ``size`` transitions deterministically from a seed."""

    def __init__(self, size=4096, seed=0, **env_kwargs):
        env = ToyActionEnv(seed=seed, **env_kwargs)
        obs, nxt, act = [], [], []
        for _ in range(size):
            o, n, a = env.sample()
            obs.append(o)
            nxt.append(n)
            act.append(a)
        self.obs = np.stack(obs)
        self.next = np.stack(nxt)
        self.act = np.asarray(act)

    def __len__(self) -> int:
        return len(self.act)

    def __getitem__(self, i: int) -> dict:
        return {
            "obs": torch.from_numpy(self.obs[i]),
            "next_obs": torch.from_numpy(self.next[i]),
            "action": torch.tensor(int(self.act[i]), dtype=torch.long),
        }
