from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ToyActionEnv:
    """Procedural 2D sprite world. One red agent square moves by a fixed step in
    one of four directions (the hidden action); optional static distractor
    squares never move. Renders to ``(3, size, size)`` float frames in [0,1]."""

    def __init__(self, size=64, agent=6, step=6, n_distractors=2, history=1, seed=0, start=None):
        self.size = size
        self.agent = agent
        self.step = step
        self.n_distractors = n_distractors
        self.history = history
        # start: fixed agent (x, y) for every sample (decouples position from the
        # action); None = random spawn. A control for action discovery.
        self.start = None if start is None else np.array(start, dtype=int)
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

    def _next_frame(self, pos, a, distractors):
        next_pos = np.clip(pos + self.actions[a], 0, self.size - self.agent)
        return self._render(next_pos, distractors)

    def sample(self, counterfactuals: bool = False):
        lo, hi = self.step, self.size - self.agent - self.step
        pos = self.start.copy() if self.start is not None else self.rng.integers(lo, hi, size=2)
        distractors = [
            (
                tuple(int(v) for v in self.rng.integers(0, self.size - 5, size=2)),
                tuple(float(v) for v in self.rng.random(3)),
            )
            for _ in range(self.n_distractors)
        ]
        a = int(self.rng.integers(0, len(self.actions)))
        cur = self._render(pos, distractors)
        obs = np.stack([cur] * self.history, axis=0)  # (T, 3, H, W)
        nxt = self._next_frame(pos, a, distractors)
        if not counterfactuals:
            return obs, nxt, a
        # next frames under the OTHER actions (same state) — label-free negatives:
        # one code cannot predict all of these different futures, so a contrastive
        # forward objective against them cannot be satisfied by codebook collapse.
        others = [ai for ai in range(len(self.actions)) if ai != a]
        ncf = np.stack([self._next_frame(pos, ai, distractors) for ai in others])  # (A-1, 3, H, W)
        return obs, nxt, a, ncf


class ToyDataset(Dataset):
    """Pre-generates ``size`` transitions deterministically from a seed."""

    def __init__(self, size=4096, seed=0, env_cfg=None, counterfactuals=False, **env_kwargs):
        # ``size`` is the number of transitions to generate. Environment settings
        # may be passed either as loose keyword args (``**env_kwargs``) or as a
        # single ``env_cfg`` mapping; the latter avoids a keyword collision when
        # the env's own grid side length (also called ``size``) must be configured.
        # ``counterfactuals``: also store the next frames under the other actions
        # (same state), as label-free negatives for a contrastive forward objective.
        if env_cfg:
            env_kwargs = {**env_cfg, **env_kwargs}
        env = ToyActionEnv(seed=seed, **env_kwargs)
        self.counterfactuals = counterfactuals
        obs, nxt, act, ncf = [], [], [], []
        for _ in range(size):
            if counterfactuals:
                o, n, a, c = env.sample(counterfactuals=True)
                ncf.append(c)
            else:
                o, n, a = env.sample()
            obs.append(o)
            nxt.append(n)
            act.append(a)
        self.obs = np.stack(obs)
        self.next = np.stack(nxt)
        self.act = np.asarray(act)
        self.next_cf = np.stack(ncf) if counterfactuals else None  # (N, A-1, 3, H, W)

    def __len__(self) -> int:
        return len(self.act)

    def __getitem__(self, i: int) -> dict:
        item = {
            "obs": torch.from_numpy(self.obs[i]),
            "next_obs": torch.from_numpy(self.next[i]),
            "action": torch.tensor(int(self.act[i]), dtype=torch.long),
        }
        if self.next_cf is not None:
            item["next_cf"] = torch.from_numpy(self.next_cf[i])  # (A-1, 3, H, W)
        return item
