import numpy as np
from env import ToyActionEnv, ToyDataset


def test_dataset_is_deterministic_for_a_seed():
    a = ToyDataset(size=8, seed=0, n_distractors=0)
    b = ToyDataset(size=8, seed=0, n_distractors=0)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.next, b.next)
    assert np.array_equal(a.act, b.act)


def test_shapes():
    ds = ToyDataset(size=4, seed=0, history=1, n_distractors=0)
    sample = ds[0]
    assert sample["obs"].shape == (1, 3, 64, 64)
    assert sample["next_obs"].shape == (3, 64, 64)
    assert sample["action"].dtype.__str__() == "torch.int64"


def test_action_moves_agent_in_expected_direction():
    # No distractors: the only red blob is the agent. Check centroid shift sign.
    env = ToyActionEnv(seed=0, n_distractors=0, step=6)
    for _ in range(20):
        obs, nxt, a = env.sample()
        cur_x, cur_y = _red_centroid(obs[-1])
        nxt_x, nxt_y = _red_centroid(nxt)
        dx, dy = env.actions[a]
        if dx != 0:
            assert np.sign(nxt_x - cur_x) == np.sign(dx) or nxt_x == cur_x  # clamped at wall
        if dy != 0:
            assert np.sign(nxt_y - cur_y) == np.sign(dy) or nxt_y == cur_y


def test_fixed_start_keeps_agent_position_constant():
    env = ToyActionEnv(seed=0, n_distractors=0, start=[20, 20])
    centroids = [_red_centroid(env.sample()[0][-1]) for _ in range(10)]
    xs = {cx for cx, _ in centroids}
    ys = {cy for _, cy in centroids}
    assert len(xs) == 1 and len(ys) == 1  # agent starts at the same spot every sample


def _red_centroid(img):
    mask = (img[0] > 0.9) & (img[1] < 0.1) & (img[2] < 0.1)
    ys, xs = np.nonzero(mask)
    return xs.mean(), ys.mean()
