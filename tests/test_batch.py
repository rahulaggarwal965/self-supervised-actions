import torch

from ssa.data.batch import TransitionBatch, transition_collate


def test_collate_stacks_samples():
    samples = [
        {
            "obs": torch.rand(1, 3, 64, 64),
            "next_obs": torch.rand(3, 64, 64),
            "action": torch.tensor(i, dtype=torch.long),
        }
        for i in range(4)
    ]
    batch = transition_collate(samples)
    assert isinstance(batch, TransitionBatch)
    assert batch.obs.shape == (4, 1, 3, 64, 64)
    assert batch.next_obs.shape == (4, 3, 64, 64)
    assert batch.action.tolist() == [0, 1, 2, 3]


def test_to_moves_tensors():
    batch = transition_collate(
        [
            {
                "obs": torch.rand(1, 3, 64, 64),
                "next_obs": torch.rand(3, 64, 64),
                "action": torch.tensor(0),
            }
        ]
    )
    assert batch.to("cpu").obs.device.type == "cpu"
