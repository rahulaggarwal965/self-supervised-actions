import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import ToyDataset  # noqa: E402
from train import build_model  # noqa: E402

from ssa.data.batch import transition_collate  # noqa: E402
from ssa.eval.clustering import nmi_ari  # noqa: E402
from ssa.eval.counterfactual import counterfactual_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model.pt")
    ap.add_argument("--out", default="subexperiments/0-baseline")
    args = ap.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="config")

    model = build_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = OmegaConf.to_container(cfg.data.env, resolve=True)
    ds = ToyDataset(cfg.data.eval_size, seed=cfg.data.seed + 1, env_cfg=env_kwargs)
    loader = DataLoader(ds, batch_size=128, collate_fn=transition_collate)

    codes, labels = [], []
    with torch.no_grad():
        for b in loader:
            o = model(b)
            codes += o.vq["codes"].tolist()
            labels += b.action.tolist()
    metrics = nmi_ari(codes, labels)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    sample = ds[0]
    grid = counterfactual_grid(model, sample["obs"])
    k = grid.shape[0]
    fig, axes = plt.subplots(1, k + 1, figsize=(2 * (k + 1), 2))
    axes[0].imshow(sample["obs"][-1].permute(1, 2, 0))
    axes[0].set_title("I_t")
    for i in range(k):
        axes[i + 1].imshow(grid[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i + 1].set_title(f"code {i}")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out / "counterfactual.png", bbox_inches="tight", dpi=120)
    print(metrics)


if __name__ == "__main__":
    main()
