import matplotlib

matplotlib.use("Agg")

import pathlib  # noqa: E402
import sys  # noqa: E402

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import ToyDataset  # noqa: E402

from ssa.data.batch import transition_collate  # noqa: E402
from ssa.eval.clustering import nmi_ari  # noqa: E402
from ssa.eval.figures import (  # noqa: E402
    code_action_confusion,
    codebook_usage_bar,
    counterfactual_figure,
    reconstruction_panel,
)
from ssa.eval.metrics import no_action_gap  # noqa: E402
from ssa.losses.base import LossTerm  # noqa: E402
from ssa.train.logging import NoopLogger, WandbLogger  # noqa: E402
from ssa.train.trainer import Trainer  # noqa: E402
from ssa.utils.seed import set_seed  # noqa: E402


def build_model(cfg):
    from ssa.models.model import LatentActionModel

    return LatentActionModel(
        encoder=instantiate(cfg.model.encoder),
        inverse=instantiate(cfg.model.inverse),
        quantizer=instantiate(cfg.model.quantizer),
        dynamics=instantiate(cfg.model.dynamics),
        head=instantiate(cfg.model.head),
        teacher_momentum=cfg.model.teacher_momentum,
    )


def make_eval_fn(eval_loader, device, num_codes, viz_n=8):
    import numpy as np

    viz_batch = next(iter(eval_loader))  # fixed batch so figures are comparable across steps

    def eval_fn(model, step):
        codes, labels = [], []
        mse_sum, count = 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                b = batch.to(device)
                out = model(b)
                codes += out.vq["codes"].tolist()
                labels += b.action.tolist()
                mse_sum += torch.nn.functional.mse_loss(
                    out.pred, out.target, reduction="sum"
                ).item()
                count += out.pred.numel()
        cluster = nmi_ari(codes, labels)
        counts = np.bincount(np.asarray(codes), minlength=num_codes).astype(float)
        probs = counts / counts.sum()
        perplexity = float(np.exp(-(probs * np.log(probs + 1e-10)).sum()))
        vb = viz_batch.to(device)
        gap = no_action_gap(model, vb)
        scalars = {
            "val/nmi": cluster["nmi"],
            "val/ari": cluster["ari"],
            "val/mse": mse_sum / count,
            "val/perplexity": perplexity,
            "val/action_err": gap["action_err"],
            "val/noaction_err": gap["noaction_err"],
            "val/noaction_gap": gap["gap"],
        }
        figures = {
            "recon/panel": reconstruction_panel(model, vb, n=viz_n),
            "counterfactual/grid": counterfactual_figure(model, vb.obs[0]),
            "codes/confusion": code_action_confusion(codes, labels, num_codes),
            "codes/usage": codebook_usage_bar(codes, num_codes),
        }
        return scalars, figures

    return eval_fn


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    env_kwargs = OmegaConf.to_container(cfg.data.env, resolve=True)

    train_ds = ToyDataset(cfg.data.train_size, seed=cfg.data.seed, env_cfg=env_kwargs)
    eval_ds = ToyDataset(cfg.data.eval_size, seed=cfg.data.seed + 1, env_cfg=env_kwargs)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=transition_collate,
    )
    eval_loader = DataLoader(eval_ds, batch_size=cfg.data.batch_size, collate_fn=transition_collate)

    model = build_model(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    terms = [LossTerm(t.name, instantiate(t.fn), t.weight) for t in cfg.loss.terms]

    logger = (
        NoopLogger()
        if cfg.wandb.mode == "disabled"
        else WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    )

    trainer = Trainer(
        model, optim, terms, device=device, logger=logger, grad_clip=cfg.train.grad_clip
    )
    trainer.fit(
        train_loader,
        max_steps=cfg.train.max_steps,
        eval_every=cfg.train.eval_every,
        eval_fn=make_eval_fn(eval_loader, device, cfg.model.num_codes),
        log_every=cfg.train.log_every,
    )
    trainer.save("model.pt")
    logger.finish()


if __name__ == "__main__":
    main()
