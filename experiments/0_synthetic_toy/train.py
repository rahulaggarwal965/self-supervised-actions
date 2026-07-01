import pathlib
import sys

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import ToyDataset  # noqa: E402

from ssa.data.batch import transition_collate
from ssa.eval.clustering import nmi_ari
from ssa.losses.base import LossTerm
from ssa.train.logging import NoopLogger, WandbLogger
from ssa.train.trainer import Trainer
from ssa.utils.seed import set_seed


def build_model(cfg):
    from ssa.models.latent_action import LatentActionModel

    return LatentActionModel(
        encoder=instantiate(cfg.model.encoder),
        inverse=instantiate(cfg.model.inverse),
        quantizer=instantiate(cfg.model.quantizer),
        dynamics=instantiate(cfg.model.dynamics),
        head=instantiate(cfg.model.head),
        teacher_momentum=cfg.model.teacher_momentum,
    )


def make_eval_fn(eval_loader, device):
    def eval_fn(model, step):
        codes, labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                out = model(batch.to(device))
                codes += out.vq["codes"].tolist()
                labels += batch.action.tolist()
        return {f"eval/{k}": v for k, v in nmi_ari(codes, labels).items()}

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
        eval_fn=make_eval_fn(eval_loader, device),
        log_every=cfg.train.log_every,
    )
    trainer.save("model.pt")
    logger.finish()


if __name__ == "__main__":
    main()
