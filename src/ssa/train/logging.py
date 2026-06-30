from typing import Protocol


class Logger(Protocol):
    def log(self, metrics: dict, step: int) -> None: ...
    def finish(self) -> None: ...


class NoopLogger:
    """Logger that drops everything; used for tests and disabled runs."""

    def log(self, metrics: dict, step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger:
    """Thin wrapper over ``wandb.init``/``run.log``."""

    def __init__(self, project: str, config: dict | None = None, **kwargs) -> None:
        import wandb

        self.run = wandb.init(project=project, config=config, **kwargs)

    def log(self, metrics: dict, step: int) -> None:
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()
