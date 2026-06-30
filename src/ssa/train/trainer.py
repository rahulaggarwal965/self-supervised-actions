import torch


class Trainer:
    """Minimal training loop: weighted multi-term loss, periodic eval + logging,
    checkpointing. Agnostic to model internals."""

    def __init__(self, model, optimizer, loss_terms, device="cpu", logger=None, grad_clip=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_terms = loss_terms
        self.device = device
        self.logger = logger
        self.grad_clip = grad_clip

    def train_step(self, batch) -> dict:
        batch = batch.to(self.device)
        out = self.model(batch)
        logs: dict = {}
        total = 0.0
        for term in self.loss_terms:
            val, log = term.fn(out, batch, self.model)
            total = total + term.weight * val
            logs.update({f"{term.name}/{k}": v for k, v in log.items()})
        self.optimizer.zero_grad()
        total.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.model.update_teacher()
        logs["loss/total"] = total.item()
        return logs

    def fit(self, loader, max_steps, eval_every=0, eval_fn=None, log_every=10) -> None:
        self.model.train()
        step = 0
        while step < max_steps:
            for batch in loader:
                logs = self.train_step(batch)
                if self.logger and step % log_every == 0:
                    self.logger.log(logs, step)
                if eval_fn and eval_every and step > 0 and step % eval_every == 0:
                    self.model.eval()
                    metrics = eval_fn(self.model, step)
                    if self.logger:
                        self.logger.log(metrics, step)
                    self.model.train()
                step += 1
                if step >= max_steps:
                    break

    def save(self, path) -> None:
        torch.save({"model": self.model.state_dict()}, path)
