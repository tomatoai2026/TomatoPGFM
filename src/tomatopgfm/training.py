from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - CPU audit hosts may not carry torch.
    torch = None
    F = None


@dataclass
class TaskHealth:
    name: str
    history: list[float] = field(default_factory=list)
    metric_history: list[float] = field(default_factory=list)
    collapsed: bool = False

    def update(self, loss: float, metric: float | None = None, patience: int = 3, eps: float = 1e-4) -> None:
        self.history.append(float(loss))
        if metric is not None:
            self.metric_history.append(float(metric))
        recent = self.history[-patience:]
        # (a) hard collapse: loss has gone (and stayed) essentially to zero.
        loss_near_zero = len(recent) == patience and max(recent) < eps
        # (b) silent collapse: the task metric has stopped improving while the
        # loss is no longer decreasing. This catches constant-prediction
        # collapse (e.g. VTP always predicts the majority class, CPC matches
        # self) which never drives the loss to zero and would otherwise be missed.
        metric_stalled = False
        if len(self.metric_history) >= patience:
            mrecent = self.metric_history[-patience:]
            no_metric_gain = (max(mrecent) - min(mrecent)) <= eps
            loss_not_improving = len(recent) == patience and recent[-1] >= recent[0] - eps
            metric_stalled = no_metric_gain and loss_not_improving
        if loss_near_zero or metric_stalled:
            self.collapsed = True


class LossRegistry:
    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        self.health = {k: TaskHealth(k) for k in weights}

    def combine(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for name, w in self.weights.items():
            if w <= 0:
                continue
            if self.health.get(name, TaskHealth(name)).collapsed:
                continue
            if name not in losses:
                raise KeyError(
                    f"Weighted task '{name}' (weight={w}) has no loss term this step; "
                    f"got {sorted(losses)}. Refusing to silently drop a configured task."
                )
            loss = losses[name]
            total = w * loss if total is None else total + w * loss
        if total is None:
            raise RuntimeError("All training tasks collapsed or have zero weight")
        return total

    def update_health(self, scalar_losses: dict[str, float], metrics: dict[str, float] | None = None,
                      patience_override: int | None = None) -> dict:
        metrics = metrics or {}
        kw = {} if patience_override is None else {"patience": int(patience_override)}
        for name, value in scalar_losses.items():
            self.health.setdefault(name, TaskHealth(name)).update(value, metrics.get(name), **kw)
        return {name: vars(h) for name, h in self.health.items()}


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if F is None:
        raise RuntimeError("torch is required for mlm_loss")
    active = mask.bool()
    if active.sum() == 0:
        return logits.sum() * 0
    return F.cross_entropy(logits[active], labels[active])


def vtp_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Variant-type prediction loss.

    Accepts optional per-class weights so the configured class-balancing
    requirement can be honoured on the imbalanced VTP label distribution.
    """
    if F is None:
        raise RuntimeError("torch is required for vtp_loss")
    return F.cross_entropy(logits, labels, weight=class_weights)


def cpc_loss(path_proj: torch.Tensor, positive_index: torch.Tensor, temperature: float = 0.07, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """InfoNCE over path projections with the self-similarity diagonal masked.

    Without masking the diagonal, every row's own (max) self-match dominates the
    softmax and the model collapses by matching itself instead of its positive
    bubble-allele path. Masking self removes that trivial solution.
    """
    if F is None or torch is None:
        raise RuntimeError("torch is required for cpc_loss")
    logits = path_proj @ path_proj.T / temperature
    diag = torch.arange(logits.shape[0], device=logits.device)
    logits[diag, diag] = float("-inf")
    loss = F.cross_entropy(logits, positive_index, reduction="none")
    if sample_weight is None:
        return loss.mean()
    w = sample_weight.to(device=loss.device, dtype=loss.dtype)
    if w.numel() != loss.numel():
        raise ValueError(f"sample_weight length {w.numel()} != CPC batch length {loss.numel()}")
    denom = w.sum().clamp_min(1e-12)
    return (loss * w).sum() / denom


def graph_recon_loss(recon: torch.Tensor, target: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
    """Denoising reconstruction of graph node features over all valid tokens."""
    if torch is None:
        raise RuntimeError("torch is required for graph_recon_loss")
    m = feature_mask.bool().unsqueeze(-1)
    if m.sum() == 0:
        return recon.sum() * 0
    diff = (recon - target) ** 2
    return (diff * m).sum() / (m.sum() * recon.shape[-1])


def masked_path_loss(recon: torch.Tensor, target: torch.Tensor, mlm_mask: torch.Tensor) -> torch.Tensor:
    """Reconstruct node features only at *masked* path positions.

    Distinct from graph_recon (which sees all positions): here the target
    features are hidden by the MLM mask, so the model must infer a masked
    node's graph features from its path context.
    """
    if torch is None:
        raise RuntimeError("torch is required for masked_path_loss")
    m = mlm_mask.bool().unsqueeze(-1)
    if m.sum() == 0:
        return recon.sum() * 0
    diff = (recon - target) ** 2
    return (diff * m).sum() / (m.sum() * recon.shape[-1])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    extra: dict | None = None,
) -> None:
    if torch is None:
        raise RuntimeError("torch is required for save_checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> int:
    if torch is None:
        raise RuntimeError("torch is required for load_checkpoint")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("torch_rng") is not None:
        torch.set_rng_state(ckpt["torch_rng"])
    return int(ckpt.get("step", 0))


def write_health_report(path: Path, registry: LossRegistry) -> None:
    path.write_text(json.dumps({k: vars(v) for k, v in registry.health.items()}, indent=2), encoding="utf-8")
