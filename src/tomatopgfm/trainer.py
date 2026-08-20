from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import FEATURE_DIM, ShardDataset, collate
from .model import TomatoPGFMConfig, TomatoPGFM
from .training import (
    LossRegistry,
    cpc_loss,
    graph_recon_loss,
    load_checkpoint,
    masked_path_loss,
    mlm_loss,
    save_checkpoint,
    vtp_loss,
)


@dataclass
class TrainArgs:
    peak_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    total_steps: int = 100000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum: int = 1
    precision: str = "bf16"
    mask_id: int = 1
    graph_noise_std: float = 1.0
    vtp_class_weights: list | None = None
    graph_mode: str = "on"


def build_optimizer(model: nn.Module, args: TrainArgs) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.peak_lr, betas=(0.9, 0.95))


def build_scheduler(optimizer: torch.optim.Optimizer, args: TrainArgs) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay from peak_lr to min_lr."""
    floor = args.min_lr / args.peak_lr

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_init_ddp() -> tuple[int, int, bool]:
    """Initialize DDP from torchrun env vars if present. Returns (rank, world, ddp)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return rank, world, True
    return 0, 1, False


def _autocast(precision: str, device: torch.device):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def compute_losses(model: nn.Module, batch: dict, args: TrainArgs) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"]
    mlm_mask = batch["mlm_mask"]
    feature_mask = batch["feature_mask"]
    graph_features = batch["graph_features"]
    edge_index = batch.get("edge_index")

    # Corrupt masked positions for the MLM objective; keep originals as labels.
    labels = input_ids.clone()
    corrupted = input_ids.clone()
    corrupted[mlm_mask.bool()] = args.mask_id

    # Corrupt the *graph features* the model sees so graph_recon / masked_path
    # become real denoising/inference tasks instead of copying a clean input
    # straight to the reconstruction head (the V5-style self-recon shortcut):
    #   - additive Gaussian noise everywhere  -> denoising (graph_recon)
    #   - zero out features at MLM-masked positions -> the model must infer a
    #     masked node's features from its path/graph context (masked_path)
    # The reconstruction targets stay the *clean* graph_features.
    gf_mask = mlm_mask.unsqueeze(-1).to(graph_features.dtype)
    noisy_gf = graph_features + torch.randn_like(graph_features) * args.graph_noise_std
    corrupted_gf = noisy_gf * (1.0 - gf_mask)

    class_weights = None
    if args.vtp_class_weights is not None:
        class_weights = torch.tensor(args.vtp_class_weights, dtype=torch.float32, device=input_ids.device)

    out = model(corrupted, corrupted_gf, graph_mode=args.graph_mode, edge_index=edge_index)
    return {
        "mlm": mlm_loss(out["mlm_logits"], labels, mlm_mask),
        "vtp": vtp_loss(out["vtp_logits"], batch["variant_class"], class_weights),
        "cpc": cpc_loss(out["path_proj"], batch["cpc_positive_index"], sample_weight=batch.get("sample_weight")),
        "graph_recon": graph_recon_loss(out["graph_recon"], graph_features, feature_mask),
        "masked_path": masked_path_loss(out["graph_recon"], graph_features, mlm_mask),
        "moe_aux": out["moe_aux"],
    }


def train_loop(
    model: nn.Module,
    loader: DataLoader,
    registry: LossRegistry,
    args: TrainArgs,
    device: torch.device,
    max_steps: int,
    ckpt_dir: Path | None = None,
    checkpoint_every: int = 0,
    start_step: int = 0,
    log_every: int = 50,
) -> dict:
    model.train()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    for _ in range(start_step):
        scheduler.step()

    step = start_step
    logs: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    while step < max_steps:
        accum_scalars: dict[str, float] = {}
        for micro in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with _autocast(args.precision, device):
                losses = compute_losses(model, batch, args)
                # MoE aux (load-balance + router z-loss) is a plain regularizer:
                # add it to the objective directly, not via the SSL-task registry
                # / health monitor (it is ~0 without MoE and is not a task that
                # can "collapse").
                moe_aux = losses.pop("moe_aux")
                total = (registry.combine(losses) + moe_aux) / args.grad_accum
            total.backward()
            for k, v in losses.items():
                accum_scalars[k] = accum_scalars.get(k, 0.0) + float(v.detach()) / args.grad_accum
            accum_scalars["moe_aux"] = accum_scalars.get("moe_aux", 0.0) + float(moe_aux.detach()) / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        registry.update_health({k: v for k, v in accum_scalars.items() if k != "moe_aux"})
        step += 1

        if step % log_every == 0 or step == max_steps:
            logs.append({"step": step, "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm), **accum_scalars})
        if ckpt_dir is not None and checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(Path(ckpt_dir) / f"step_{step}.pt", model, optimizer, step, scheduler)

    return {"final_step": step, "logs": logs, "health": {k: vars(v) for k, v in registry.health.items()}}


def dry_run(steps: int = 1000, batch_size: int = 4, seq_len: int = 64, vocab_size: int = 256) -> dict:
    """End-to-end 1000-step style dry-run on synthetic shards.

    Exercises dataloader -> model -> all five losses -> optimizer -> warmup/cosine
    scheduler -> grad clip -> health monitor, so the training plumbing can be
    validated before committing A100 time. Runs on CPU if no GPU is present.
    """
    import tempfile

    from .shards import dummy_examples, write_jsonl_shard

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "shard.jsonl"
        write_jsonl_shard(dummy_examples(n=batch_size * 4, seq_len=seq_len, vocab_size=vocab_size), shard)
        ds = ShardDataset(shard)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

        cfg = TomatoPGFMConfig(vocab_size=vocab_size, d_model=64, n_layers=3,
                               n_heads=4, graph_feature_dim=FEATURE_DIM,
                               max_seq_len=seq_len + 2, attn_window=16,
                               use_real_mamba=False)
        model = TomatoPGFM(cfg).to(device)
        weights = {"mlm": 1.0, "vtp": 0.5, "cpc": 0.5, "masked_path": 0.25, "graph_recon": 0.25}
        registry = LossRegistry(weights)
        args = TrainArgs(warmup_steps=max(1, steps // 10), total_steps=steps, precision="bf16")

        result = train_loop(model, loader, registry, args, device, max_steps=steps, log_every=max(1, steps // 5))

    first = result["logs"][0] if result["logs"] else {}
    last = result["logs"][-1] if result["logs"] else {}
    mlm_dropped = bool(first and last and last.get("mlm", 1e9) < first.get("mlm", -1e9))
    return {
        "device": device.type,
        "steps": result["final_step"],
        "first_log": first,
        "last_log": last,
        "mlm_decreased": mlm_dropped,
        "any_collapsed": any(h["collapsed"] for h in result["health"].values()),
        "status": "pass" if result["final_step"] == steps else "fail",
    }
