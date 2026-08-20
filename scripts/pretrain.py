#!/usr/bin/env python
"""Five-stage distributed pretraining launcher for TomatoPGFM.

The active objectives and their weights are read from the training YAML. Tasks
with non-positive weights are excluded from loss aggregation and health checks.
`LazyShardDataset` stores byte offsets rather than materializing every JSONL row,
allowing the five curriculum shards to be streamed independently by each rank.

DDP launch (4x A100-80GB):
    cd /path/to/TomatoPGFM
    torchrun --nproc_per_node=4 scripts/pretrain.py \
        --run-name tomatopgfm_pretraining [--max-steps-per-stage N] [--smoke ...]

Resume:
    --resume {RUN_DIR}/ckpt/latest.pt   (restores stage/epoch/step/opt/sched/rng)

This file deliberately does NOT fire anything by itself; it is a normal program.
Use `--smoke` and `--max-steps-per-stage` for interface checks before a full run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from tomatopgfm.config import load_model_config
from tomatopgfm.dataset import collate
from tomatopgfm.dataset_lazy import LazyShardDataset
from tomatopgfm.model import TomatoPGFM
from tomatopgfm.trainer import TrainArgs, build_optimizer, build_scheduler, compute_losses
from tomatopgfm.training import LossRegistry, save_checkpoint, load_checkpoint

PROD = Path(os.environ.get("TOMATOPGFM_REPO", Path(__file__).resolve().parents[1]))
DEFAULT_SHARD_ROOT = Path(os.environ.get("TOMATOPGFM_SHARD_ROOT", PROD / "data/shards"))
DEFAULT_TRAIN_YAML = PROD / "configs/train_final.yaml"
DEFAULT_MODEL_YAML = PROD / "configs/model_final.yaml"
# 512-stage shard predates the seqN naming convention; allow aliasing it.
LEGACY_SEQ512_SHARD = DEFAULT_SHARD_ROOT / "phaseC_pan66_full_shard_v1/pan66_full_shard.jsonl"


# ----------------------------------------------------------------------------
# DDP / logging helpers
# ----------------------------------------------------------------------------
def init_ddp() -> tuple[int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend, timeout=timedelta(hours=2))
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world, local_rank, True
    return 0, 1, 0, False


def make_logger(is_main: bool):
    def log(msg: str):
        if is_main:
            print(f"[{time.strftime('%H:%M:%S')}][rank0] {msg}", flush=True)
    return log


# ----------------------------------------------------------------------------
# Stage / shard resolution
# ----------------------------------------------------------------------------
def stage_shard_path(seq_len: int, shard_root: Path, seq512_override: Path | None) -> Path:
    if seq_len == 512:
        if seq512_override is not None:
            return seq512_override
        cand = shard_root / "phaseC_pan66_full_shard_seq512_v1/pan66_full_shard.jsonl"
        return cand if cand.exists() else LEGACY_SEQ512_SHARD
    return shard_root / f"phaseC_pan66_full_shard_seq{seq_len}_v1/pan66_full_shard.jsonl"


def resolve_stages(train_cfg: dict, args, log) -> list[dict]:
    """Return the curriculum as a list of stage dicts with resolved shard paths.

    In --smoke mode the stages are overridden by --smoke-stages (seq lens) and
    every stage points at the provided --smoke-shard, so launcher logic is
    exercised without the real (possibly not-yet-cut) full shards.
    """
    if args.smoke:
        seqs = [int(s) for s in args.smoke_stages.split(",")]
        shard_list = [p.strip() for p in args.smoke_shard.split(",") if p.strip()]
        if len(shard_list) == 1:
            shard_list = shard_list * len(seqs)
        if len(shard_list) != len(seqs):
            raise ValueError(f"--smoke-shard count {len(shard_list)} != --smoke-stages count {len(seqs)}")
        stages = [{"name": f"smoke_{s}", "seq_len": s, "epochs": args.smoke_epochs,
                   "shard": Path(shard_list[i])} for i, s in enumerate(seqs)]
        log(f"SMOKE mode: stages={[(s['name'], str(s['shard'])) for s in stages]}")
        return stages
    stages = []
    for st in train_cfg["stages"]:
        shard = stage_shard_path(int(st["seq_len"]), Path(args.shard_root),
                                 Path(args.seq512_shard) if args.seq512_shard else None)
        stages.append({"name": st["name"], "seq_len": int(st["seq_len"]),
                       "epochs": int(st["epochs"]), "shard": shard})
    return stages


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="tomatopgfm_pretraining")
    ap.add_argument("--train-yaml", default=str(DEFAULT_TRAIN_YAML))
    ap.add_argument("--model-yaml", default=str(DEFAULT_MODEL_YAML))
    ap.add_argument("--shard-root", default=str(DEFAULT_SHARD_ROOT))
    ap.add_argument("--seq512-shard", default="", help="explicit path for the 512-stage shard (else auto)")
    ap.add_argument("--out-root", default=str(PROD / "runs"))
    ap.add_argument("--per-rank-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-steps-per-stage", type=int, default=-1, help="cap steps/stage (<=0 = full epochs)")
    ap.add_argument("--resume", default="", help="path to latest.pt to resume from")
    # smoke (阶段3-B / 阶段4): exercise launcher logic on tiny real shards
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-shard", default="")
    ap.add_argument("--smoke-stages", default="512,8192")
    ap.add_argument("--smoke-epochs", type=int, default=1)
    ap.add_argument("--smoke-steps-per-stage", type=int, default=6)
    args = ap.parse_args()

    rank, world, local_rank, ddp = init_ddp()
    is_main = rank == 0
    log = make_logger(is_main)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---- config (single source of truth) ----
    train_cfg = yaml.safe_load(Path(args.train_yaml).read_text())
    weights = train_cfg["fixed_loss_weights"]
    active_tasks = [k for k, v in weights.items() if v > 0]
    disabled_tasks = [k for k, v in weights.items() if v <= 0]
    collapse_patience = int(train_cfg.get("collapse_patience", 3))
    ckpt_every_steps = int(train_cfg.get("checkpoint_every_steps", 5000))
    ckpt_every_epoch = bool(train_cfg.get("checkpoint_every_epoch", True))
    peak_lr = float(train_cfg.get("peak_lr", 3e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-5))
    warmup_steps = int(train_cfg.get("warmup_steps", 12000))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    precision = train_cfg.get("precision", "bf16")

    if args.smoke:
        warmup_steps = max(1, args.smoke_steps_per_stage // 2)

    log(f"run={args.run_name} world={world} device={device}")
    log(f"loss weights = {weights}")
    log(f"ACTIVE tasks (weight>0, collapse-guarded) = {active_tasks}")
    log(f"DISABLED tasks (weight<=0, logging-only, NEVER collapse-trip) = {disabled_tasks}")

    # ---- stages ----
    stages = resolve_stages(train_cfg, args, log)

    # validate shard existence up-front (fail loud, not mid-stage)
    missing = [str(s["shard"]) for s in stages if not Path(s["shard"]).exists()]
    if missing:
        log(f"FATAL: missing stage shards: {missing}")
        if ddp:
            dist.barrier(); dist.destroy_process_group()
        sys.exit(2)

    # ---- model ----
    cfg = load_model_config(args.model_yaml)
    model = TomatoPGFM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params = {n_params/1e6:.2f}M | max_seq_len={cfg.max_seq_len}")
    if ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=True)

    # ---- total step budget across all stages (for one global cosine schedule) ----
    # Precompute steps/stage so warmup+cosine spans the WHOLE curriculum, not reset
    # per-stage. steps/stage = ceil(len(shard) / (world*batch*accum)) * epochs.
    stage_steps = []
    for st in stages:
        n_ex = len(LazyShardDataset(st["shard"]))
        steps_per_epoch = max(1, n_ex // (world * args.per_rank_batch * args.grad_accum))
        s_steps = steps_per_epoch * st["epochs"]
        if args.smoke:
            s_steps = args.smoke_steps_per_stage
        elif args.max_steps_per_stage > 0:
            s_steps = min(s_steps, args.max_steps_per_stage)
        stage_steps.append(s_steps)
        log(f"  stage {st['name']}: examples={n_ex} steps/epoch={steps_per_epoch} "
            f"epochs={st['epochs']} -> steps={s_steps}")
    total_steps = sum(stage_steps)
    log(f"TOTAL optimizer steps across curriculum = {total_steps}")

    # ---- optimizer + ONE global scheduler ----
    base_args = TrainArgs(peak_lr=peak_lr, min_lr=min_lr, warmup_steps=warmup_steps,
                          total_steps=total_steps, weight_decay=weight_decay,
                          grad_clip=grad_clip, grad_accum=args.grad_accum,
                          precision=precision, graph_mode="on")
    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = build_optimizer(model, base_args)
    scheduler = build_scheduler(optimizer, base_args)
    registry = LossRegistry(weights)

    # ---- output dirs ----
    run_dir = Path(args.out_root) / args.run_name
    ckpt_dir = run_dir / "ckpt"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_snapshot.json").write_text(json.dumps({
            "weights": weights, "active_tasks": active_tasks, "disabled_tasks": disabled_tasks,
            "stages": [{"name": s["name"], "seq_len": s["seq_len"], "epochs": s["epochs"],
                        "shard": str(s["shard"]), "steps": stage_steps[i]}
                       for i, s in enumerate(stages)],
            "total_steps": total_steps, "train_args": asdict(base_args),
            "world": world, "per_rank_batch": args.per_rank_batch, "grad_accum": args.grad_accum,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- resume ----
    global_step = 0
    start_stage = 0
    resume_opt_steps_in_stage = 0
    if args.resume:
        rp = Path(args.resume)
        if not rp.exists():
            log(f"FATAL: --resume not found: {rp}")
            if ddp:
                dist.barrier(); dist.destroy_process_group()
            sys.exit(2)
        ckpt = torch.load(rp, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("torch_rng") is not None:
            torch.set_rng_state(ckpt["torch_rng"])
        ex = ckpt.get("extra", {})
        global_step = int(ckpt.get("step", 0))
        start_stage = int(ex.get("stage_idx", 0))
        resume_opt_steps_in_stage = int(ex.get("opt_steps_in_stage", 0))
        # If the resumed stage was already fully done (stage-end ckpt), advance to
        # the next stage; otherwise resume the REMAINDER of that stage.
        if start_stage < len(stage_steps) and resume_opt_steps_in_stage >= stage_steps[start_stage]:
            start_stage += 1
            resume_opt_steps_in_stage = 0
        log(f"RESUMED from {rp}: global_step={global_step} -> start_stage={start_stage} "
            f"opt_steps_already_in_stage={resume_opt_steps_in_stage}")

    # ---- curriculum loop ----
    health_report = {}
    for si in range(start_stage, len(stages)):
        st = stages[si]
        ds = LazyShardDataset(st["shard"])
        if ddp:
            sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        else:
            sampler = None
        loader = DataLoader(ds, batch_size=args.per_rank_batch, sampler=sampler,
                            shuffle=(sampler is None), collate_fn=collate,
                            drop_last=True, num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)
        log(f"=== STAGE {si} {st['name']} seq_len={st['seq_len']} examples={len(ds)} "
            f"target_steps={stage_steps[si]} ===")

        stage_target = stage_steps[si]   # in OPTIMIZER steps
        # On the first (resumed) stage, pick up where we left off; else from 0.
        opt_steps_this_stage = resume_opt_steps_in_stage if si == start_stage else 0
        resume_opt_steps_in_stage = 0  # consumed
        micro_in_accum = 0
        epoch = 0
        model.train()
        peak_mem = 0.0
        while opt_steps_this_stage < stage_target:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                if opt_steps_this_stage >= stage_target:
                    break
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                      if (precision == "bf16" and device.type == "cuda")
                      else torch.autocast(device_type=device.type, enabled=False)):
                    losses = compute_losses(model, batch, base_args)
                    moe_aux = losses.pop("moe_aux")
                    total = (registry.combine(losses) + moe_aux) / args.grad_accum
                total.backward()
                micro_in_accum += 1

                # optimizer step boundary every grad_accum micro-batches
                if micro_in_accum >= args.grad_accum:
                    micro_in_accum = 0
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    opt_steps_this_stage += 1

                    # health: ONLY active (weight>0) tasks can trip collapse (治本)
                    scalar = {k: float(v.detach()) for k, v in losses.items()}
                    registry.update_health({k: scalar[k] for k in active_tasks if k in scalar},
                                           patience_override=collapse_patience)
                    collapsed_active = [k for k in active_tasks
                                        if registry.health[k].collapsed]
                    if collapsed_active:
                        log(f"!! COLLAPSE in ACTIVE task(s) {collapsed_active} at step {global_step} "
                            f"-> saving emergency ckpt and aborting")
                        if is_main:
                            save_stage_ckpt(ckpt_dir / "collapse.pt", raw_model, optimizer,
                                            scheduler, global_step, si, opt_steps_this_stage, st)
                        if ddp:
                            dist.barrier(); dist.destroy_process_group()
                        sys.exit(3)

                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9
                                   if torch.cuda.is_available() else 0.0)
                    if is_main and (global_step % args.log_every == 0):
                        log(f"stage{si} {st['name']} step{global_step}/{total_steps} "
                            f"ep{epoch} lr={scheduler.get_last_lr()[0]:.2e} "
                            f"gn={float(grad_norm):.2f} "
                            + " ".join(f"{k}={scalar.get(k, float('nan')):.4f}" for k in weights)
                            + f" mem={peak_mem:.1f}G")
                    if is_main and ckpt_every_steps and global_step % ckpt_every_steps == 0:
                        save_stage_ckpt(ckpt_dir / "latest.pt", raw_model, optimizer,
                                        scheduler, global_step, si, opt_steps_this_stage, st)
            epoch += 1
            if is_main and ckpt_every_epoch:
                save_stage_ckpt(ckpt_dir / "latest.pt", raw_model, optimizer,
                                scheduler, global_step, si, opt_steps_this_stage, st)
        # end of stage checkpoint
        if is_main:
            save_stage_ckpt(ckpt_dir / f"stage{si}_{st['name']}_end.pt", raw_model, optimizer,
                            scheduler, global_step, si, stage_target, st)
            save_stage_ckpt(ckpt_dir / "latest.pt", raw_model, optimizer,
                            scheduler, global_step, si, stage_target, st)
        health_report = {k: vars(v) for k, v in registry.health.items()}
        log(f"=== STAGE {si} {st['name']} DONE at global_step={global_step} peak_mem={peak_mem:.1f}G ===")

    # ---- final report ----
    if is_main:
        report = {
            "run_name": args.run_name, "smoke": args.smoke,
            "world": world, "total_steps_target": total_steps, "final_global_step": global_step,
            "weights": weights, "active_tasks": active_tasks, "disabled_tasks": disabled_tasks,
            "stages": [{"name": s["name"], "seq_len": s["seq_len"], "steps": stage_steps[i]}
                       for i, s in enumerate(stages)],
            "health": health_report,
            "collapsed_active_tasks": [k for k in active_tasks if health_report.get(k, {}).get("collapsed")],
            "elapsed_min": round((time.time() - t0) / 60, 2),
        }
        (run_dir / "g5_run_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"FINAL report -> {run_dir/'g5_run_report.json'}")
        print(json.dumps({k: v for k, v in report.items() if k != "health"},
                         indent=2, ensure_ascii=False), flush=True)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def save_stage_ckpt(path: Path, model, optimizer, scheduler, step, stage_idx, opt_steps_in_stage, stage):
    save_checkpoint(path, model, optimizer, step, scheduler,
                    extra={"stage_idx": stage_idx, "opt_steps_in_stage": opt_steps_in_stage,
                           "stage_name": stage["name"], "seq_len": stage["seq_len"]})


if __name__ == "__main__":
    main()
