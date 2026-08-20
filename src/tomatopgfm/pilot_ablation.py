from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import FEATURE_DIM, ShardDataset, collate
from .model import TomatoPGFMConfig, TomatoPGFM
from .training import LossRegistry
from .trainer import TrainArgs, train_loop


def reservoir_sample_jsonl(src: Path, dst: Path, max_examples: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    with src.open("rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            if len(sample) < max_examples:
                sample.append(line)
            else:
                j = rng.randrange(seen)
                if j < max_examples:
                    sample[j] = line
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(sample), encoding="utf-8")
    return {"source_examples_seen": seen, "sampled_examples": len(sample)}


def run_one_mode(
    shard: Path,
    graph_mode: str,
    steps: int,
    batch_size: int,
    seed: int,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
) -> dict:
    torch.manual_seed(seed)
    ds = ShardDataset(shard)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    cfg = TomatoPGFMConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        graph_feature_dim=FEATURE_DIM,
        max_seq_len=seq_len,
        attn_window=min(64, seq_len),
        use_real_mamba=False,
    )
    model = TomatoPGFM(cfg)
    weights = {"mlm": 1.0, "vtp": 0.25, "cpc": 0.0, "masked_path": 0.25, "graph_recon": 0.25}
    registry = LossRegistry(weights)
    args = TrainArgs(
        warmup_steps=max(1, steps // 5),
        total_steps=steps,
        precision="fp32",
        graph_mode=graph_mode,
        graph_noise_std=0.5,
    )
    result = train_loop(model, loader, registry, args, torch.device("cpu"), max_steps=steps, log_every=max(1, steps // 5))
    logs = result.get("logs", [])
    first = logs[0] if logs else {}
    last = logs[-1] if logs else {}
    return {
        "graph_mode": graph_mode,
        "steps": result.get("final_step"),
        "first_log": first,
        "last_log": last,
        "any_collapsed": any(h["collapsed"] for h in result["health"].values()),
    }


def run_pilot_graph_ablation(args: argparse.Namespace) -> dict:
    if args.modes is None:
        modes = ["on", "shuffle", "off"]
    else:
        modes = args.modes.split(",")
    bad = sorted(set(modes) - {"on", "shuffle", "off"})
    if bad:
        raise ValueError(f"Unsupported graph modes: {bad}")
    out = Path(args.out)
    subset = out.with_suffix(".sample.jsonl")
    sample_report = reservoir_sample_jsonl(Path(args.shard), subset, args.max_examples, args.seed)
    mode_reports = [
        run_one_mode(
            shard=subset,
            graph_mode=mode,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
        )
        for mode in modes
    ]
    by_mode = {r["graph_mode"]: r for r in mode_reports}
    on = by_mode.get("on", {}).get("last_log", {})
    shuffle = by_mode.get("shuffle", {}).get("last_log", {})
    off = by_mode.get("off", {}).get("last_log", {})
    graph_recon_on = float(on.get("graph_recon", 0.0))
    graph_recon_shuffle = float(shuffle.get("graph_recon", 0.0))
    graph_recon_off = float(off.get("graph_recon", 0.0))
    masked_path_on = float(on.get("masked_path", 0.0))
    masked_path_shuffle = float(shuffle.get("masked_path", 0.0))
    masked_path_off = float(off.get("masked_path", 0.0))
    report = {
        "stage": "C12B_route_A_pilot_graph_ablation",
        "decision": "PILOT_ABLATION_RESULT_NOT_FULL_SHARD_GATE",
        "shard": str(args.shard),
        "subset_shard": str(subset),
        "sample": sample_report,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "vocab_size": args.vocab_size,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "modes": modes,
        },
        "mode_reports": mode_reports,
        "summary": {
            "cpc_weight": 0.0,
            "cpc_not_used_for_route_A_gate": True,
            "graph_recon_last_on": graph_recon_on,
            "graph_recon_last_shuffle": graph_recon_shuffle,
            "graph_recon_last_off": graph_recon_off,
            "masked_path_last_on": masked_path_on,
            "masked_path_last_shuffle": masked_path_shuffle,
            "masked_path_last_off": masked_path_off,
            "graph_recon_on_better_than_shuffle": graph_recon_on < graph_recon_shuffle,
            "graph_recon_on_better_than_off": graph_recon_on < graph_recon_off,
            "masked_path_on_better_than_shuffle": masked_path_on < masked_path_shuffle,
            "masked_path_on_better_than_off": masked_path_on < masked_path_off,
        },
        "allow_full_shard_generation": False,
        "status": "pass" if all(r.get("steps") == args.steps for r in mode_reports) else "fail",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pilot-graph-ablation")
    p.add_argument("--shard", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-examples", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vocab-size", type=int, default=2085)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--modes", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run_pilot_graph_ablation(args), indent=2))


if __name__ == "__main__":
    main()
