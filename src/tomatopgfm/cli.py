from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import core_smoke_audit, external_tools, phase0_audit, static_smoke_audit
from .graph import export_graph_report
from .pan66_shard_builder import EXPECTED_TOKENIZER_SHA256, EXPECTED_VOCAB_SIZE


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tomatopgfm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("phase0-audit")
    p0.add_argument("--data-root", required=True)
    p0.add_argument("--out", required=True)

    tools = sub.add_parser("external-tools")

    smoke = sub.add_parser("smoke-model")
    static = sub.add_parser("static-smoke")

    gr = sub.add_parser("graph-report")
    gr.add_argument("--gfa", required=True)
    gr.add_argument("--out", required=True)
    gr.add_argument("--max-segments", type=int, default=None)

    dr = sub.add_parser("dry-run")
    dr.add_argument("--steps", type=int, default=1000)
    dr.add_argument("--batch-size", type=int, default=4)
    dr.add_argument("--seq-len", type=int, default=64)

    bp = sub.add_parser("build-pansn")
    bp.add_argument("--data-root", required=True)
    bp.add_argument("--out", required=True)

    sc = sub.add_parser("show-config")
    sc.add_argument("--config", required=True)

    bs = sub.add_parser("build-shards")
    bs.add_argument("--gfa", required=True)
    bs.add_argument("--out", required=True)
    bs.add_argument("--reference", default="SL6.0")
    bs.add_argument("--max-segments", type=int, default=None)

    p66 = sub.add_parser("build-pan66-shard-pilot")
    p66.add_argument("--gfa", required=True)
    p66.add_argument("--segments", required=True)
    p66.add_argument("--links", required=True)
    p66.add_argument("--features", required=True)
    p66.add_argument("--tokenizer", required=True)
    p66.add_argument("--out", required=True)
    p66.add_argument("--manifest", default=None)
    p66.add_argument("--preflight", default=None)
    p66.add_argument("--seq-len", type=int, default=512)
    p66.add_argument("--pilot-max-segments", type=int, default=10000)
    p66.add_argument("--pilot-token-limit", type=int, default=None)
    p66.add_argument("--pilot-sampling", choices=("prefix", "uniform_blocks"), default="prefix")
    p66.add_argument("--pilot-block-size", type=int, default=1000)
    p66.add_argument("--n-fraction-skip", type=float, default=0.50)
    p66.add_argument("--cpc-reweighted", default=None)
    p66.add_argument("--expected-tokenizer-sha256", default=EXPECTED_TOKENIZER_SHA256)
    p66.add_argument("--expected-vocab-size", type=int, default=EXPECTED_VOCAB_SIZE)
    p66.add_argument("--holdout", action="append", default=["LA1974", "MicroTom", "GCF_036512215", "holdout"])

    abl = sub.add_parser("pilot-graph-ablation")
    abl.add_argument("--shard", required=True)
    abl.add_argument("--out", required=True)
    abl.add_argument("--max-examples", type=int, default=1024)
    abl.add_argument("--steps", type=int, default=20)
    abl.add_argument("--batch-size", type=int, default=4)
    abl.add_argument("--seed", type=int, default=7)
    abl.add_argument("--vocab-size", type=int, default=2085)
    abl.add_argument("--seq-len", type=int, default=512)
    abl.add_argument("--d-model", type=int, default=64)
    abl.add_argument("--n-layers", type=int, default=3)
    abl.add_argument("--modes", default=None)

    sub.add_parser("build-shards-smoke")

    args = parser.parse_args(argv)
    if args.cmd == "phase0-audit":
        print(json.dumps(phase0_audit(Path(args.data_root), Path(args.out)), indent=2))
    elif args.cmd == "external-tools":
        print(json.dumps(external_tools(), indent=2))
    elif args.cmd == "smoke-model":
        print(json.dumps(core_smoke_audit(), indent=2))
    elif args.cmd == "static-smoke":
        print(json.dumps(static_smoke_audit(), indent=2))
    elif args.cmd == "graph-report":
        print(json.dumps(export_graph_report(Path(args.gfa), Path(args.out), args.max_segments), indent=2))
    elif args.cmd == "dry-run":
        from .trainer import dry_run

        print(json.dumps(dry_run(steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len), indent=2))
    elif args.cmd == "build-pansn":
        from .minigraph import build_train_pansn

        print(json.dumps(build_train_pansn(Path(args.data_root), Path(args.out)), indent=2))
    elif args.cmd == "show-config":
        from dataclasses import asdict

        from .config import load_model_config

        print(json.dumps(asdict(load_model_config(Path(args.config))), indent=2))
    elif args.cmd == "build-shards":
        from .shard_builder import build_shards

        print(json.dumps(build_shards(Path(args.gfa), Path(args.out), reference_accession=args.reference, max_segments=args.max_segments), indent=2))
    elif args.cmd == "build-pan66-shard-pilot":
        from .pan66_shard_builder import build_pan66_shard

        print(json.dumps(build_pan66_shard(args), indent=2))
    elif args.cmd == "pilot-graph-ablation":
        from .pilot_ablation import run_pilot_graph_ablation

        print(json.dumps(run_pilot_graph_ablation(args), indent=2))
    elif args.cmd == "build-shards-smoke":
        from .shard_builder import build_shards_smoke

        print(json.dumps(build_shards_smoke(), indent=2))


if __name__ == "__main__":
    main()
