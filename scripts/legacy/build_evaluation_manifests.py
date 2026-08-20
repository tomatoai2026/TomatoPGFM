#!/usr/bin/env python
"""Build chromosome-partitioned external-accession evaluation manifests.

LA1974 and MicroTom windows retain genome and window-size strata. Every
chromosome is assigned wholly to one split so locally adjacent windows cannot
be divided between training and test partitions.
"""
import argparse, json, hashlib, os
from pathlib import Path
from collections import defaultdict, Counter

RUN = Path(os.environ.get("TOMATOPGFM_RUN_ROOT", Path(__file__).resolve().parents[1] / "runs"))
M1_DIR = RUN / "m1_holdout"
OUT = RUN / "eval_manifests"

# Assign each chromosome wholly to one split.
# 用确定性 hash 排序避免人为偏置, 但保证 train/test 不混染色体.
def assign_chrom_split(chroms, test_frac=0.2, seed="v432"):
    ordered = sorted(chroms)
    n_test = max(1, round(len(ordered) * test_frac))
    # 按 hash 确定性挑 test 染色体 (可复现)
    scored = sorted(ordered, key=lambda c: hashlib.md5((seed + c).encode()).hexdigest())
    test_set = set(scored[:n_test])
    return {c: ("test" if c in test_set else "train") for c in ordered}


def build_m1_m4a():
    """M1 + M4① 共用一套窗口, 不同视角. 产 manifest 行: 引用窗口文件偏移 + split + 分层标签."""
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    stats = defaultdict(Counter)
    for genome in ["LA1974", "MicroTom"]:
        fn = M1_DIR / f"{genome}_windows.jsonl"
        chroms = set()
        rows = []
        with open(fn) as f:
            for lineno, line in enumerate(f):
                r = json.loads(line)
                chroms.add(r["chrom"])
                rows.append((lineno, r["chrom"], r["window_bp"], r["genome"]))
        split_map = assign_chrom_split(chroms)
        for lineno, chrom, wbp, g in rows:
            split = split_map[chrom]
            manifest.append({
                "module": "M1_M4a",
                "genome": g,
                "source_file": str(fn),
                "line_no": lineno,
                "chrom": chrom,
                "window_bp": wbp,
                "split": split,
                "stratum_genome": g,          # M4① 分层: 按 holdout 基因组
                "stratum_window_bp": wbp,      # M4① 分层: 按窗口长度
            })
            stats[g][f"{split}_{wbp}"] += 1
        # Store the deterministic chromosome assignment with the manifest.
        (OUT / f"M1_split_{genome}.json").write_text(
            json.dumps(split_map, indent=2, ensure_ascii=False))
    (OUT / "M1_M4a_manifest.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in manifest))
    return {"n_manifest": len(manifest), "stats": {k: dict(v) for k, v in stats.items()},
            "split_LA1974": json.loads((OUT / "M1_split_LA1974.json").read_text()),
            "split_MicroTom": json.loads((OUT / "M1_split_MicroTom.json").read_text())}


def build_m2():
    """M2 wild/cult 分层 manifest. 标签来自 66-accession training-panel 的 g4 分层文件（训练面板内分组，非外部留出）."""
    OUT.mkdir(parents=True, exist_ok=True)
    src = {
        "wild": RUN / "g4_heldout_wild.jsonl",
        "cult": RUN / "g4_heldout_cult.jsonl",
    }
    manifest = []
    stats = Counter()
    # 平衡采样: cult 少(3250), wild 多. M2 二分类取等量避免类不平衡偏置.
    counts = {}
    for label, fn in src.items():
        n = sum(1 for _ in open(fn))
        counts[label] = n
    n_per = min(counts.values())  # 平衡到少数类
    for label, fn in src.items():
        # 确定性下采样: 均匀步长
        step = max(1, counts[label] // n_per)
        with open(fn) as f:
            taken = 0
            for i, line in enumerate(f):
                if i % step != 0:
                    continue
                if taken >= n_per:
                    break
                manifest.append({
                    "module": "M2",
                    "label": label,              # wild / cult 二分类目标
                    "label_id": 0 if label == "wild" else 1,
                    "source_file": str(fn),
                    "line_no": i,
                })
                stats[label] += 1
                taken += 1
    # split: 简单 80/20 确定性 (M2 是表征 probe, 按行 hash)
    for m in manifest:
        h = int(hashlib.md5(f"{m['source_file']}:{m['line_no']}".encode()).hexdigest(), 16)
        m["split"] = "test" if h % 5 == 0 else "train"
    (OUT / "M2_manifest.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in manifest))
    split_stats = Counter(f"{m['label']}_{m['split']}" for m in manifest)
    return {"n_manifest": len(manifest), "raw_counts": counts, "balanced_per_class": n_per,
            "label_stats": dict(stats), "split_stats": dict(split_stats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=["m1m4a", "m2", "all"], default="all")
    args = ap.parse_args()
    report = {}
    if args.module in ("m1m4a", "all"):
        report["M1_M4a"] = build_m1_m4a()
    if args.module in ("m2", "all"):
        report["M2"] = build_m2()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
