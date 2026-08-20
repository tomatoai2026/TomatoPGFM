#!/usr/bin/env python
"""Analyze graph-input sensitivity across the five curriculum stages.

The script reads paired per-batch graph-on, graph-shuffle and graph-off losses,
checks their means against the stage report, computes 1,000 paired percentile-
bootstrap resamples, applies BH-FDR to the predefined comparisons and writes
machine-readable results and summary figures.
"""
from __future__ import annotations
import sys, json, os, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROD = Path(os.environ.get("TOMATOPGFM_REPO", Path(__file__).resolve().parents[1]))
PERBATCH_DIR = Path(os.environ.get(
    "TOMATOPGFM_M0_PERBATCH",
    PROD / "results/graph_sensitivity/per_batch",
))
FROZEN_REPORT = PROD / "results/graph_sensitivity/stage_probe_report.json"
OUT_REPORT = PROD / "results/graph_sensitivity/M0_BOOTSTRAP_REPORT_rerun.md"
OUT_JSON = PROD / "results/graph_sensitivity/m0_bootstrap_rerun.json"
OUT_FIG_DIR = PROD / "results/graph_sensitivity/figures_rerun"

STAGES = ["p1_512", "p2_1024", "p3_2048", "p4_4096", "p5_8192"]
STRATA = ["wild", "cult"]
COMPARISONS = ["on_minus_off", "on_minus_shuffle"]
N_BOOT = 1000
SEED = 1234
ALPHA = 0.05


def load_frozen_ref(frozen_path: Path) -> dict:
    """Load the reference stage report used for consistency checks."""
    return {
        e["stage"]: e
        for e in json.loads(frozen_path.read_text(encoding="utf-8"))["entries"]
    }


def repro_gate(stage: str, stratum: str, npz_path: Path, frozen: dict, tol=5e-4) -> dict:
    """Compare full-precision per-batch means with the rounded stage report."""
    if not npz_path.exists():
        return {"pass": False, "note": f"npz 不存在: {npz_path}"}
    if stage not in frozen:
        return {"pass": False, "note": f"reference report has no {stage}"}
    
    data = np.load(npz_path)
    on_mean = float(data["on"].mean())
    off_mean = float(data["off"].mean())
    shuf_mean = float(data["shuffle"].mean())
    
    fr_key = "wild_HEADLINE" if stratum == "wild" else "cult_corroboration"
    fr = frozen[stage]["per_stratum"][fr_key]
    frozen_on = fr["on"]["mean"]
    frozen_off = fr["off"]["mean"]
    frozen_shuf = fr["shuffle"]["mean"]
    
    diffs = {
        "on": abs(on_mean - frozen_on),
        "off": abs(off_mean - frozen_off),
        "shuffle": abs(shuf_mean - frozen_shuf)
    }
    passed = all(d < tol for d in diffs.values())
    max_diff = max(diffs.values())

    return {
        "pass": passed,
        "max_diff": max_diff,
        "tol": tol,
        "note": "PASS" if passed else f"mean 不吻合 (max_diff={max_diff:.2e})",
        "frozen_means": {"on": frozen_on, "off": frozen_off, "shuffle": frozen_shuf},
        "rerun_means": {"on": on_mean, "off": off_mean, "shuffle": shuf_mean},
        "diffs": diffs
    }


def bootstrap_paired_delta(on: np.ndarray, off: np.ndarray, n_boot=N_BOOT, seed=SEED) -> dict:
    """配对 bootstrap: 有放回重采样 per-batch,算 mean(on−off)。
    返回 {"mean", "ci95": [lo, hi], "boot_dist": array(n_boot)}
    """
    rng = np.random.default_rng(seed)
    n = len(on)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = (on[idx] - off[idx]).mean()
    
    mean = float(boot.mean())
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"mean": mean, "ci95": [ci_lo, ci_hi], "boot_dist": boot.tolist()}


def bh_fdr_correction(pvals: list[float], alpha=ALPHA) -> list[float]:
    """BH-FDR 校正: 返回每个 p 的 adjusted p (p_BH)。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    bh = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        k = m - rank
        val = min(prev, pvals[idx] * m / k)
        bh[idx] = val
        prev = val
    return bh


def paired_t_test(on: np.ndarray, off: np.ndarray) -> float:
    """配对 t 检验,返回双侧 p 值。"""
    delta = on - off
    t_stat, p_val = stats.ttest_rel(on, off)
    return float(p_val)


def process_stage(stage: str, frozen: dict) -> dict:
    """处理单档: 复现门 + bootstrap + 配对 t。"""
    result = {"stage": stage, "strata": {}}
    
    for stratum in STRATA:
        npz_path = PERBATCH_DIR / f"{stage}_{stratum}.npz"
        gate = repro_gate(stage, stratum, npz_path, frozen)
        
        if not gate["pass"]:
            result["strata"][stratum] = {"repro_gate": gate, "status": "REPRO_FAIL"}
            continue
        
        data = np.load(npz_path)
        on, off, shuf = data["on"], data["off"], data["shuffle"]
        
        # bootstrap on−off
        boot_off = bootstrap_paired_delta(on, off, n_boot=N_BOOT, seed=SEED)
        boot_shuf = bootstrap_paired_delta(on, shuf, n_boot=N_BOOT, seed=SEED + 1)
        
        # paired t
        p_off = paired_t_test(on, off)
        p_shuf = paired_t_test(on, shuf)
        
        # 符号一致率
        sign_off = float((on < off).mean())
        sign_shuf = float((on < shuf).mean())
        
        result["strata"][stratum] = {
            "repro_gate": gate,
            "status": "OK",
            "n_batch": len(on),
            "on_minus_off": {
                "mean": boot_off["mean"],
                "ci95": boot_off["ci95"],
                "p_raw": p_off,
                "sign_consistency": sign_off,
                "boot_dist": boot_off["boot_dist"]
            },
            "on_minus_shuffle": {
                "mean": boot_shuf["mean"],
                "ci95": boot_shuf["ci95"],
                "p_raw": p_shuf,
                "sign_consistency": sign_shuf,
                "boot_dist": boot_shuf["boot_dist"]
            }
        }
    
    return result


def compact_console_result(result: dict) -> dict:
    """Remove bootstrap draws from smoke-test console output only."""
    compact = json.loads(json.dumps(result))
    for stratum in compact.get("strata", {}).values():
        for comparison in COMPARISONS:
            stratum.get(comparison, {}).pop("boot_dist", None)
    return compact


def run_batch_all(frozen: dict) -> list[dict]:
    """批量处理 5 档,返回结果列表。"""
    results = []
    for stage in STAGES:
        print(f"=== {stage} ===", flush=True)
        res = process_stage(stage, frozen)
        results.append(res)
        # 检查复现门
        for stratum in STRATA:
            st = res["strata"].get(stratum, {})
            status = st.get("status", "UNKNOWN")
            if status != "OK":
                print(f"  {stratum}: {status}", flush=True)
            else:
                d_off = st["on_minus_off"]
                print(
                    f"  {stratum}: delta(on-off)={d_off['mean']:.5f} "
                    f"CI95={d_off['ci95']}",
                    flush=True,
                )
    return results


def apply_bh_fdr(results: list[dict]) -> list[dict]:
    """对所有比较应用 BH-FDR,更新每个 result。"""
    # 收集所有 p_raw
    p_list = []
    for res in results:
        for stratum in STRATA:
            st = res["strata"].get(stratum, {})
            if st.get("status") != "OK":
                continue
            for comp in ["on_minus_off", "on_minus_shuffle"]:
                p_list.append(st[comp]["p_raw"])
    
    # BH 校正
    p_bh = bh_fdr_correction(p_list, alpha=ALPHA)
    
    # 回填
    idx = 0
    for res in results:
        for stratum in STRATA:
            st = res["strata"].get(stratum, {})
            if st.get("status") != "OK":
                continue
            for comp in ["on_minus_off", "on_minus_shuffle"]:
                st[comp]["p_BH"] = p_bh[idx]
                idx += 1
    
    return results


def plot_figures(results: list[dict]):
    """生成三类图表: 趋势图 + p 值热图 + 符号一致率条图。"""
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    stages_clean = [s.replace("p", "Stage").replace("_", " ") for s in STAGES]
    
    # 1. Δon−off 趋势图 (wild headline)
    means, ci_lo, ci_hi = [], [], []
    for res in results:
        st = res["strata"].get("wild", {})
        if st.get("status") == "OK":
            d = st["on_minus_off"]
            means.append(d["mean"])
            ci_lo.append(d["ci95"][0])
            ci_hi.append(d["ci95"][1])
        else:
            means.append(np.nan)
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(STAGES))
    ax.plot(x, means, marker='o', label='Mean Δ(on−off)', color='#C4612F')
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color='#C4612F', label='95% CI (bootstrap)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages_clean)
    ax.set_ylabel('Δ MLM Loss (on − off)', fontsize=11)
    ax.set_xlabel('Training Stage', fontsize=11)
    ax.set_title('M0 Graph Sensitivity: Δ(graph-on − graph-off) across stages', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "m0_delta_trend_wild.png", dpi=150)
    plt.close(fig)
    
    # 2. p_BH 热图
    p_matrix = np.full((len(STAGES), 4), np.nan)  # 5档 × 4列(wild_off, wild_shuf, cult_off, cult_shuf)
    for i, res in enumerate(results):
        col = 0
        for stratum in STRATA:
            st = res["strata"].get(stratum, {})
            if st.get("status") == "OK":
                p_matrix[i, col] = st["on_minus_off"]["p_BH"]
                p_matrix[i, col+1] = st["on_minus_shuffle"]["p_BH"]
            col += 2
    
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.log10(p_matrix + 1e-320), cmap='RdYlGn_r', aspect='auto', vmin=-50, vmax=-1)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['wild\non−off', 'wild\non−shuf', 'cult\non−off', 'cult\non−shuf'], fontsize=9)
    ax.set_yticks(np.arange(len(STAGES)))
    ax.set_yticklabels(stages_clean, fontsize=9)
    ax.set_title('M0 BH-FDR corrected p-values (log10)', fontsize=12, weight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(p_BH)', fontsize=10)
    for i in range(len(STAGES)):
        for j in range(4):
            val = p_matrix[i, j]
            if not np.isnan(val):
                text = f"{val:.1e}" if val > 1e-10 else "~0"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if val < 1e-5 else "black")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "m0_pval_heatmap.png", dpi=150)
    plt.close(fig)
    
    # 3. 符号一致率条图 (wild on−off)
    sign_rates = []
    for res in results:
        st = res["strata"].get("wild", {})
        if st.get("status") == "OK":
            sign_rates.append(st["on_minus_off"]["sign_consistency"])
        else:
            sign_rates.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#C4612F' if r >= 0.95 else '#E7E1D7' for r in sign_rates]
    ax.bar(x, sign_rates, color=colors, alpha=0.8)
    ax.axhline(0.95, color='gray', linestyle='--', linewidth=0.8, label='95% threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages_clean)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Sign Consistency (on < off)', fontsize=11)
    ax.set_xlabel('Training Stage', fontsize=11)
    ax.set_title('M0 Sign Consistency: fraction of batches where on < off', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "m0_sign_consistency_wild.png", dpi=150)
    plt.close(fig)
    
    print(f"\n图表已生成 -> {OUT_FIG_DIR}/", flush=True)


def write_report(results: list[dict]):
    """生成 markdown 报告。"""
    lines = ["# Graph-input sensitivity: paired bootstrap and BH-FDR\n"]
    lines.append(f"**生成时间**: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**方法**: 配对 bootstrap (n={N_BOOT}), BH-FDR 校正 (α={ALPHA}), 双侧 percentile 95% CI\n")
    lines.append(f"**比较数**: {len(STAGES)} 档 × {len(STRATA)} 层 × {len(COMPARISONS)} 组 = {len(STAGES)*len(STRATA)*len(COMPARISONS)} 对\n")
    
    lines.append("\n## 复现门检查\n")
    lines.append("Per-batch means were checked against g5_stage_probe_report.FROZEN.json with tolerance 5e-4.\n\n")
    lines.append("| Stage | Stratum | max_diff | 判定 |\n")
    lines.append("|-------|---------|----------|------|\n")
    all_pass = True
    for res in results:
        for stratum in STRATA:
            st = res["strata"].get(stratum, {})
            gate = st.get("repro_gate", {})
            md = gate.get("max_diff", None)
            passed = gate.get("pass", False)
            md_str = f"{md:.2e}" if isinstance(md, (int, float)) else "—"
            lines.append(f"| {res['stage']} | {stratum} | {md_str} | {'✅ PASS' if passed else '❌ FAIL'} |\n")
            if not passed:
                all_pass = False
    if all_pass:
        lines.append("\nAll stage/stratum means passed the reference-report consistency check.\n")
    else:
        lines.append("\nOne or more stage/stratum means failed the reference-report consistency check.\n")
    
    lines.append("\n## 主结果表 (wild headline)\n")
    lines.append("| Stage | Δ(on−off) | 95% CI | p_BH | Sign% | Δ(on−shuf) | 95% CI | p_BH |\n")
    lines.append("|-------|-----------|--------|------|-------|------------|--------|------|\n")
    for res in results:
        st = res["strata"].get("wild", {})
        if st.get("status") == "OK":
            d_off = st["on_minus_off"]
            d_shuf = st["on_minus_shuffle"]
            lines.append(f"| {res['stage']} | {d_off['mean']:.5f} | [{d_off['ci95'][0]:.5f}, {d_off['ci95'][1]:.5f}] | "
                        f"{d_off['p_BH']:.2e} | {d_off['sign_consistency']*100:.1f}% | "
                        f"{d_shuf['mean']:.5f} | [{d_shuf['ci95'][0]:.5f}, {d_shuf['ci95'][1]:.5f}] | {d_shuf['p_BH']:.2e} |\n")
        else:
            lines.append(f"| {res['stage']} | FAIL | — | — | — | — | — | — |\n")
    
    lines.append("\n## 统计结论\n")
    all_sig = all(
        res["strata"].get("wild", {}).get("on_minus_off", {}).get("p_BH", 1.0) < ALPHA
        for res in results
        if res["strata"].get("wild", {}).get("status") == "OK"
    )
    if all_sig:
        lines.append(f"✅ **全部 {len(STAGES)} 档在 wild headline 层的 Δ(on−off) 通过 BH-FDR (p_BH < {ALPHA})**\n")
        lines.append("Sequence-aligned graph input produced lower MLM loss across the evaluated stages.\n")
    else:
        lines.append("⚠️ 部分档未通过 BH-FDR\n")
    
    lines.append("\n## 图表\n")
    lines.append("- `m0_delta_trend_wild.png`: Δ(on−off) 趋势 + 95% CI\n")
    lines.append("- `m0_pval_heatmap.png`: BH 校正 p 值热图\n")
    lines.append("- `m0_sign_consistency_wild.png`: 符号一致率条图\n")
    
    lines.append("\n---\n")
    lines.append("**原始数据**: `runs/stage5/m0_bootstrap_raw.json` (bootstrap 分布/CI/p_BH,可复核)\n")
    
    OUT_REPORT.write_text("".join(lines), encoding="utf-8")
    print(f"\n报告已生成 -> {OUT_REPORT}", flush=True)


def main():
    global PERBATCH_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="", help="单档 smoke: p1_512 等")
    ap.add_argument("--batch-all", action="store_true", help="批量 5 档")
    ap.add_argument("--frozen-report", default=str(FROZEN_REPORT), help="reference stage-report path")
    ap.add_argument("--perbatch-dir", default=str(PERBATCH_DIR), help="directory containing the ten paired NPZ files")
    args = ap.parse_args()

    PERBATCH_DIR = Path(args.perbatch_dir)
    
    frozen = load_frozen_ref(Path(args.frozen_report))
    
    if args.stage:
        print(f"=== 单档 smoke: {args.stage} ===", flush=True)
        res = process_stage(args.stage, frozen)
        print(json.dumps(compact_console_result(res), indent=2))
        for stratum in STRATA:
            st = res["strata"].get(stratum, {})
            print(f"\n{stratum}: {st.get('status')}")
            if st.get("status") == "OK":
                gate = st["repro_gate"]
                print(f"  复现门: {gate['note']}")
                print(
                    f"  delta(on-off): mean={st['on_minus_off']['mean']:.5f}, "
                    f"CI95={st['on_minus_off']['ci95']}"
                )
    elif args.batch_all:
        print("=== 批量 5 档 ===", flush=True)
        results = run_batch_all(frozen)
        results = apply_bh_fdr(results)
        OUT_JSON.write_text(
            json.dumps({"results": results, "n_boot": N_BOOT, "alpha": ALPHA}, indent=2),
            encoding="utf-8",
        )
        print(f"\n原始数据 -> {OUT_JSON}", flush=True)
        plot_figures(results)
        write_report(results)
    else:
        print("FATAL: 须给 --stage 或 --batch-all", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
