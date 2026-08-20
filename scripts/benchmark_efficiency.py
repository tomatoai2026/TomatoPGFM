#!/usr/bin/env python
"""Benchmark TomatoPGFM zero-feature GraphAdapter-on/off efficiency.

At 512, 1,024 and 2,048 model positions, the script records parameter counts,
peak allocated GPU memory, sequence/token throughput, and batch-1 median and
90th-percentile latency. Adapter-on and graph-off use identical synthetic input
batches under FP32 forward-only inference. Adapter-on supplies zero-valued graph
features with edge_index=None, so GraphMessage adjacency aggregation is disabled.
"""
from __future__ import annotations
import sys, os, json, time, gc
from pathlib import Path
import numpy as np
import torch

REPO = Path(os.environ.get("TOMATOPGFM_REPO", Path(__file__).resolve().parents[1]))
ROOT = Path(os.environ.get("TOMATOPGFM_DATA_ROOT", REPO / "data"))
TOK_VOCAB = Path(os.environ.get("TOMATOPGFM_TOKENIZER", REPO / "assets/tokenizer_vocab.json"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src"))

TOMATOPGFM_CHECKPOINT_PATH = Path(os.environ.get("TOMATOPGFM_CHECKPOINT", REPO / "model.safetensors"))
DNABERT2 = os.environ.get("DNABERT2_MODEL", "")
PLANTMAMBA = os.environ.get("PLANTDNAMAMBA_MODEL", "")

SEQLENS = [512, 1024, 2048]
BATCH = 16          # 吞吐测量 batch
N_BATCH = 10        # 计时 batch 数(warmup 后)
N_WARMUP = 3
LAT_N = 30          # 单序列延迟采样数
MASK_BP = 12
SEED = 1234
DEVICE = "cuda"

# smoke 开关: 环境变量 BENCH_SMOKE=1 时缩小规模验证接口
if os.environ.get("BENCH_SMOKE") == "1":
    SEQLENS = [512]
    BATCH = 4
    N_BATCH = 2
    N_WARMUP = 1
    LAT_N = 3


def synth_seqs(n: int, L: int, seed: int) -> list[str]:
    """生成 n 条长 L 的随机 ACGT 序列, 中心 mask MASK_BP 个 N(与 M1 口径一致)。"""
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    seqs = []
    for _ in range(n):
        arr = rng.choice(bases, size=L)
        c = L // 2
        arr[c - MASK_BP // 2: c + MASK_BP // 2] = "N"
        seqs.append("".join(arr))
    return seqs


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


# ---------------- TomatoPGFM ----------------
def load_tomatopgfm():
    from tomatopgfm.config import load_model_config
    from tomatopgfm.checkpoint_io import load_model_weights
    from tomatopgfm.model import TomatoPGFM
    from tomatopgfm.tokenizer import RCKmerTokenizer
    cfg = load_model_config(REPO / "configs/model_final.yaml")
    model = TomatoPGFM(cfg)
    load_model_weights(model, TOMATOPGFM_CHECKPOINT_PATH)
    model.eval().to(DEVICE)
    tok = RCKmerTokenizer.load(TOK_VOCAB)
    return model, tok, cfg.graph_feature_dim


def tomatopgfm_forward_batch(model, tok, gfdim, seqs, L, mode):
    ids_list = [tok.encode(s, L) for s in seqs]
    Lmax = max(len(x) for x in ids_list)
    ids = torch.zeros(len(seqs), Lmax, dtype=torch.long)
    for j, x in enumerate(ids_list):
        ids[j, :len(x)] = torch.tensor(x)
    ids = ids.to(DEVICE)
    # mode="on" enables the GraphAdapter software path with zero-valued
    # features. edge_index=None keeps GraphMessage adjacency aggregation disabled.
    gf = torch.zeros(len(seqs), Lmax, gfdim, device=DEVICE)
    with torch.no_grad():
        _ = model(ids, gf, mode, edge_index=None)["pooled"]


# ---------------- baseline ----------------
def load_baseline(path):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().to(DEVICE)
    return model, tok


def baseline_forward_batch(model, tok, seqs):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=False)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        hs = out[0] if isinstance(out, tuple) else out.last_hidden_state
        _ = hs.mean(1)


def bench_one(name, fwd_fn, seqs_full, L, tokens_per_seq):
    """fwd_fn(seqs_batch) -> None. 返回吞吐+显存+延迟。"""
    torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # warmup
    for _ in range(N_WARMUP):
        fwd_fn(seqs_full[:BATCH])
    torch.cuda.synchronize()

    # 吞吐: N_BATCH 个 batch 计时
    t0 = time.perf_counter()
    n_seq = 0
    for b in range(N_BATCH):
        batch = seqs_full[(b * BATCH) % (len(seqs_full) - BATCH): (b * BATCH) % (len(seqs_full) - BATCH) + BATCH]
        fwd_fn(batch)
        n_seq += len(batch)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    seq_per_s = n_seq / dt
    tok_per_s = seq_per_s * tokens_per_seq
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    # 单序列延迟 (batch=1)
    lat = []
    for i in range(LAT_N):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fwd_fn(seqs_full[i:i + 1])
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    return {
        "seq_per_s": round(seq_per_s, 2),
        "tok_per_s": round(tok_per_s, 1),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "lat_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "lat_p90_ms": round(float(np.percentile(lat, 90)), 2),
        "tokens_per_seq": tokens_per_seq,
        "n_seq_timed": n_seq,
        "wall_s": round(dt, 3),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark TomatoPGFM zero-feature GraphAdapter-on/off efficiency on a CUDA GPU."
    )
    parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the inference benchmark")
    results = {"meta": {"batch": BATCH, "n_batch": N_BATCH, "n_warmup": N_WARMUP,
                        "lat_n": LAT_N, "device": DEVICE, "dtype": "fp32", "mask_bp": MASK_BP,
                        "gpu": torch.cuda.get_device_name(0),
                        "tomatopgfm_conditions": {
                            "graph_on_key_label": "zero-feature adapter-on",
                            "graph_features": "all zeros",
                            "edge_index": None,
                            "graph_message_executed": False,
                            "graph_off_key_label": "graph-off",
                        }}, "models": {}}

    n_needed = BATCH * (N_BATCH + 2) + LAT_N + BATCH

    # ---- TomatoPGFM ----
    print("=== TomatoPGFM loading ===", flush=True)
    tomatopgfm_model, tomatopgfm_tokenizer, gfdim = load_tomatopgfm()
    tomatopgfm_total, tomatopgfm_train = count_params(tomatopgfm_model)
    for arm in ["on", "off"]:
        # Keep the historical graph_on/off key names for result-file compatibility.
        key = f"TomatoPGFM_graph_{arm}"
        results["models"][key] = {"params_total": tomatopgfm_total, "params_trainable": tomatopgfm_train, "by_seqlen": {}}
        for L in SEQLENS:
            seqs = synth_seqs(n_needed, L, SEED)
            # token 数 = 实测 encode 后长度
            tps = len(tomatopgfm_tokenizer.encode(seqs[0], L))
            r = bench_one(key, lambda s, L=L, arm=arm: tomatopgfm_forward_batch(tomatopgfm_model, tomatopgfm_tokenizer, gfdim, s, L, arm), seqs, L, tps)
            results["models"][key]["by_seqlen"][str(L)] = r
            print(f"  {key} L={L}: {r['seq_per_s']} seq/s, {r['peak_mem_mb']} MB, p50={r['lat_p50_ms']}ms, tok/seq={tps}", flush=True)
    del tomatopgfm_model; torch.cuda.empty_cache(); gc.collect()

    # ---- baselines ----
    for name, path in [("dnabert2", DNABERT2), ("plantmamba", PLANTMAMBA)]:
        print(f"=== {name} 加载 ===", flush=True)
        try:
            model, tok = load_baseline(path)
        except Exception as e:
            print(f"  {name} 加载失败: {e}", flush=True)
            results["models"][name] = {"error": str(e)}
            continue
        total, train = count_params(model)
        results["models"][name] = {"params_total": total, "params_trainable": train, "by_seqlen": {}}
        for L in SEQLENS:
            seqs = synth_seqs(n_needed, L, SEED)
            enc = tok(seqs[:1], return_tensors="pt", truncation=False)
            tps = enc["input_ids"].shape[1]
            try:
                r = bench_one(name, lambda s: baseline_forward_batch(model, tok, s), seqs, L, tps)
                results["models"][name]["by_seqlen"][str(L)] = r
                print(f"  {name} L={L}: {r['seq_per_s']} seq/s, {r['peak_mem_mb']} MB, p50={r['lat_p50_ms']}ms, tok/seq={tps}", flush=True)
            except Exception as e:
                results["models"][name]["by_seqlen"][str(L)] = {"error": str(e)}
                print(f"  {name} L={L} 失败: {e}", flush=True)
        del model; torch.cuda.empty_cache(); gc.collect()

    outp = REPO / "runs/stage5/bench_efficiency.json"
    outp.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n结果 -> {outp}", flush=True)

    # ---- 报告 ----
    write_report(results)


def write_report(results):
    lines = ["# 效率 Benchmark 报告\n"]
    lines.append(f"**生成**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    m = results["meta"]
    lines.append(f"**GPU**: {m['gpu']} | **dtype**: {m['dtype']} | **batch**: {m['batch']} | "
                 f"**计时 batch**: {m['n_batch']} (warmup {m['n_warmup']}) | **延迟采样**: {m['lat_n']}\n")
    lines.append("**口径**: 纯前向推理, torch.no_grad, eval(), 中心 12bp mask 合成 ACGT 序列(固定 seed), mask-aware/mean pool。\n\n")

    lines.append("## 参数量\n")
    lines.append("| 模型 | 总参数 | 可训练参数 |\n|---|---|---|\n")
    for k, v in results["models"].items():
        if "params_total" in v:
            lines.append(f"| {k} | {v['params_total']/1e6:.2f}M | {v['params_trainable']/1e6:.2f}M |\n")
    lines.append("\n")

    for L in SEQLENS:
        lines.append(f"## 序列长度 {L} bp\n")
        lines.append("| 模型 | tok/seq | seq/s | token/s | 峰值显存(MB) | 延迟 p50(ms) | 延迟 p90(ms) |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for k, v in results["models"].items():
            r = v.get("by_seqlen", {}).get(str(L))
            if r and "error" not in r:
                lines.append(f"| {k} | {r['tokens_per_seq']} | {r['seq_per_s']} | {r['tok_per_s']} | "
                             f"{r['peak_mem_mb']} | {r['lat_p50_ms']} | {r['lat_p90_ms']} |\n")
            elif r:
                lines.append(f"| {k} | — | ERROR | {r.get('error','')[:40]} | — | — | — |\n")
        lines.append("\n")

    outp = REPO / "runs/stage5/BENCH_EFFICIENCY_REPORT.md"
    outp.write_text("".join(lines))
    print(f"报告 -> {outp}", flush=True)


if __name__ == "__main__":
    main()
