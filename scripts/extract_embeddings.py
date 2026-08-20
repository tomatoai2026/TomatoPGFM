#!/usr/bin/env python
"""Extract frozen embeddings for centre-base and centre-6-mer prediction.

The central 12-bp interval is replaced with N before model-specific
tokenization. TomatoPGFM is evaluated in graph-on and graph-off modes;
DNABERT-2 and PlantDNAMamba use their native tokenizers and output protocols.
"""
from __future__ import annotations
import argparse, json, sys, os, hashlib
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("TOMATOPGFM_REPO", Path(__file__).resolve().parents[1]))
ROOT = Path(os.environ.get("TOMATOPGFM_DATA_ROOT", REPO / "data"))  # tokenizer/pangenome 资产在根, 不带 production
TOK_VOCAB = Path(os.environ.get("TOMATOPGFM_TOKENIZER", REPO / "assets/tokenizer_vocab.json"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src"))

MASK_BP = 12          # 中心 mask 碱基数(覆盖中心 6-mer + 两侧缓冲, 防邻接泄漏)
CENTER_K = 6          # 中心 k-mer 长度(label)
TOMATOPGFM_CHECKPOINT_PATH = Path(os.environ.get("TOMATOPGFM_CHECKPOINT", REPO / "model.safetensors"))
DNABERT2 = os.environ.get("DNABERT2_MODEL", "")
PLANTMAMBA = os.environ.get("PLANTDNAMAMBA_MODEL", "")


def center_kmer_label(seq: str) -> int | None:
    """窗口中心 6-mer 的类别 id (0-4095). 含 N 返回 None(跳过)."""
    c = len(seq) // 2
    s = c - CENTER_K // 2
    kmer = seq[s:s + CENTER_K]
    if len(kmer) != CENTER_K or any(b not in "ACGT" for b in kmer):
        return None
    idx = 0
    for b in kmer:
        idx = idx * 4 + "ACGT".index(b)
    return idx


def center_base_label(seq: str) -> int | None:
    """Return the central A/C/G/T class (0-3), or None for an ambiguous base."""
    c = len(seq) // 2
    b = seq[c]
    if b not in "ACGT":
        return None
    return "ACGT".index(b)


def mask_center(seq: str) -> str:
    """Replace the central MASK_BP bases with N before tokenization."""
    c = len(seq) // 2
    s = c - MASK_BP // 2
    e = s + MASK_BP
    return seq[:s] + "N" * (e - s) + seq[e:]


def assert_no_leak(masked_seq: str) -> bool:
    """F4 断言: masked 窗口中心区间确实全 N(probe 输入不含目标位原序列)."""
    c = len(masked_seq) // 2
    s = c - MASK_BP // 2
    e = s + MASK_BP
    return all(b == "N" for b in masked_seq[s:e])


# ---------- TomatoPGFM ----------
def load_tomatopgfm():
    import torch
    from tomatopgfm.config import load_model_config
    from tomatopgfm.checkpoint_io import load_model_weights
    from tomatopgfm.model import TomatoPGFM
    from tomatopgfm.tokenizer import RCKmerTokenizer
    cfg = load_model_config(REPO / "configs/model_final.yaml")
    model = TomatoPGFM(cfg)
    load_model_weights(model, TOMATOPGFM_CHECKPOINT_PATH)
    model.eval().cuda()
    tok = RCKmerTokenizer.load(TOK_VOCAB)
    gfdim = cfg.graph_feature_dim
    return model, tok, gfdim


def tomatopgfm_embed(model, tok, gfdim, masked_seqs, max_len):
    import torch
    embs_on, embs_off = [], []
    B = 32
    for i in range(0, len(masked_seqs), B):
        chunk = masked_seqs[i:i + B]
        ids_list = [tok.encode(s, max_len) for s in chunk]
        L = max(len(x) for x in ids_list)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, x in enumerate(ids_list):
            ids[j, :len(x)] = torch.tensor(x)
        ids = ids.cuda()
        gf = torch.zeros(len(chunk), L, gfdim, device="cuda")  # M1 无图边: 零特征
        with torch.no_grad():
            on = model(ids, gf, "on", edge_index=None)["pooled"]
            off = model(ids, gf, "off", edge_index=None)["pooled"]
        embs_on.append(on.cpu().numpy())
        embs_off.append(off.cpu().numpy())
    return np.concatenate(embs_on), np.concatenate(embs_off)


# ---------- baseline ----------
def load_baseline(path, force_cuda):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # F6: 显式关 truncation/padding(PlantDNAMamba 默认偷砍 512)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
    if force_cuda:
        model = model.cuda()
    return model, tok


def baseline_embed(model, tok, masked_seqs, on_cuda):
    import torch
    embs = []
    B = 16
    for i in range(0, len(masked_seqs), B):
        chunk = masked_seqs[i:i + B]
        # 逐条 encode(长度不一), 再 pad 到 batch max
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=False)
        if on_cuda:
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hs = out[0] if isinstance(out, tuple) else out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype) if "attention_mask" in enc else None
        if mask is not None:
            emb = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)  # mask-aware mean-pool
        else:
            emb = hs.mean(1)
        embs.append(emb.cpu().numpy())
    return np.concatenate(embs)


def load_windows(genome, limit=None, only_bp=None, chroms=None):
    rows = []
    # 优先用 v2(B方案 5000窗口/染色体, headline 可过稀疏度gate); 回退 v1
    fn = REPO / f"runs/m1_holdout_v2/{genome}_windows.jsonl"
    if not fn.exists():
        fn = REPO / f"runs/m1_holdout/{genome}_windows.jsonl"
    per_chrom = {}
    with open(fn) as f:
        for line in f:
            r = json.loads(line)
            if only_bp and r["window_bp"] != only_bp:
                continue
            if chroms is not None:
                if r["chrom"] not in chroms:
                    continue
                # 每染色体最多取 limit//len(chroms) 条(混合 smoke 均衡 train/test)
                cap = max(1, (limit or 999999) // len(chroms))
                if per_chrom.get(r["chrom"], 0) >= cap:
                    continue
                per_chrom[r["chrom"]] = per_chrom.get(r["chrom"], 0) + 1
            rows.append(r)
            if limit and chroms is None and len(rows) >= limit:
                break
            if chroms is not None and all(per_chrom.get(c, 0) >= max(1, (limit or 999999)//len(chroms)) for c in chroms):
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default="LA1974")
    ap.add_argument("--only-bp", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default="tomatopgfm,dnabert2,plantmamba")
    ap.add_argument("--chroms", default=None, help="逗号分隔染色体过滤(混合 smoke 用, 如 chr01,chr03,chr05)")
    args = ap.parse_args()
    limit = 96 if args.smoke else (args.limit or None)
    chroms = args.chroms.split(",") if args.chroms else None
    rows = load_windows(args.genome, limit=limit, only_bp=args.only_bp, chroms=chroms)

    # 1) label + mask + 泄漏断言(纯 CPU, 先做)
    kept = []
    for r in rows:
        lab = center_kmer_label(r["seq"])
        if lab is None:
            continue  # 中心含 N, 跳过
        blab = center_base_label(r["seq"])  # center-base aux 标签(kept 下必非 None)
        masked = mask_center(r["seq"])
        assert assert_no_leak(masked), "LEAK: 中心区间未全 N"
        kept.append({"label": lab, "base_label": blab, "masked_seq": masked, "chrom": r["chrom"],
                     "window_bp": r["window_bp"], "raw_center_preview": r["seq"][len(r["seq"])//2-3:len(r["seq"])//2+3]})
    masked_seqs = [k["masked_seq"] for k in kept]
    labels = np.array([k["label"] for k in kept])
    base_labels = np.array([k["base_label"] for k in kept])
    assert not np.any(base_labels < 0) and len(base_labels) == len(labels), "base_label 缺失/未对齐"
    # 断言落盘: masked_seq 中心确实不含原 center kmer
    leak_check = {"n_windows": len(kept), "all_center_masked": all(assert_no_leak(s) for s in masked_seqs),
                  "mask_bp": MASK_BP, "center_k": CENTER_K, "label_vocab": 4 ** CENTER_K}
    print(f"[label+mask] {len(kept)}/{len(rows)} 窗口保留(中心非N), 泄漏断言: {leak_check['all_center_masked']}")

    models = args.models.split(",")
    out = {"genome": args.genome, "window_bp": args.only_bp, "leak_check": leak_check,
           "labels": labels.tolist(), "embeddings": {}}

    if "tomatopgfm" in models:
        model, tok, gfdim = load_tomatopgfm()
        on, off = tomatopgfm_embed(model, tok, gfdim, masked_seqs, max_len=args.only_bp)
        out["embeddings"]["TomatoPGFM_graph_on"] = on
        out["embeddings"]["TomatoPGFM_graph_off"] = off
        print(f"[TomatoPGFM] on={on.shape} off={off.shape}")
        del model
        import torch; torch.cuda.empty_cache()

    if "dnabert2" in models:
        m, t = load_baseline(DNABERT2, force_cuda=True)  # 必须 GPU
        e = baseline_embed(m, t, masked_seqs, on_cuda=True)
        out["embeddings"]["dnabert2"] = e
        print(f"[DNABERT2] {e.shape}")
        del m; import torch; torch.cuda.empty_cache()

    if "plantmamba" in models:
        m, t = load_baseline(PLANTMAMBA, force_cuda=True)
        e = baseline_embed(m, t, masked_seqs, on_cuda=True)
        out["embeddings"]["plantmamba"] = e
        print(f"[PlantDNAMamba] {e.shape}")
        del m; import torch; torch.cuda.empty_cache()

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        chroms = np.array([k["chrom"] for k in kept])
        np.savez_compressed(outp, labels=labels, base_labels=base_labels, chroms=chroms,
                            **{k: v for k, v in out["embeddings"].items()})
        (outp.with_suffix(".leak.json")).write_text(json.dumps(leak_check, indent=2))
        print(f"[saved] {outp} + leak.json")
    print(json.dumps({"genome": args.genome, "window_bp": args.only_bp,
                      "n": len(kept), "models": list(out["embeddings"].keys()),
                      "emb_dims": {k: v.shape[-1] for k, v in out["embeddings"].items()},
                      "leak_assert_pass": leak_check["all_center_masked"]}, indent=2))


if __name__ == "__main__":
    main()
