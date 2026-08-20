#!/usr/bin/env python
"""Extract embeddings for gene/CDS-versus-intergenic frozen probes.

Windows use zero-based half-open coordinates. Positive labels are assigned when
the window centre lies in a gene or CDS; negative windows are entirely
intergenic and boundary-crossing windows are excluded. The output contains
TomatoPGFM graph-on/off, DNABERT-2 and PlantDNAMamba embeddings plus chromosome
and task labels.
"""
from __future__ import annotations
import sys, os, json, gzip, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(os.environ.get("TOMATOPGFM_REPO", Path(__file__).resolve().parents[1]))
ROOT = Path(os.environ.get("TOMATOPGFM_DATA_ROOT", REPO / "data"))
TOK_VOCAB = Path(os.environ.get("TOMATOPGFM_TOKENIZER", REPO / "assets/tokenizer_vocab.json"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src"))

TOMATOPGFM_CHECKPOINT_PATH = Path(os.environ.get("TOMATOPGFM_CHECKPOINT", REPO / "model.safetensors"))
DNABERT2 = os.environ.get("DNABERT2_MODEL", "")
PLANTMAMBA = os.environ.get("PLANTDNAMAMBA_MODEL", "")
WBP = 512
SEED = 1234


def load_intervals(gff_path):
    genes = defaultdict(list); cds = defaultdict(list)
    with gzip.open(gff_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            chrom, ftype, s, e = p[0], p[2], int(p[3]), int(p[4])
            if ftype == "gene":
                genes[chrom].append((s, e))
            elif ftype == "CDS":
                cds[chrom].append((s, e))
    for d in (genes, cds):
        for c in d:
            d[c].sort()
    return genes, cds


def point_in_any(chrom, pos, intervals):
    ivs = intervals.get(chrom)
    if not ivs:
        return False
    for s, e in ivs:
        if s > pos:
            break
        if s <= pos <= e:
            return True
    return False


def window_overlaps_gene(chrom, w_s, w_e, genes):
    ivs = genes.get(chrom)
    if not ivs:
        return False
    for s, e in ivs:
        if s > w_e:
            break
        if e >= w_s:
            return True
    return False


def select_windows(genome, smoke=False):
    """返回选中窗口 list[dict(seq,chrom,is_gene,is_cds,is_intergenic)]。"""
    gff = str(ROOT / "holdout" / f"{genome}.gff3.gz")
    win = REPO / f"runs/m1_holdout_v2/{genome}_windows.jsonl"
    genes, cds = load_intervals(gff)

    gene_rows = []      # (seq, chrom, is_cds)
    intergenic_by_chrom = defaultdict(list)  # chrom -> [seq]
    with open(win) as f:
        for line in f:
            r = json.loads(line)
            if r["window_bp"] != WBP:
                continue
            chrom = r["chrom"]
            w_s = r["start"] + 1
            w_e = r["end"]
            center = r["start"] + WBP // 2 + 1
            if point_in_any(chrom, center, genes):
                is_cds = point_in_any(chrom, center, cds)
                gene_rows.append((r["seq"], chrom, is_cds))
            else:
                if not window_overlaps_gene(chrom, w_s, w_e, genes):
                    intergenic_by_chrom[chrom].append((r["seq"], chrom))
            if smoke and len(gene_rows) >= 30 and sum(len(v) for v in intergenic_by_chrom.values()) >= 30:
                break

    # 每染色体等量采 intergenic (=该染色体 gene 数, 供 gene 任务 1:1)
    rng = np.random.default_rng(SEED)
    gene_by_chrom = defaultdict(int)
    for _, c, _ in gene_rows:
        gene_by_chrom[c] += 1
    intergenic_sel = []
    for c, pool in intergenic_by_chrom.items():
        need = gene_by_chrom.get(c, 0) if not smoke else min(5, len(pool))
        need = min(need, len(pool))
        if need > 0:
            idx = rng.choice(len(pool), size=need, replace=False)
            intergenic_sel += [pool[i] for i in idx]

    rows = []
    for seq, chrom, is_cds in gene_rows:
        rows.append({"seq": seq, "chrom": chrom, "is_gene": True, "is_cds": is_cds, "is_intergenic": False})
    for seq, chrom in intergenic_sel:
        rows.append({"seq": seq, "chrom": chrom, "is_gene": False, "is_cds": False, "is_intergenic": True})
    return rows


# ---------- 模型 forward (复用 extract_embeddings 口径) ----------
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
    return model, tok, cfg.graph_feature_dim


def tomatopgfm_embed(model, tok, gfdim, seqs, L=WBP, B=32):
    import torch
    on_all, off_all = [], []
    for i in range(0, len(seqs), B):
        chunk = seqs[i:i + B]
        ids_list = [tok.encode(s, L) for s in chunk]
        Lm = max(len(x) for x in ids_list)
        ids = torch.zeros(len(chunk), Lm, dtype=torch.long)
        for j, x in enumerate(ids_list):
            ids[j, :len(x)] = torch.tensor(x)
        ids = ids.cuda()
        gf = torch.zeros(len(chunk), Lm, gfdim, device="cuda")
        with torch.no_grad():
            on = model(ids, gf, "on", edge_index=None)["pooled"]
            off = model(ids, gf, "off", edge_index=None)["pooled"]
        on_all.append(on.cpu().numpy()); off_all.append(off.cpu().numpy())
    return np.concatenate(on_all), np.concatenate(off_all)


def load_baseline(path):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().cuda()
    return model, tok


def baseline_embed(model, tok, seqs, B=16):
    import torch
    out_all = []
    for i in range(0, len(seqs), B):
        chunk = seqs[i:i + B]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            hs = out[0] if isinstance(out, tuple) else out.last_hidden_state
            if "attention_mask" in enc:
                m = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                emb = (hs * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                emb = hs.mean(1)
        out_all.append(emb.cpu().numpy())
    return np.concatenate(out_all)


def main():
    import torch, gc
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default="LA1974")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    rows = select_windows(args.genome, smoke=args.smoke)
    seqs = [r["seq"] for r in rows]
    n = len(rows)
    print(f"[{args.genome}] 选中窗口: {n} "
          f"(gene={sum(r['is_gene'] for r in rows)}, cds={sum(r['is_cds'] for r in rows)}, "
          f"intergenic={sum(r['is_intergenic'] for r in rows)})", flush=True)

    tomatopgfm_model, tomatopgfm_tokenizer, gfdim = load_tomatopgfm()
    tomatopgfm_graph_on, tomatopgfm_graph_off = tomatopgfm_embed(tomatopgfm_model, tomatopgfm_tokenizer, gfdim, seqs)
    print(f"  TomatoPGFM done: on{tomatopgfm_graph_on.shape} off{tomatopgfm_graph_off.shape}", flush=True)
    del tomatopgfm_model; torch.cuda.empty_cache(); gc.collect()

    db, dbtok = load_baseline(DNABERT2)
    dnabert2 = baseline_embed(db, dbtok, seqs)
    print(f"  dnabert2 done: {dnabert2.shape}", flush=True)
    del db; torch.cuda.empty_cache(); gc.collect()

    pm, pmtok = load_baseline(PLANTMAMBA)
    plantmamba = baseline_embed(pm, pmtok, seqs)
    print(f"  plantmamba done: {plantmamba.shape}", flush=True)
    del pm; torch.cuda.empty_cache(); gc.collect()

    outd = REPO / "runs/stage5/genecds_embeddings"
    outd.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    outp = outd / f"{args.genome}_512{suffix}.npz"
    np.savez_compressed(
        outp,
        TomatoPGFM_graph_on=tomatopgfm_graph_on.astype(np.float32), TomatoPGFM_graph_off=tomatopgfm_graph_off.astype(np.float32),
        dnabert2=dnabert2.astype(np.float32), plantmamba=plantmamba.astype(np.float32),
        chroms=np.array([r["chrom"] for r in rows]),
        is_gene=np.array([r["is_gene"] for r in rows]),
        is_cds=np.array([r["is_cds"] for r in rows]),
        is_intergenic=np.array([r["is_intergenic"] for r in rows]),
    )
    print(f"落盘 -> {outp}", flush=True)


if __name__ == "__main__":
    main()
