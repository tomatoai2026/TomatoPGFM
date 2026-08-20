#!/usr/bin/env python
"""Build coordinate-bearing sequence windows for LA1974 and MicroTom.

Each record contains raw sequence, TomatoPGFM token IDs, genome, chromosome,
zero-based half-open coordinates, window size and N fraction. Baseline models
apply their native tokenizers during embedding extraction. No graph features or
adjacency are assigned to these external-accession windows.
"""
import argparse, gzip, json, os, sys
from pathlib import Path

REPO = Path(os.environ.get("TOMATOPGFM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
TOK_CODE = Path(__file__).resolve().parents[1] / "src"
TOK_VOCAB = Path(os.environ.get("TOMATOPGFM_TOKENIZER", Path(__file__).resolve().parents[1] / "assets/tokenizer_vocab.json"))
HOLDOUT = Path(os.environ.get("TOMATOPGFM_HOLDOUT_ROOT", REPO / "holdout"))

GENOME_FA = {
    "LA1974": HOLDOUT / "LA1974.fa.gz",
    "MicroTom": HOLDOUT / "MicroTom.fasta.gz",
}

sys.path.insert(0, str(TOK_CODE))
from tomatopgfm.tokenizer import RCKmerTokenizer  # noqa: E402


def iter_fasta(path: Path):
    """Yield (record_id, sequence). record_id 经 .strip() 规范化(防 \\r / 空白污染, F1 防御)."""
    rid, chunks = None, []
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if rid is not None:
                    yield rid, "".join(chunks)
                rid = line[1:].strip().split()[0]  # 取第一个空白前 token, strip 防 \r
                chunks = []
            else:
                chunks.append(line.strip())
        if rid is not None:
            yield rid, "".join(chunks)


def n_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    return sum(1 for c in seq.upper() if c not in "ACGT") / len(seq)


def build(genome, window_sizes, max_windows_per_chrom, n_max, only_chrom, out_path, tok):
    vocab_size = max(tok.vocab.values()) + 1
    fa = GENOME_FA[genome]
    n_written = 0
    n_skipped_N = 0
    max_tok_id = 0
    per_chrom = {}
    with open(out_path, "w") as out:
        for chrom, seq in iter_fasta(fa):
            if only_chrom and chrom != only_chrom:
                continue
            seq = seq.upper()
            L = len(seq)
            for wbp in window_sizes:
                # 非重叠切窗
                starts = list(range(0, L - wbp + 1, wbp))
                # 可选均匀稀疏采样
                if max_windows_per_chrom and len(starts) > max_windows_per_chrom:
                    step = len(starts) / max_windows_per_chrom
                    starts = [starts[int(i * step)] for i in range(max_windows_per_chrom)]
                for st in starts:
                    sub = seq[st:st + wbp]
                    nf = n_fraction(sub)
                    if nf > n_max:
                        n_skipped_N += 1
                        continue
                    ids = tok.encode(sub)  # 不 pad, 真实长度; CLS+kmers+SEP
                    if ids:
                        max_tok_id = max(max_tok_id, max(ids))
                    rec = {
                        "genome": genome,
                        "chrom": chrom,
                        "start": st,
                        "end": st + wbp,
                        "window_bp": wbp,
                        "seq": sub,
                        "n_fraction": round(nf, 4),
                        "stratum": genome,
                        "non_ref_rich": "not_available",
                        "graph_edge": None,
                        "tomatopgfm_input_ids": ids,
                    }
                    out.write(json.dumps(rec) + "\n")
                    n_written += 1
                    per_chrom[chrom] = per_chrom.get(chrom, 0) + 1
    return {
        "genome": genome,
        "out": str(out_path),
        "n_written": n_written,
        "n_skipped_N": n_skipped_N,
        "max_token_id": max_tok_id,
        "vocab_size": vocab_size,
        "token_within_vocab": max_tok_id < vocab_size,
        "per_chrom": per_chrom,
        "window_sizes": window_sizes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True, choices=list(GENOME_FA))
    ap.add_argument("--window-sizes", default="512,1024,2048")
    ap.add_argument("--max-windows-per-chrom", type=int, default=0, help="0=不采样(全切)")
    ap.add_argument("--n-max", type=float, default=0.05, help="N 比例上限, >此值滤掉")
    ap.add_argument("--only-chrom", default="", help="smoke: 只处理某条染色体")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = RCKmerTokenizer.load(TOK_VOCAB)
    wsizes = [int(x) for x in args.window_sizes.split(",")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = build(args.genome, wsizes, args.max_windows_per_chrom, args.n_max,
                   args.only_chrom or None, args.out, tok)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
