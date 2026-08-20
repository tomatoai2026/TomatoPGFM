#!/usr/bin/env python
"""LoRA adaptation for external-accession gene/CDS classification.

TomatoPGFM is evaluated with external graph input disabled and compared with
DNABERT-2 and PlantDNAMamba. Chromosomes 03 and 05 form the test partition;
the remaining chromosomes supply fitting and validation data. Only LoRA
adapters and a linear classifier are trainable.
"""
from __future__ import annotations
import sys, os, json, argparse, time, gzip
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("TOMATOPGFM_REPO", Path(__file__).resolve().parents[1]))
ROOT = Path(os.environ.get("TOMATOPGFM_DATA_ROOT", REPO / "data"))
TOK_VOCAB = Path(os.environ.get("TOMATOPGFM_TOKENIZER", REPO / "assets/tokenizer_vocab.json"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

TOMATOPGFM_CHECKPOINT_PATH = Path(os.environ.get("TOMATOPGFM_CHECKPOINT", REPO / "model.safetensors"))
DNABERT2 = os.environ.get("DNABERT2_MODEL", "")
PLANTMAMBA = os.environ.get("PLANTDNAMAMBA_MODEL", "")
WBP = 512
SEED = 1234
TEST_CHROMS = {"chr03", "chr05"}

# Fixed LoRA and optimization hyperparameters.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LR = 2e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 15
PATIENCE = 3
BATCH = 32


def build_dataset(manifest, task, smoke=False):
    """Load a materialized evaluation manifest and construct a balanced task."""
    opener = gzip.open if str(manifest).endswith(".gz") else open
    with opener(manifest, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows or any("seq" not in row for row in rows):
        raise ValueError(
            "The LoRA input must contain a seq field. Reconstruct it with "
            "scripts/materialize_manifest_sequences.py before running LoRA."
        )
    if smoke:
        rows = rows[: min(2000, len(rows))]
    rng = np.random.default_rng(SEED)
    if task == "gene":
        pos = [r for r in rows if r["is_gene"]]
        neg = [r for r in rows if r["is_intergenic"]]
    elif task == "cds":
        pos = [r for r in rows if r["is_cds"]]
        neg = [r for r in rows if r["is_intergenic"]]
    else:
        raise ValueError(task)
    # 1:1 平衡 (取 min)
    k = min(len(pos), len(neg))
    if len(pos) > k:
        idx = rng.choice(len(pos), size=k, replace=False); pos = [pos[i] for i in idx]
    if len(neg) > k:
        idx = rng.choice(len(neg), size=k, replace=False); neg = [neg[i] for i in idx]
    data = [(r["seq"], r["chrom"], 1) for r in pos] + [(r["seq"], r["chrom"], 0) for r in neg]
    seqs = [d[0] for d in data]
    chroms = np.array([d[1] for d in data])
    labels = np.array([d[2] for d in data], dtype=np.int64)
    is_test = np.array([c in TEST_CHROMS for c in chroms])
    return seqs, labels, chroms, is_test


# ---------------- TomatoPGFM backbone wrapper ----------------
class TomatoPGFMClassifier:
    """TomatoPGFM backbone with LoRA and a linear head in graph-off mode."""
    def __init__(self):
        import torch, torch.nn as nn
        from tomatopgfm.config import load_model_config
        from tomatopgfm.checkpoint_io import load_model_weights
        from tomatopgfm.model import TomatoPGFM
        from tomatopgfm.tokenizer import RCKmerTokenizer
        from peft import LoraConfig, get_peft_model
        cfg = load_model_config(REPO / "configs/model_final.yaml")
        model = TomatoPGFM(cfg)
        load_model_weights(model, TOMATOPGFM_CHECKPOINT_PATH)
        self.gfdim = cfg.graph_feature_dim
        self.tok = RCKmerTokenizer.load(TOK_VOCAB)
        # LoRA on mamba/attn projections
        lc = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                        target_modules=["in_proj", "out_proj", "mix"], bias="none")
        self.backbone = get_peft_model(model, lc)
        self.head = nn.Linear(cfg.d_model, 2)
        self.backbone.cuda(); self.head.cuda()

    def trainable_params(self):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        return bb, list(self.head.parameters())

    def encode_batch(self, seqs):
        import torch
        ids_list = [self.tok.encode(s, WBP) for s in seqs]
        Lm = max(len(x) for x in ids_list)
        ids = torch.zeros(len(seqs), Lm, dtype=torch.long)
        for j, x in enumerate(ids_list):
            ids[j, :len(x)] = torch.tensor(x)
        return ids.cuda()

    def forward(self, seqs):
        import torch
        ids = self.encode_batch(seqs)
        gf = torch.zeros(ids.shape[0], ids.shape[1], self.gfdim, device="cuda")
        pooled = self.backbone(ids, gf, "off", edge_index=None)["pooled"]
        return self.head(pooled)

    def train(self): self.backbone.train(); self.head.train()
    def eval(self): self.backbone.eval(); self.head.eval()


# ---------------- HF baseline wrapper ----------------
class HFClassifier:
    def __init__(self, path):
        import torch, torch.nn as nn
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        from peft import LoraConfig, get_peft_model
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # DNABERT-2 治本: config attention_probs_dropout_prob=0 会走自带 triton flash-attn,
        # 其反向 kernel 在本环境 triton 版本下编译失败(CompilationError)。设正 dropout
        # 令 bert_layers.py:161 走 PyTorch 原生 attention 路径 (反向稳定, 微调本应开 dropout)。
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if getattr(cfg, "attention_probs_dropout_prob", None) == 0:
            cfg.attention_probs_dropout_prob = 0.1
        model = AutoModel.from_pretrained(path, trust_remote_code=True, config=cfg)
        # auto-detect linear target modules for LoRA
        import torch.nn as nn2
        names = set()
        for n, m in model.named_modules():
            if isinstance(m, nn2.Linear):
                names.add(n.split(".")[-1])
        is_mamba = getattr(cfg, "model_type", "") == "mamba"
        if is_mamba:
            # peft 禁止对 mamba out_proj/conv1d 挂 LoRA (破坏状态传播); 只挂投影输入侧
            cand = ["in_proj", "x_proj", "dt_proj"]
        else:
            cand = ["query", "key", "value", "Wqkv", "dense", "out_proj"]
        targets = [t for t in cand if t in names]
        if not targets:
            targets = list(names)[:4]
        self.d_model = model.config.hidden_size
        lc = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                        target_modules=targets, bias="none")
        self.backbone = get_peft_model(model, lc)
        self.head = nn.Linear(self.d_model, 2)
        self.backbone.cuda(); self.head.cuda()
        self.targets = targets

    def trainable_params(self):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        return bb, list(self.head.parameters())

    def forward(self, seqs):
        import torch
        enc = self.tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=WBP)
        enc = {k: v.cuda() for k, v in enc.items()}
        out = self.backbone(**enc)
        hs = out[0] if isinstance(out, tuple) else out.last_hidden_state
        if "attention_mask" in enc:
            m = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            emb = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            emb = hs.mean(1)
        return self.head(emb)

    def train(self): self.backbone.train(); self.head.train()
    def eval(self): self.backbone.eval(); self.head.eval()


def count_trainable(clf):
    bb, hd = clf.trainable_params()
    return sum(p.numel() for p in bb) + sum(p.numel() for p in hd)


def run_metrics(y_true, logits):
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
    import numpy as np
    prob = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
    pred = (prob >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "bacc": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def train_one(clf, seqs, labels, is_test, tag, smoke=False):
    import torch, torch.nn as nn
    rng = np.random.default_rng(SEED)
    tr_idx = np.where(~is_test)[0]
    te_idx = np.where(is_test)[0]
    # carve val from train (10%, stratified-ish by shuffle)
    rng.shuffle(tr_idx)
    n_val = max(1, int(0.1 * len(tr_idx)))
    val_idx = tr_idx[:n_val]; fit_idx = tr_idx[n_val:]

    bb, hd = clf.trainable_params()
    opt = torch.optim.AdamW([
        {"params": bb, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": hd, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
    ])
    lossf = nn.CrossEntropyLoss()
    y = torch.tensor(labels).cuda()

    def eval_split(idx):
        clf.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(idx), BATCH):
                b = idx[i:i + BATCH]
                logit = clf.forward([seqs[j] for j in b])
                outs.append(logit.float().cpu().numpy())
        return np.concatenate(outs)

    best_val = -1; best_state = None; bad = 0
    max_ep = 2 if smoke else MAX_EPOCHS
    for ep in range(max_ep):
        clf.train()
        perm = fit_idx.copy(); rng.shuffle(perm)
        tot = 0.0; nb = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            logit = clf.forward([seqs[j] for j in b])
            loss = lossf(logit, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if smoke and nb >= 3:
                break
        val_logit = eval_split(val_idx)
        vm = run_metrics(labels[val_idx], val_logit)
        print(f"    [{tag}] ep{ep} loss={tot/max(nb,1):.4f} val_auroc={vm['auroc']:.4f}", flush=True)
        if vm["auroc"] > best_val:
            best_val = vm["auroc"]; bad = 0
            best_state = {"head": {k: v.detach().cpu().clone() for k, v in clf.head.state_dict().items()}}
            # save LoRA adapter state in-memory (small)
            best_state["bb"] = {k: v.detach().cpu().clone() for k, v in clf.backbone.state_dict().items() if "lora" in k.lower()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"    [{tag}] early stop @ ep{ep}", flush=True)
                break
    # restore best
    if best_state is not None:
        clf.head.load_state_dict(best_state["head"])
        cur = clf.backbone.state_dict()
        for k, v in best_state["bb"].items():
            cur[k] = v.cuda()
        clf.backbone.load_state_dict(cur)
    test_logit = eval_split(te_idx)
    tm = run_metrics(labels[te_idx], test_logit)
    tm["best_val_auroc"] = float(best_val)
    tm["n_train"] = int(len(fit_idx)); tm["n_val"] = int(len(val_idx)); tm["n_test"] = int(len(te_idx))
    return tm


def main():
    import torch, gc
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default="LA1974")
    ap.add_argument("--task", default="gene", choices=["gene", "cds"])
    ap.add_argument("--models", default="tomatopgfm,dnabert2,plantmamba")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--manifest",
        type=Path,
        help="materialized JSONL(.gz) with seq, chrom and gene/CDS/intergenic labels",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    t0 = time.time()
    manifest = args.manifest or Path(os.environ.get(
        "TOMATOPGFM_EVAL_MANIFEST",
        REPO / "manifests" / f"{args.genome}_evaluation_windows_with_sequences.jsonl.gz",
    ))
    if not manifest.exists():
        raise FileNotFoundError(
            f"Materialized manifest not found: {manifest}. Run "
            "scripts/materialize_manifest_sequences.py first."
        )
    seqs, labels, chroms, is_test = build_dataset(manifest, args.task, smoke=args.smoke)
    print(f"[{args.genome}/{args.task}] N={len(seqs)} pos={int(labels.sum())} "
          f"test={int(is_test.sum())} (chr03+chr05)", flush=True)

    results = {}
    for mdl in args.models.split(","):
        mdl = mdl.strip()
        print(f"  === {mdl} ===", flush=True)
        if mdl == "tomatopgfm":
            clf = TomatoPGFMClassifier()
        elif mdl == "dnabert2":
            clf = HFClassifier(DNABERT2)
        elif mdl == "plantmamba":
            clf = HFClassifier(PLANTMAMBA)
        else:
            continue
        ntr = count_trainable(clf)
        print(f"    trainable params: {ntr:,}", flush=True)
        tm = train_one(clf, seqs, labels, is_test, mdl, smoke=args.smoke)
        tm["trainable_params"] = int(ntr)
        results[mdl] = tm
        print(f"    TEST {mdl}: AUROC={tm['auroc']:.4f} AUPRC={tm['auprc']:.4f} "
              f"bAcc={tm['bacc']:.4f} MCC={tm['mcc']:.4f}", flush=True)
        del clf; torch.cuda.empty_cache(); gc.collect()

    outd = (args.out.parent if args.out else REPO / "results/lora_rerun")
    outd.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    outp = args.out or outd / f"lora_genecds_{args.genome}_{args.task}{suffix}.json"
    payload = {
        "genome": args.genome, "task": args.task, "manifest": str(manifest),
        "n_total": len(seqs), "n_pos": int(labels.sum()), "n_test": int(is_test.sum()),
        "hparams": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lr": LR, "head_lr": HEAD_LR,
                    "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch": BATCH,
                    "test_chroms": sorted(TEST_CHROMS), "TomatoPGFM_graph_mode": "off"},
        "results": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    outp.write_text(json.dumps(payload, indent=2))
    print(f"落盘 -> {outp}  ({payload['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
