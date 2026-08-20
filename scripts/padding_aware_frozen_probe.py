"""Re-extract TomatoPGFM embeddings with matched pooling and refit probes.

The production tree is read-only. The script refuses to infer the historical
60k-window sample from a larger candidate pool: sequences must either be
embedded in the authoritative NPZ or supplied through an explicitly selected
manifest whose row order is checked against the NPZ metadata.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 1234
WINDOW_BP = 512
TEST_CHROMS = {"chr03", "chr05"}
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
BOOTSTRAP_RESAMPLES = 1000


def _first_array(npz, aliases, required=True):
    for name in aliases:
        if name in npz.files:
            return npz[name], name
    if required:
        raise KeyError(f"None of {aliases} found. Available NPZ keys: {npz.files}")
    return None, None


def _decode_text(values):
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def _as_bool(values):
    return np.asarray(values).astype(bool)


def _row_from_mapping(row):
    sequence = row.get("seq", row.get("sequence"))
    chromosome = row.get("chrom", row.get("chromosome"))
    if sequence is None or chromosome is None:
        raise KeyError("Each manifest row requires seq/sequence and chrom/chromosome")
    is_gene = bool(row.get("is_gene", row.get("gene", False)))
    is_cds = bool(row.get("is_cds", row.get("cds", False)))
    is_intergenic = bool(row.get("is_intergenic", row.get("intergenic", not is_gene)))
    return {
        "seq": str(sequence),
        "chrom": str(chromosome),
        "is_gene": is_gene,
        "is_cds": is_cds,
        "is_intergenic": is_intergenic,
    }


def load_manifest(path: Path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as manifest:
            sequences, _ = _first_array(manifest, ["sequences", "seqs", "sequence", "seq"])
            chroms, _ = _first_array(manifest, ["chroms", "chromosomes", "chrom"])
            is_gene, _ = _first_array(manifest, ["is_gene", "gene_labels", "labels_gene"])
            is_cds, _ = _first_array(manifest, ["is_cds", "cds_labels", "labels_cds"])
            intergenic, _ = _first_array(
                manifest, ["is_intergenic", "intergenic_labels"], required=False
            )
            if intergenic is None:
                intergenic = ~_as_bool(is_gene)
            return [
                {
                    "seq": sequence,
                    "chrom": chromosome,
                    "is_gene": bool(gene),
                    "is_cds": bool(cds),
                    "is_intergenic": bool(inter),
                }
                for sequence, chromosome, gene, cds, inter in zip(
                    _decode_text(sequences),
                    _decode_text(chroms),
                    _as_bool(is_gene),
                    _as_bool(is_cds),
                    _as_bool(intergenic),
                )
            ]

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        if path.name.endswith((".jsonl", ".jsonl.gz")):
            return [_row_from_mapping(json.loads(line)) for line in handle if line.strip()]
        payload = json.load(handle)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    return [_row_from_mapping(row) for row in rows]


def rows_from_npz(npz):
    sequences, _ = _first_array(
        npz, ["sequences", "seqs", "sequence", "seq"], required=False
    )
    if sequences is None:
        return None
    chroms, _ = _first_array(npz, ["chroms", "chromosomes", "chrom"])
    is_gene, _ = _first_array(npz, ["is_gene", "gene_labels", "labels_gene"])
    is_cds, _ = _first_array(npz, ["is_cds", "cds_labels", "labels_cds"])
    intergenic, _ = _first_array(
        npz, ["is_intergenic", "intergenic_labels"], required=False
    )
    if intergenic is None:
        intergenic = ~_as_bool(is_gene)
    return [
        {
            "seq": sequence,
            "chrom": chromosome,
            "is_gene": bool(gene),
            "is_cds": bool(cds),
            "is_intergenic": bool(inter),
        }
        for sequence, chromosome, gene, cds, inter in zip(
            _decode_text(sequences),
            _decode_text(chroms),
            _as_bool(is_gene),
            _as_bool(is_cds),
            _as_bool(intergenic),
        )
    ]


def validate_rows(rows, npz):
    baseline, baseline_key = _first_array(
        npz, ["dnabert2", "dnabert_2", "DNABERT_2", "emb_dnabert2"]
    )
    if len(rows) != len(baseline):
        raise RuntimeError(
            f"Manifest has {len(rows)} rows but {baseline_key} has {len(baseline)}. "
            "Do not guess the historical 60k-window subsample."
        )
    chroms, chrom_key = _first_array(
        npz, ["chroms", "chromosomes", "chrom"], required=False
    )
    if chroms is not None:
        expected = _decode_text(chroms)
        observed = np.asarray([row["chrom"] for row in rows])
        mismatch = np.flatnonzero(expected != observed)
        if len(mismatch):
            first = int(mismatch[0])
            raise RuntimeError(
                f"Manifest order differs from NPZ {chrom_key} at row {first}: "
                f"{observed[first]!r} != {expected[first]!r}"
            )
    for aliases, field in [
        (["is_gene", "gene_labels", "labels_gene"], "is_gene"),
        (["is_cds", "cds_labels", "labels_cds"], "is_cds"),
        (["is_intergenic", "intergenic_labels"], "is_intergenic"),
    ]:
        expected, key = _first_array(npz, aliases, required=False)
        if expected is None:
            continue
        observed = np.asarray([row[field] for row in rows], dtype=bool)
        mismatch = np.flatnonzero(_as_bool(expected) != observed)
        if len(mismatch):
            raise RuntimeError(f"Manifest {field} differs from NPZ {key} at row {int(mismatch[0])}")


def load_model(source_root: Path, device: str):
    import torch

    repository = source_root
    sys.path.insert(0, str(repository / "src"))
    config_module = importlib.import_module("tomatopgfm.config")
    checkpoint_module = importlib.import_module("tomatopgfm.checkpoint_io")
    model_module = importlib.import_module("tomatopgfm.model")
    tokenizer_module = importlib.import_module("tomatopgfm.tokenizer")
    config = config_module.load_model_config(repository / "configs" / "model_final.yaml")
    model = model_module.TomatoPGFM(config)
    checkpoint = Path(
        os.environ.get("TOMATOPGFM_CHECKPOINT", repository / "model.safetensors")
    )
    checkpoint_module.load_model_weights(model, checkpoint)
    model.eval().to(device)
    tokenizer_path = Path(
        os.environ.get("TOMATOPGFM_TOKENIZER", repository / "assets" / "tokenizer_vocab.json")
    )
    tokenizer = tokenizer_module.RCKmerTokenizer.load(tokenizer_path)
    return model, tokenizer


def extract_embeddings(model, tokenizer, sequences, device: str, batch_size: int):
    import torch

    captured = {}

    def capture_norm(_module, _inputs, output):
        captured["hidden"] = output.detach()

    hook = model.norm.register_forward_hook(capture_norm)
    output = {"unmasked": [], "masked": [], "nopad": []}
    nonpad_lengths = []
    try:
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            padded_lists = [tokenizer.encode(sequence, WINDOW_BP) for sequence in chunk]
            padded = torch.tensor(padded_lists, dtype=torch.long, device=device)
            mask = padded.ne(tokenizer.pad_id).unsqueeze(-1)
            compact_lists = [
                [token_id for token_id in ids if token_id != tokenizer.pad_id]
                for ids in padded_lists
            ]
            lengths = [len(ids) for ids in compact_lists]
            nonpad_lengths.extend(lengths)
            if len(set(lengths)) != 1:
                raise RuntimeError(
                    "Direct no-padding batches require equal token counts; "
                    f"observed lengths {sorted(set(lengths))}."
                )

            with torch.inference_mode():
                model(padded, None, "off", edge_index=None)
            hidden = captured.pop("hidden")
            output["unmasked"].append(hidden.mean(dim=1).cpu().numpy())
            masked = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            output["masked"].append(masked.cpu().numpy())

            compact = torch.tensor(compact_lists, dtype=torch.long, device=device)
            with torch.inference_mode():
                model(compact, None, "off", edge_index=None)
            output["nopad"].append(captured.pop("hidden").mean(dim=1).cpu().numpy())
    finally:
        hook.remove()
    embeddings = {
        key: np.concatenate(value).astype(np.float32) for key, value in output.items()
    }
    return embeddings, np.asarray(nonpad_lengths, dtype=np.int16)


def select_smoke_subset(rows, maximum):
    if not maximum or len(rows) <= maximum:
        return np.arange(len(rows))
    groups = {}
    for index, row in enumerate(rows):
        category = "cds" if row["is_cds"] else "gene" if row["is_gene"] else "intergenic"
        key = (row["chrom"] in TEST_CHROMS, category)
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(SEED)
    quota = max(1, maximum // max(1, len(groups)))
    selected = []
    remainder = []
    for indices in groups.values():
        indices = np.asarray(indices)
        take = min(quota, len(indices))
        chosen = rng.choice(indices, size=take, replace=False)
        selected.extend(chosen.tolist())
        remainder.extend(np.setdiff1d(indices, chosen, assume_unique=False).tolist())
    slots = maximum - len(selected)
    if slots > 0 and remainder:
        selected.extend(rng.choice(remainder, size=min(slots, len(remainder)), replace=False).tolist())
    return np.asarray(sorted(selected[:maximum]))


def balanced_split(pos_mask, neg_mask, chroms, seed_offset=0):
    rng = np.random.default_rng(SEED + seed_offset)

    def one_partition(test_partition):
        partition = np.asarray([chrom in TEST_CHROMS for chrom in chroms]) == test_partition
        positives = np.flatnonzero(pos_mask & partition)
        negatives = np.flatnonzero(neg_mask & partition)
        count = min(len(positives), len(negatives))
        if count < 3:
            raise RuntimeError(
                f"Too few balanced examples in {'test' if test_partition else 'train'} partition: "
                f"positive={len(positives)}, negative={len(negatives)}"
            )
        selected = np.concatenate(
            [
                rng.choice(positives, size=count, replace=False),
                rng.choice(negatives, size=count, replace=False),
            ]
        )
        selected = selected[rng.permutation(len(selected))]
        return selected, pos_mask[selected].astype(int)

    train_idx, train_y = one_partition(False)
    test_idx, test_y = one_partition(True)
    return train_idx, train_y, test_idx, test_y


def bootstrap_auroc(labels, probabilities, resamples=BOOTSTRAP_RESAMPLES):
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    scores = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        scores[index] = roc_auc_score(labels[sampled], probabilities[sampled])
    return [float(value) for value in np.quantile(scores, [0.025, 0.975])]


def fit_probe(matrix, train_idx, train_y, test_idx, test_y):
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_c = None
    best_auc = -np.inf
    for c_value in C_GRID:
        fold_scores = []
        for fold_train, fold_valid in folds.split(matrix[train_idx], train_y):
            pipe = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=c_value,
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                ]
            )
            pipe.fit(matrix[train_idx][fold_train], train_y[fold_train])
            probability = pipe.predict_proba(matrix[train_idx][fold_valid])[:, 1]
            fold_scores.append(roc_auc_score(train_y[fold_valid], probability))
        mean_auc = float(np.mean(fold_scores))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_c = c_value

    final_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=best_c,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    final_pipe.fit(matrix[train_idx], train_y)
    probability = final_pipe.predict_proba(matrix[test_idx])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    auroc = float(roc_auc_score(test_y, probability))
    result = {
        "best_C": best_c,
        "inner_cv_auroc": best_auc,
        "auroc": auroc,
        "auroc_ci95": bootstrap_auroc(test_y, probability),
        "auprc": float(average_precision_score(test_y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "mcc": float(matthews_corrcoef(test_y, prediction)),
        "f1": float(f1_score(test_y, prediction)),
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
    }
    return result, probability


def update_pooling_review(out_dir: Path):
    expected = [out_dir / f"padding_aware_probe_{genome}_v2.json" for genome in ("LA1974", "MicroTom")]
    if not all(path.exists() for path in expected):
        return
    review_path = out_dir / "pooling_review_v2.json"
    review = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.exists()
        else {"schema_version": "TomatoPGFM.pooling_review.v1"}
    )
    review["execution_status"] = "complete_pending_manuscript_integration"
    review["completed_outputs"] = [path.name for path in expected]
    review["manuscript_updated"] = False
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--genome", choices=["LA1974", "MicroTom"], required=True)
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument(
        "--window-manifest",
        type=Path,
        help="Exact ordered manifest for the authoritative stage5 embedding NPZ",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-windows", type=int, default=0, help="Stratified smoke subset; 0 uses all rows")
    parser.add_argument("--inspect-only", action="store_true", help="Print NPZ keys/shapes without loading the model")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory; defaults to results/pooling_audit in this repository",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir or (repository_root / "results" / "pooling_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    input_npz = args.input_npz or (
        args.source_root
        / "runs"
        / "stage5"
        / "genecds_embeddings"
        / f"{args.genome}_512.npz"
    )
    original = np.load(input_npz, allow_pickle=True)
    if args.inspect_only:
        inspection = {
            "input_npz": str(input_npz),
            "arrays": {
                key: {"shape": list(original[key].shape), "dtype": str(original[key].dtype)}
                for key in original.files
            },
            "contains_sequence_array": any(
                key in original.files for key in ("sequences", "seqs", "sequence", "seq")
            ),
        }
        print(json.dumps(inspection, indent=2))
        return
    rows = rows_from_npz(original)
    row_source = "sequences embedded in authoritative NPZ"
    if rows is None:
        if args.window_manifest is None:
            raise RuntimeError(
                "The authoritative NPZ does not embed sequences. Supply --window-manifest with the "
                "exact ordered 60k-window manifest; m1_holdout_v2 is not used implicitly because its "
                "sampling relationship to the stage5 NPZ is not recorded. "
                f"Available NPZ keys: {original.files}"
            )
        rows = load_manifest(args.window_manifest)
        row_source = str(args.window_manifest)
    validate_rows(rows, original)

    selected_indices = select_smoke_subset(rows, args.max_windows)
    selected_rows = [rows[index] for index in selected_indices]
    sequences = [row["seq"] for row in selected_rows]
    model, tokenizer = load_model(args.source_root, args.device)
    tomatopgfm_embeddings, nonpad_lengths = extract_embeddings(
        model, tokenizer, sequences, args.device, args.batch_size
    )

    chroms = np.asarray([row["chrom"] for row in selected_rows])
    flags = {
        "is_gene": np.asarray([row["is_gene"] for row in selected_rows], dtype=bool),
        "is_cds": np.asarray([row["is_cds"] for row in selected_rows], dtype=bool),
        "is_intergenic": np.asarray(
            [row["is_intergenic"] for row in selected_rows], dtype=bool
        ),
    }
    dnabert, dnabert_key = _first_array(
        original, ["dnabert2", "dnabert_2", "DNABERT_2", "emb_dnabert2"]
    )
    plant, plant_key = _first_array(
        original, ["plantmamba", "plant_dna_mamba", "PlantDNAMamba", "emb_plantmamba"]
    )
    matrices = {
        "TomatoPGFM_unmasked": tomatopgfm_embeddings["unmasked"],
        "TomatoPGFM_masked": tomatopgfm_embeddings["masked"],
        "TomatoPGFM_nopad": tomatopgfm_embeddings["nopad"],
        "DNABERT_2": dnabert[selected_indices],
        "PlantDNAMamba": plant[selected_indices],
    }

    results = {
        "genome": args.genome,
        "scope": "smoke" if args.max_windows else "full",
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "input_npz": str(input_npz),
        "row_source": row_source,
        "baseline_keys": {"DNABERT_2": dnabert_key, "PlantDNAMamba": plant_key},
        "selected_rows": int(len(selected_indices)),
        "nonpad_token_lengths": {
            "minimum": int(nonpad_lengths.min()),
            "maximum": int(nonpad_lengths.max()),
            "unique": [int(value) for value in np.unique(nonpad_lengths)],
        },
        "test_chromosomes": sorted(TEST_CHROMS),
        "C_grid": C_GRID,
        "pooling": {
            "historical": "mean over 512 model positions",
            "primary": "mean over non-padding positions, including special tokens",
            "sensitivity": "direct no-padding input using the same non-padding token IDs",
        },
        "tasks": {},
    }
    prediction_rows = []
    tasks = {
        "gene_vs_intergenic": (flags["is_gene"], flags["is_intergenic"]),
        "cds_vs_intergenic": (flags["is_cds"], flags["is_intergenic"]),
    }
    for task_offset, (task_name, (positive, negative)) in enumerate(tasks.items()):
        train_idx, train_y, test_idx, test_y = balanced_split(
            positive, negative, chroms, seed_offset=100 * task_offset
        )
        results["tasks"][task_name] = {"models": {}}
        for model_name, matrix in matrices.items():
            metrics, probabilities = fit_probe(
                matrix, train_idx, train_y, test_idx, test_y
            )
            results["tasks"][task_name]["models"][model_name] = metrics
            for index, label, probability in zip(test_idx, test_y, probabilities):
                prediction_rows.append(
                    {
                        "genome": args.genome,
                        "task": task_name,
                        "model": model_name,
                        "pooling_mode": model_name.replace("TomatoPGFM_", "")
                        if model_name.startswith("TomatoPGFM_")
                        else "baseline_native_masked",
                        "sample_index": int(selected_indices[index]),
                        "chromosome": str(chroms[index]),
                        "label": int(label),
                        "probability": float(probability),
                        "seed": SEED,
                        "best_C": metrics["best_C"],
                    }
                )

    suffix = f"_smoke{args.max_windows}" if args.max_windows else ""
    stem = f"padding_aware_probe_{args.genome}_v2{suffix}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with gzip.open(
        out_dir / f"{stem}_predictions.csv.gz", "wt", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    np.savez_compressed(
        out_dir / f"{stem}_embeddings.npz",
        **matrices,
        source_indices=selected_indices,
        chroms=chroms,
        nonpad_lengths=nonpad_lengths,
        **flags,
    )
    if not args.max_windows:
        update_pooling_review(out_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
