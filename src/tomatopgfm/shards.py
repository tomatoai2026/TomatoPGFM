from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


VTP_CLASSES = {
    "none": 0,
    "snp": 1,
    "small_indel": 2,
    "sv_ins": 3,
    "sv_del": 4,
    "sv_complex": 5,
    "te_gypsy": 6,
    "te_copia": 7,
    "te_sine": 8,
    "te_dna": 9,
}


@dataclass(frozen=True)
class ShardExample:
    input_ids: list[int]
    accession_id: str
    path_id: str
    bubble_id: str
    node_ids: list[int]
    variant_class: int
    feature_mask: list[int]
    mlm_mask: list[int]
    cpc_positive_path_id: str
    cpc_negative_path_ids: list[str]
    graph_edge_index: list[tuple[int, int]] = field(default_factory=list)
    sequence_edge_index: list[tuple[int, int]] = field(default_factory=list)
    real_dag_edges: list[tuple[int, int]] = field(default_factory=list)
    sample_weight: float = 1.0


def make_mlm_mask(length: int, pad_id: int = 0, mask_rate: float = 0.15, seed: int = 42) -> list[int]:
    rng = random.Random(seed)
    mask = []
    for i in range(length):
        if i == 0:
            mask.append(0)
        else:
            mask.append(1 if rng.random() < mask_rate else 0)
    return mask


def validate_vtp_labels(labels: Iterable[int], n_classes: int = len(VTP_CLASSES)) -> dict:
    labels = list(labels)
    bad = [x for x in labels if x < 0 or x >= n_classes]
    present = sorted(set(labels))
    return {"n": len(labels), "present": present, "bad": bad[:20], "status": "pass" if not bad else "fail"}


def non_reference_fraction(examples: Iterable[ShardExample], reference: str = "SL6.0") -> float:
    ex = list(examples)
    if not ex:
        return 0.0
    return sum(1 for e in ex if e.accession_id != reference) / len(ex)


def write_jsonl_shard(examples: list[ShardExample], out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(asdict(e)) for e in examples) + "\n", encoding="utf-8")
    frac = non_reference_fraction(examples)
    labels = validate_vtp_labels(e.variant_class for e in examples)
    report = {"examples": len(examples), "non_reference_fraction": frac, "vtp": labels, "status": "pass"}
    (out.with_suffix(".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def dummy_examples(n: int = 16, seq_len: int = 64, vocab_size: int = 128) -> list[ShardExample]:
    rng = np.random.default_rng(7)
    examples = []
    for i in range(n):
        accession = "SL6.0" if i % 2 == 0 else f"ACC{i:03d}"
        ids = rng.integers(3, vocab_size, size=seq_len).tolist()
        examples.append(
            ShardExample(
                input_ids=ids,
                accession_id=accession,
                path_id=f"path_{i}",
                bubble_id=f"bubble_{i//2}",
                node_ids=list(range(i * seq_len, (i + 1) * seq_len)),
                real_dag_edges=[(j, j + 1) for j in range(seq_len - 1)],
                variant_class=i % len(VTP_CLASSES),
                feature_mask=[1] * seq_len,
                mlm_mask=make_mlm_mask(seq_len, seed=i),
                cpc_positive_path_id=f"path_{i ^ 1}",
                cpc_negative_path_ids=[f"path_{(i + 3) % n}", f"path_{(i + 5) % n}"],
                sample_weight=1.0,
            )
        )
    return examples
