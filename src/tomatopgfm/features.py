from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .graph import Edge, Segment, adjacency


@dataclass(frozen=True)
class NodeFeature:
    node_id: str
    length_log: float
    gc_fraction: float
    indegree_log: float
    outdegree_log: float
    accession_count_log: float
    has_gene_tag: int
    has_repeat_tag: int
    is_random_only: int = 0


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    denom = sum(1 for c in seq if c in "ACGT")
    if denom == 0:
        return 0.0
    return (seq.count("G") + seq.count("C")) / denom


def accession_count(tags: dict[str, str]) -> int:
    for key in ("SN", "SR", "BO", "RC"):
        if key in tags:
            return max(1, len(str(tags[key]).split(",")))
    return 1


def build_node_features(
    segments: dict[str, Segment],
    edges: list[Edge],
    node_accessions: dict[str, set[str]] | None = None,
) -> list[NodeFeature]:
    out_adj = adjacency(edges)
    indeg = {sid: 0 for sid in segments}
    for e in edges:
        indeg[e.dst] = indeg.get(e.dst, 0) + 1
    feats: list[NodeFeature] = []
    for sid, seg in segments.items():
        tags_text = " ".join(f"{k}:{v}" for k, v in seg.tags.items()).lower()
        # Prefer the real per-node accession membership parsed from GFA P/W lines
        # (node_accessions); minigraph S-line SN tags carry only a single sample
        # name, so the tag-based count is ~1 for most nodes and near-constant.
        if node_accessions is not None:
            acc_count = max(1, len(node_accessions.get(sid, ())))
        else:
            acc_count = accession_count(seg.tags)
        feats.append(
            NodeFeature(
                node_id=sid,
                length_log=math.log1p(seg.length),
                gc_fraction=gc_fraction(seg.seq),
                indegree_log=math.log1p(indeg.get(sid, 0)),
                outdegree_log=math.log1p(len(out_adj.get(sid, ()))),
                accession_count_log=math.log1p(acc_count),
                has_gene_tag=1 if "gene" in tags_text else 0,
                has_repeat_tag=1 if "repeat" in tags_text or "te" in tags_text else 0,
                is_random_only=0,
            )
        )
    return feats


NODE_FEATURE_FIELDS = (
    "length_log",
    "gc_fraction",
    "indegree_log",
    "outdegree_log",
    "accession_count_log",
    "has_gene_tag",
    "has_repeat_tag",
    "is_random_only",
)


def node_feature_vector(f: NodeFeature) -> list[float]:
    """Dense numeric vector for a node, in a fixed field order."""
    return [float(getattr(f, name)) for name in NODE_FEATURE_FIELDS]


def feature_table(features: list[NodeFeature]) -> dict[str, list[float]]:
    """Map node id -> feature vector for token->node graph-feature lookup."""
    return {f.node_id: node_feature_vector(f) for f in features}


def audit_node_features(features: list[NodeFeature]) -> dict:
    if not features:
        return {"nodes": 0, "status": "fail", "reason": "no features"}
    random_only = sum(f.is_random_only for f in features)
    variable_gc = len({round(f.gc_fraction, 3) for f in features[:10000]}) > 1
    return {
        "nodes": len(features),
        "random_only_nodes": random_only,
        "variable_gc_observed": variable_gc,
        "status": "pass" if random_only == 0 and variable_gc else "fail",
    }


def write_node_features(features: list[NodeFeature], out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(asdict(f)) for f in features) + "\n", encoding="utf-8")
    report = audit_node_features(features)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

