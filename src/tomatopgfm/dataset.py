from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .features import NODE_FEATURE_FIELDS

PAD_ID = 0
FEATURE_DIM = len(NODE_FEATURE_FIELDS)


class ShardDataset(Dataset):
    """Reads JSONL shards (one ShardExample per line) into tensor-ready dicts.

    ``node_feature_table`` maps node id -> feature vector so per-token graph
    features are gathered from the *real* graph node assigned to each token
    (the token->node mapping carried in each example's ``node_ids``). Tokens
    without a known node get a zero vector.
    """

    def __init__(self, shard_path: Path, node_feature_table: dict | None = None):
        self.examples = []
        with Path(shard_path).open("rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.node_feature_table = node_feature_table or {}

    def __len__(self) -> int:
        return len(self.examples)

    def _graph_features(self, node_ids: list) -> list[list[float]]:
        zero = [0.0] * FEATURE_DIM
        return [list(self.node_feature_table.get(str(n), zero)) for n in node_ids]

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        node_ids = ex.get("node_ids", [])
        graph_features = ex.get("graph_features")
        if graph_features is None:
            graph_features = self._graph_features(node_ids)
        return {
            "input_ids": ex["input_ids"],
            "mlm_mask": ex["mlm_mask"],
            "feature_mask": ex["feature_mask"],
            "graph_features": graph_features,
            "graph_edge_index": ex.get("graph_edge_index", ex.get("real_dag_edges", [])),
            "sequence_edge_index": ex.get("sequence_edge_index", []),
            "variant_class": int(ex["variant_class"]),
            "path_id": ex["path_id"],
            "cpc_positive_path_id": ex.get("cpc_positive_path_id", ex["path_id"]),
            "sample_weight": float(ex.get("sample_weight", 1.0)),
        }


def collate(batch: list[dict], pad_id: int = PAD_ID) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    b = len(batch)

    def pad_int(key: str, fill: int) -> torch.Tensor:
        return torch.tensor([row[key] + [fill] * (max_len - len(row[key])) for row in batch], dtype=torch.long)

    input_ids = pad_int("input_ids", pad_id)
    mlm_mask = pad_int("mlm_mask", 0)
    feature_mask = pad_int("feature_mask", 0)

    gf = torch.zeros(b, max_len, FEATURE_DIM)
    for i, row in enumerate(batch):
        feats = row["graph_features"]
        if feats:
            gf[i, : len(feats)] = torch.tensor(feats, dtype=torch.float32)

    variant_class = torch.tensor([row["variant_class"] for row in batch], dtype=torch.long)

    # Build a single (2, E) edge_index over the flattened (B*max_len) node set.
    # Each example's graph_edge_index holds *local* token positions (0..len-1); we
    # offset by i*max_len so messages never cross example boundaries, and drop
    # any endpoint that would land in the padding region.
    src_list: list[int] = []
    dst_list: list[int] = []
    for i, row in enumerate(batch):
        n = len(row["input_ids"])
        offset = i * max_len
        for edge in row.get("graph_edge_index", []):
            s, d = int(edge[0]), int(edge[1])
            if 0 <= s < n and 0 <= d < n:
                src_list.append(offset + s)
                dst_list.append(offset + d)
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    # CPC positive index within the batch; fall back to a neighbour when the
    # positive path is not co-batched so the (self-masked) InfoNCE stays finite.
    path_to_idx = {row["path_id"]: i for i, row in enumerate(batch)}
    positive_indices: list[int] = []
    for i, row in enumerate(batch):
        j = path_to_idx.get(row["cpc_positive_path_id"])
        if j is None or j == i:
            j = (i + 1) % b
        positive_indices.append(j)
    positive_index = torch.tensor(positive_indices, dtype=torch.long)
    sample_weight = torch.tensor([float(row.get("sample_weight", 1.0)) for row in batch], dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "mlm_mask": mlm_mask,
        "feature_mask": feature_mask,
        "graph_features": gf,
        "variant_class": variant_class,
        "edge_index": edge_index,
        "cpc_positive_index": positive_index,
        "sample_weight": sample_weight,
    }
