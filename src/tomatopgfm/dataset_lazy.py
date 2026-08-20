"""Lazy, byte-offset-indexed JSONL shard dataset for TomatoPGFM pretraining.

WHY THIS EXISTS (measured, not guessed):
  The eager ``ShardDataset`` (dataset.py) reads the entire JSONL into a Python
  list. Measured cost on the 512-stage full shard (842,963 examples):
  240.3 KB/example as live Python objects -> 202.6 GB per rank. Under 4-rank
  DDP each rank holds its own copy -> ~810 GB, over the 700 GB safety line on a
  968 GB host. One fragmentation spike or stray process and the OOM killer ends
  a multi-hour burn mid-run. See m0_ram_probe.py for the measurement.

  This class instead holds only the byte offset of each line (8 bytes/example
  -> ~6.7 MB for the full shard, identical across ranks) and seeks+reads+parses
  one line per __getitem__. RAM is O(num_lines * 8 bytes), not O(shard bytes).

The per-item dict shape is byte-identical to ShardDataset.__getitem__ so the
existing ``collate`` (dataset.py) consumes it unchanged.

The offset index is cached next to the shard as ``<shard>.offidx.npy`` so the
one-time newline scan of a 53 GB file is paid once, not every run / every rank.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import FEATURE_DIM  # noqa: F401  (kept for parity / external import)


def _build_offsets(shard_path: Path) -> np.ndarray:
    """Scan the file once for line-start byte offsets of non-empty lines."""
    offsets: list[int] = []
    with open(shard_path, "rb") as fh:
        pos = 0
        for line in fh:
            stripped = line.strip()
            if stripped:
                offsets.append(pos)
            pos += len(line)
    return np.asarray(offsets, dtype=np.int64)


def _load_or_build_index(shard_path: Path, rebuild: bool = False) -> np.ndarray:
    idx_path = shard_path.with_suffix(shard_path.suffix + ".offidx.npy")
    if idx_path.exists() and not rebuild:
        # Index is valid only if it predates no shard modification.
        if idx_path.stat().st_mtime >= shard_path.stat().st_mtime:
            return np.load(idx_path)
    offsets = _build_offsets(shard_path)
    try:
        np.save(idx_path, offsets)
    except OSError:
        pass  # read-only fs: fall back to in-RAM index (still cheap)
    return offsets


class LazyShardDataset(Dataset):
    """Byte-offset-indexed JSONL shard. RAM = O(num_lines), not O(file size)."""

    def __init__(self, shard_path: str | Path, node_feature_table: dict | None = None,
                 rebuild_index: bool = False):
        self.shard_path = Path(shard_path)
        if not self.shard_path.exists():
            raise FileNotFoundError(f"shard not found: {self.shard_path}")
        self.offsets = _load_or_build_index(self.shard_path, rebuild=rebuild_index)
        self.node_feature_table = node_feature_table or {}
        self._fh = None
        self._fh_pid = None

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def _handle(self):
        # One file handle per process (DataLoader workers are forked): reopen if
        # the cached handle belongs to a different pid.
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            self._fh = open(self.shard_path, "rb")
            self._fh_pid = pid
        return self._fh

    def _graph_features(self, node_ids: list) -> list[list[float]]:
        zero = [0.0] * FEATURE_DIM
        return [list(self.node_feature_table.get(str(n), zero)) for n in node_ids]

    def __getitem__(self, idx: int) -> dict:
        fh = self._handle()
        fh.seek(int(self.offsets[idx]))
        ex = json.loads(fh.readline())
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

    def __getstate__(self):
        # Don't pickle the open file handle across the fork boundary.
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_pid"] = None
        return state
