from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .shards import VTP_CLASSES, make_mlm_mask
from .tokenizer import RCKmerTokenizer


EXPECTED_TOKENIZER_SHA256 = "2c3f8cd25c1656a35f2f486aab91c798e70d523925805bbaa2e9df5cd8cd89b9"
EXPECTED_VOCAB_SIZE = 2085
TOKENIZER_SPECIALS = ("[PAD]", "[MASK]", "[UNK]", "[CLS]", "[SEP]")
DEFAULT_HOLDOUTS = ("LA1974", "MicroTom", "GCF_036512215", "holdout")
C5_UNIQUE_DIRECTED_EDGE_BASELINE = 4_449_920
FULL_UNIQUE_TOKEN_INSTANCE_MIN = 430_000_000


@dataclass(frozen=True)
class SegmentMeta:
    segment_idx: int
    segment_id: str
    length: int
    gc_fraction: float
    n_fraction: float
    sn: str = ""
    origin_accession: str = ""
    origin_chr: str = ""
    sr_rank: str = ""
    so: str = ""


@dataclass
class FeatureRow:
    vector: list[float]
    feature_valid: bool
    presence_unknown: bool
    source: str
    is_cyclic: bool = False


@dataclass
class BuildStats:
    segments_seen: int = 0
    segments_tokenized: int = 0
    short_segments: int = 0
    n_rich_segments_skipped: int = 0
    total_bp_seen: int = 0
    tokenized_bp: int = 0
    token_instances: int = 0
    unk_tokens: int = 0
    masked_token_instances: int = 0
    cyclic_token_instances: int = 0
    graph_edges_emitted: int = 0
    graph_edges_self_loop_skipped: int = 0
    graph_edges_outside_window: int = 0
    examples: int = 0
    max_token_id: int = 0
    origin_token_counts: Counter[str] = field(default_factory=Counter)
    accession_token_counts: Counter[str] = field(default_factory=Counter)
    window_sources: Counter[str] = field(default_factory=Counter)
    holdout_hits: list[dict[str, object]] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _contains_holdout(value: object, holdouts: Iterable[str]) -> str | None:
    text = str(value or "")
    lower = text.lower()
    for holdout in holdouts:
        h = str(holdout)
        if h == "holdout":
            if "holdout" in lower:
                return h
        elif h and h.lower() in lower:
            return h
    return None


def scan_holdouts(fields: dict[str, object], holdouts: Iterable[str]) -> list[dict[str, object]]:
    hits = []
    for name, value in fields.items():
        hit = _contains_holdout(value, holdouts)
        if hit is not None:
            hits.append({"field": name, "value": value, "holdout": hit})
    return hits


def load_frozen_tokenizer(path: Path, expected_sha256: str, expected_vocab_size: int) -> tuple[RCKmerTokenizer, dict[str, object]]:
    actual_sha = sha256_file(path)
    tok = RCKmerTokenizer.load(path)
    vocab_size = len(tok.vocab or {})
    report = {
        "path": str(path),
        "sha256": actual_sha,
        "expected_sha256": expected_sha256,
        "vocab_size": vocab_size,
        "expected_vocab_size": expected_vocab_size,
        "k": tok.k,
        "specials_present": all(s in (tok.vocab or {}) for s in TOKENIZER_SPECIALS),
        "status": "pass",
    }
    failures = []
    if actual_sha != expected_sha256:
        failures.append("tokenizer_sha256_mismatch")
    if vocab_size != expected_vocab_size:
        failures.append("tokenizer_vocab_size_mismatch")
    if tok.k != 6:
        failures.append("tokenizer_k_not_6")
    if not report["specials_present"]:
        failures.append("tokenizer_specials_missing")
    if failures:
        report["status"] = "fail"
        report["failures"] = failures
        raise ValueError(json.dumps(report, indent=2))
    return tok, report


def parse_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def count_tsv_rows(path: Path) -> int:
    with path.open("rt", encoding="utf-8") as fh:
        next(fh, None)
        return sum(1 for _ in fh)


def build_pilot_selection(
    total_segments: int,
    max_segments: int | None,
    sampling: str,
    block_size: int,
) -> tuple[set[int] | None, dict[str, object]]:
    if max_segments is None:
        return None, {"sampling": "full", "selected_segments": total_segments}
    if sampling == "prefix":
        selected = set(range(min(max_segments, total_segments)))
        return selected, {
            "sampling": "prefix",
            "selected_segments": len(selected),
            "segment_idx_min": min(selected) if selected else None,
            "segment_idx_max": max(selected) if selected else None,
        }
    if sampling != "uniform_blocks":
        raise ValueError(f"Unsupported pilot sampling mode: {sampling}")
    if block_size <= 0:
        raise ValueError(f"pilot_block_size must be > 0, got {block_size}")
    if max_segments >= total_segments:
        selected = set(range(total_segments))
        return selected, {
            "sampling": "uniform_blocks_degenerate_all_segments",
            "selected_segments": len(selected),
            "segment_idx_min": 0 if selected else None,
            "segment_idx_max": total_segments - 1 if selected else None,
            "blocks": 1 if selected else 0,
            "block_size": block_size,
        }
    blocks = max(1, math.ceil(max_segments / block_size))
    max_start = max(0, total_segments - block_size)
    starts = []
    if blocks == 1:
        starts = [max_start // 2]
    else:
        for i in range(blocks):
            starts.append(round(i * max_start / (blocks - 1)))
    selected: set[int] = set()
    block_ranges = []
    for start in starts:
        end = min(total_segments, start + block_size)
        block_ranges.append([start, end])
        selected.update(range(start, end))
    if len(selected) > max_segments:
        selected = set(sorted(selected)[:max_segments])
    return selected, {
        "sampling": "uniform_blocks",
        "selected_segments": len(selected),
        "requested_segments": max_segments,
        "total_segments": total_segments,
        "blocks": len(starts),
        "block_size": block_size,
        "segment_idx_min": min(selected) if selected else None,
        "segment_idx_max": max(selected) if selected else None,
        "block_ranges_preview": block_ranges[:10],
    }


def iter_segments_tsv(path: Path, holdouts: Iterable[str]) -> Iterable[tuple[SegmentMeta, list[dict[str, object]]]]:
    with path.open("rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"segment_idx", "segment_id", "length", "gc_fraction", "n_fraction"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"segments.tsv missing required columns: {missing}")
        for row in reader:
            meta = SegmentMeta(
                segment_idx=int(row["segment_idx"]),
                segment_id=row["segment_id"],
                length=int(row["length"]),
                gc_fraction=parse_float(row.get("gc_fraction")),
                n_fraction=parse_float(row.get("n_fraction")),
                sn=row.get("sn", ""),
                origin_accession=row.get("origin_accession", ""),
                origin_chr=row.get("origin_chr", ""),
                sr_rank=row.get("sr_rank", ""),
                so=row.get("so", ""),
            )
            hits = scan_holdouts(row, holdouts)
            yield meta, hits


def iter_gfa_s_lines(path: Path, holdouts: Iterable[str]) -> Iterable[tuple[str, str, dict[str, str], list[dict[str, object]]]]:
    with path.open("rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("S\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Malformed S-line in {path}: {line[:120]}")
            tags: dict[str, str] = {}
            for field in parts[3:]:
                bits = field.split(":", 2)
                if len(bits) == 3:
                    tags[bits[0]] = bits[2]
            fields: dict[str, object] = {"segment_id": parts[1], **{f"tag_{k}": v for k, v in tags.items()}}
            yield parts[1], parts[2], tags, scan_holdouts(fields, holdouts)


def iter_joined_segments(
    segments_tsv: Path,
    gfa_path: Path,
    holdouts: Iterable[str],
    max_segments: int | None = None,
    selected_indices: set[int] | None = None,
) -> Iterable[tuple[SegmentMeta, str, list[dict[str, object]]]]:
    seg_iter = iter_segments_tsv(segments_tsv, holdouts)
    gfa_iter = iter_gfa_s_lines(gfa_path, holdouts)
    emitted = 0
    for n, ((meta, seg_hits), (sid, seq, _tags, gfa_hits)) in enumerate(zip(seg_iter, gfa_iter), start=1):
        if meta.segment_id != sid:
            raise ValueError(
                "segments.tsv and GFA S-line order diverged; refusing unordered full-graph sequence join "
                f"at row {n}: {meta.segment_id!r} != {sid!r}"
            )
        if selected_indices is not None and meta.segment_idx not in selected_indices:
            continue
        yield meta, seq, [*seg_hits, *gfa_hits]
        emitted += 1
        if max_segments is not None and emitted >= max_segments:
            break


def _load_feature_records(path: Path, max_segments: int | None, selected_indices: set[int] | None = None) -> list[dict[str, object]]:
    if path.suffix.lower() in {".tsv", ".txt"}:
        with path.open("rt", encoding="utf-8", newline="") as fh:
            records = list(csv.DictReader(fh, delimiter="\t"))
        if selected_indices is not None:
            records = [r for r in records if int(r["segment_idx"]) in selected_indices]
        if max_segments is not None:
            records = [r for r in records if int(r["segment_idx"]) < max_segments]
        return records
    import pandas as pd

    df = pd.read_parquet(path)
    if selected_indices is not None:
        df = df[df["segment_idx"].isin(selected_indices)]
    if max_segments is not None:
        df = df[df["segment_idx"] < max_segments]
    return df.to_dict("records")


def load_feature_rows(path: Path, max_segments: int | None = None, selected_indices: set[int] | None = None) -> dict[int, FeatureRow]:
    records = _load_feature_records(path, max_segments, selected_indices)
    rows: dict[int, FeatureRow] = {}
    for rec in records:
        idx = int(rec["segment_idx"])
        count_valid = parse_bool(rec.get("count_is_valid"), default=True)
        presence_unknown = parse_bool(rec.get("presence_unknown"), default=not count_valid)
        acc_count = parse_float(rec.get("accession_count_value_for_model"), default=0.0) if count_valid else 0.0
        indegree = parse_float(rec.get("indegree", rec.get("in_degree", rec.get("in_degree_raw"))), default=0.0)
        outdegree = parse_float(rec.get("outdegree", rec.get("out_degree", rec.get("out_degree_raw"))), default=0.0)
        tags_text = " ".join(str(rec.get(k, "")) for k in ("annotation", "feature", "repeat", "gene", "so")).lower()
        vector = [
            math.log1p(max(0.0, parse_float(rec.get("length"), default=0.0))),
            parse_float(rec.get("gc_fraction"), default=0.0),
            math.log1p(max(0.0, indegree)),
            math.log1p(max(0.0, outdegree)),
            math.log1p(max(0.0, acc_count)),
            1.0 if "gene" in tags_text else 0.0,
            1.0 if "repeat" in tags_text or "te" in tags_text else 0.0,
            0.0,
        ]
        rows[idx] = FeatureRow(
            vector=vector,
            feature_valid=count_valid and not presence_unknown,
            presence_unknown=presence_unknown,
            source=str(rec.get("origin_accession", "")),
            is_cyclic=parse_bool(rec.get("scc_is_cyclic"), default=False),
        )
    return rows


def load_links_by_src(
    path: Path,
    max_segment_idx: int | None = None,
    selected_indices: set[int] | None = None,
) -> tuple[dict[int, list[int]], dict[str, object]]:
    links: dict[int, list[int]] = defaultdict(list)
    total = 0
    self_loops = 0
    unique: set[tuple[int, int]] = set()
    with path.open("rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"src_idx", "dst_idx"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"links.int.tsv missing required columns: {missing}")
        for row in reader:
            total += 1
            src = int(row["src_idx"])
            dst = int(row["dst_idx"])
            if src == dst:
                self_loops += 1
                continue
            if selected_indices is not None and (src not in selected_indices or dst not in selected_indices):
                continue
            if max_segment_idx is not None and (src >= max_segment_idx or dst >= max_segment_idx):
                continue
            links[src].append(dst)
            unique.add((src, dst))
    report = {
        "links_rows": total,
        "unique_directed_nonself_edges_loaded": len(unique),
        "self_loops_skipped": self_loops,
        "c5_unique_directed_edge_baseline": C5_UNIQUE_DIRECTED_EDGE_BASELINE,
        "baseline_reconciled": len(unique) == C5_UNIQUE_DIRECTED_EDGE_BASELINE if max_segment_idx is None else None,
        "scope": "selected_indices" if selected_indices is not None else ("full" if max_segment_idx is None else f"subset_segment_idx_lt_{max_segment_idx}"),
    }
    return links, report


def classify_origin(meta: SegmentMeta) -> str:
    source = meta.origin_accession or meta.sn or ""
    if source == "SL6.0" or source.startswith("SL6.0"):
        return "SL6.0"
    if source.startswith("LA"):
        return "wild_or_LA_panel"
    if source:
        return "panel_or_cultivar"
    return "unknown"


def window_source(origin_counts: Counter[str]) -> str:
    if not origin_counts:
        return "unknown"
    return origin_counts.most_common(1)[0][0]


@dataclass
class WindowBuilder:
    out_fh: object
    tokenizer: RCKmerTokenizer
    links_by_src: dict[int, list[int]]
    seq_len: int
    stats: BuildStats
    pilot_token_limit: int | None = None
    current_ids: list[int] = field(default_factory=list)
    current_node_ids: list[int] = field(default_factory=list)
    current_graph_features: list[list[float]] = field(default_factory=list)
    current_feature_mask: list[int] = field(default_factory=list)
    current_segment_positions: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    current_origin_counts: Counter[str] = field(default_factory=Counter)
    current_accession_counts: Counter[str] = field(default_factory=Counter)
    window_id: int = 0
    stopped_by_token_limit: bool = False

    def add_segment_tokens(self, meta: SegmentMeta, token_ids: list[int], feature: FeatureRow, source: str) -> None:
        origin = classify_origin(meta)
        for token_id in token_ids:
            if self.pilot_token_limit is not None and self.stats.token_instances >= self.pilot_token_limit:
                self.stopped_by_token_limit = True
                return
            if len(self.current_ids) >= self.seq_len:
                self.flush()
            pos = len(self.current_ids)
            self.current_ids.append(int(token_id))
            self.current_node_ids.append(meta.segment_idx)
            self.current_graph_features.append(feature.vector)
            self.current_feature_mask.append(1 if feature.feature_valid else 0)
            self.current_segment_positions[meta.segment_idx].append(pos)
            self.current_origin_counts[origin] += 1
            self.current_accession_counts[source or "unknown"] += 1
            self.stats.origin_token_counts[origin] += 1
            self.stats.accession_token_counts[source or "unknown"] += 1
            self.stats.token_instances += 1
            self.stats.max_token_id = max(self.stats.max_token_id, int(token_id))
            if token_id == self.tokenizer.unk_id:
                self.stats.unk_tokens += 1
            if not feature.feature_valid:
                self.stats.masked_token_instances += 1
            if feature.is_cyclic:
                self.stats.cyclic_token_instances += 1

    def flush(self) -> None:
        if not self.current_ids:
            return
        edge_index: list[tuple[int, int]] = []
        for src_idx, src_positions in self.current_segment_positions.items():
            src_anchor = src_positions[0]
            for dst_idx in self.links_by_src.get(src_idx, ()):
                dst_positions = self.current_segment_positions.get(dst_idx)
                if not dst_positions:
                    self.stats.graph_edges_outside_window += 1
                    continue
                dst_anchor = dst_positions[0]
                if src_anchor == dst_anchor:
                    self.stats.graph_edges_self_loop_skipped += 1
                    continue
                edge_index.append((src_anchor, dst_anchor))
        self.stats.graph_edges_emitted += len(edge_index)
        source = window_source(self.current_origin_counts)
        self.stats.window_sources[source] += 1
        example = {
            "input_ids": self.current_ids,
            "graph_features": self.current_graph_features,
            "accession_id": source,
            "path_id": f"pan66_window_{self.window_id:012d}",
            "bubble_id": "pan66_context",
            "node_ids": self.current_node_ids,
            "graph_edge_index": edge_index,
            "sequence_edge_index": [],
            "real_dag_edges": [],
            "variant_class": VTP_CLASSES["none"],
            "feature_mask": self.current_feature_mask,
            "mlm_mask": make_mlm_mask(len(self.current_ids), seed=self.window_id),
            "cpc_positive_path_id": f"pan66_window_{self.window_id:012d}",
            "cpc_negative_path_ids": [],
            "sample_weight": 1.0,
            "window_source": source,
        }
        self.out_fh.write(json.dumps(example, separators=(",", ":")) + "\n")
        self.window_id += 1
        self.stats.examples += 1
        self.current_ids = []
        self.current_node_ids = []
        self.current_graph_features = []
        self.current_feature_mask = []
        self.current_segment_positions = defaultdict(list)
        self.current_origin_counts = Counter()
        self.current_accession_counts = Counter()


def read_cpc_weight_report(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"provided": False, "status": "not_checked"}
    import pandas as pd

    df = pd.read_parquet(path)
    cols = set(df.columns)
    # Physical duplication means byte-for-byte/logical row duplication. A single
    # bubble may legitimately produce multiple positive pairs, so subset keys
    # such as bubble_id would create false failures.
    safe = df.astype(str)
    duplicate_rows = int(safe.duplicated().sum())
    duplicate_scope = "all_columns_as_string_exact_row"
    report: dict[str, object] = {
        "provided": True,
        "path": str(path),
        "rows": int(len(df)),
        "has_sample_weight": "sample_weight" in cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_scope": duplicate_scope,
        "status": "pass",
    }
    if "sample_weight" in cols and len(df):
        report["sample_weight_min"] = float(df["sample_weight"].min())
        report["sample_weight_max"] = float(df["sample_weight"].max())
    failures = []
    if "sample_weight" not in cols:
        failures.append("missing_sample_weight")
    if report["duplicate_rows"]:
        failures.append("physical_duplicate_rows_detected")
    if failures:
        report["status"] = "fail"
        report["failures"] = failures
    return report


def build_pan66_shard(args: argparse.Namespace) -> dict[str, object]:
    holdouts = tuple(args.holdout)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")
    preflight_path = Path(args.preflight) if args.preflight else out.with_suffix(".preflight.json")

    tokenizer, tokenizer_report = load_frozen_tokenizer(Path(args.tokenizer), args.expected_tokenizer_sha256, args.expected_vocab_size)
    pilot_mode = args.pilot_max_segments is not None or args.pilot_token_limit is not None
    selected_indices: set[int] | None = None
    sampling_report: dict[str, object] = {"sampling": "full"}
    max_link_idx = None
    if args.pilot_max_segments is not None:
        if args.pilot_sampling == "prefix":
            max_link_idx = args.pilot_max_segments
        total_segments = count_tsv_rows(Path(args.segments))
        selected_indices, sampling_report = build_pilot_selection(
            total_segments=total_segments,
            max_segments=args.pilot_max_segments,
            sampling=args.pilot_sampling,
            block_size=args.pilot_block_size,
        )
    links_by_src, links_report = load_links_by_src(Path(args.links), max_link_idx, selected_indices)
    feature_max = args.pilot_max_segments if args.pilot_sampling == "prefix" else None
    features = load_feature_rows(Path(args.features), feature_max, selected_indices)
    cpc_report = read_cpc_weight_report(Path(args.cpc_reweighted)) if args.cpc_reweighted else {"provided": False, "status": "not_checked"}

    preflight = {
        "decision": "ALLOW_PAN66_PILOT_SHARD_BUILD" if pilot_mode else "ALLOW_PAN66_FULL_SHARD_BUILD",
        "pilot_mode": pilot_mode,
        "tokenizer": tokenizer_report,
        "links": links_report,
        "pilot_sampling": sampling_report,
        "feature_rows_loaded": len(features),
        "cpc_reweighted": cpc_report,
        "holdouts": list(holdouts),
        "red_lines": {
            "tokenizer_loaded_not_retrained": True,
            "old_p_line_simple_bubble_path_used": False,
            "sequence_adjacency_labeled_as_graph": False,
            "graph_edges_from_links_int_topology": True,
        },
        "status": "pass" if cpc_report.get("status") in {"pass", "not_checked"} else "fail",
    }
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if preflight["status"] != "pass":
        raise RuntimeError(f"pan66 builder preflight failed; see {preflight_path}")

    stats = BuildStats()
    with out.open("wt", encoding="utf-8") as fh:
        builder = WindowBuilder(
            out_fh=fh,
            tokenizer=tokenizer,
            links_by_src=links_by_src,
            seq_len=args.seq_len,
            stats=stats,
            pilot_token_limit=args.pilot_token_limit,
        )
        iter_max = args.pilot_max_segments if args.pilot_sampling == "prefix" else None
        for meta, seq, holdout_hits in iter_joined_segments(Path(args.segments), Path(args.gfa), holdouts, iter_max, selected_indices):
            stats.segments_seen += 1
            stats.total_bp_seen += meta.length
            if holdout_hits:
                for hit in holdout_hits[:20]:
                    stats.holdout_hits.append({"segment_idx": meta.segment_idx, "segment_id": meta.segment_id, **hit})
            if meta.length < tokenizer.k:
                stats.short_segments += 1
                continue
            if meta.n_fraction > args.n_fraction_skip:
                stats.n_rich_segments_skipped += 1
                continue
            feat = features.get(meta.segment_idx, FeatureRow(vector=[0.0] * 8, feature_valid=False, presence_unknown=True, source=meta.origin_accession))
            token_ids = tokenizer.encode_kmers(seq)
            if not token_ids:
                stats.short_segments += 1
                continue
            stats.segments_tokenized += 1
            stats.tokenized_bp += len(token_ids) * tokenizer.k
            builder.add_segment_tokens(
                meta=meta,
                token_ids=token_ids,
                feature=feat,
                source=meta.origin_accession or feat.source or meta.sn,
            )
            if builder.stopped_by_token_limit:
                break
        builder.flush()

    failures: list[str] = []
    if stats.holdout_hits:
        failures.append("holdout_hit_detected")
    if stats.max_token_id >= args.expected_vocab_size:
        failures.append("token_id_exceeds_vocab_size")
    if not pilot_mode and stats.token_instances < FULL_UNIQUE_TOKEN_INSTANCE_MIN:
        failures.append("full_unique_token_instance_count_below_430M")
    if not pilot_mode and links_report.get("baseline_reconciled") is not True:
        failures.append("full_c5_edge_baseline_not_reconciled")
    projected_edge_candidates = (
        stats.graph_edges_emitted
        + stats.graph_edges_outside_window
        + stats.graph_edges_self_loop_skipped
    )
    graph_edge_retention_fraction = (
        stats.graph_edges_emitted / projected_edge_candidates if projected_edge_candidates else 0.0
    )
    graph_edge_outside_window_fraction = (
        stats.graph_edges_outside_window / projected_edge_candidates if projected_edge_candidates else 0.0
    )

    manifest = {
        "decision": "PAN66_PILOT_SHARD_BUILT_NOT_FULL_PASS" if pilot_mode else "PAN66_FULL_SHARD_BUILT_PENDING_C13_AUDIT",
        "pilot_mode": pilot_mode,
        "shard_path": str(out),
        "preflight_path": str(preflight_path),
        "seq_len": args.seq_len,
        "stride": args.seq_len,
        "pilot_sampling": sampling_report,
        "tokenizer_vocab_size": args.expected_vocab_size,
        "tokenizer_sha256": tokenizer_report["sha256"],
        "examples": stats.examples,
        "segments_seen": stats.segments_seen,
        "segments_tokenized": stats.segments_tokenized,
        "short_segments_zero_token_accounted": stats.short_segments,
        "n_rich_segments_skipped": stats.n_rich_segments_skipped,
        "total_bp_seen": stats.total_bp_seen,
        "tokenized_bp": stats.tokenized_bp,
        "tokenized_bp_fraction": stats.tokenized_bp / stats.total_bp_seen if stats.total_bp_seen else 0.0,
        "unique_token_instances_single_pass": stats.token_instances,
        "unique_token_full_gate_min": FULL_UNIQUE_TOKEN_INSTANCE_MIN,
        "unique_token_full_gate_applied": not pilot_mode,
        "unk_tokens": stats.unk_tokens,
        "unk_token_fraction": stats.unk_tokens / stats.token_instances if stats.token_instances else 0.0,
        "masked_token_instances": stats.masked_token_instances,
        "masked_token_fraction": stats.masked_token_instances / stats.token_instances if stats.token_instances else 0.0,
        "cyclic_token_instances": stats.cyclic_token_instances,
        "graph_edge_window_projection": {
            "candidate_edges_from_loaded_topology": projected_edge_candidates,
            "emitted_in_window_edges": stats.graph_edges_emitted,
            "outside_window_edges_dropped": stats.graph_edges_outside_window,
            "self_loop_edges_skipped": stats.graph_edges_self_loop_skipped,
            "retention_fraction": graph_edge_retention_fraction,
            "outside_window_fraction": graph_edge_outside_window_fraction,
            "interpretation": "true topology is projected onto per-example token windows; edges crossing window boundaries are reported, not mislabeled as sequence adjacency",
        },
        "graph_edges_emitted": stats.graph_edges_emitted,
        "graph_edges_outside_window": stats.graph_edges_outside_window,
        "graph_edges_self_loop_skipped": stats.graph_edges_self_loop_skipped,
        "c5_edge_reconciliation": links_report,
        "origin_proxy_token_counts": dict(stats.origin_token_counts),
        "accession_proxy_token_counts_top50": dict(stats.accession_token_counts.most_common(50)),
        "window_source_counts": dict(stats.window_sources),
        "g7_path_level_status": "not_final_from_origin_proxy_builder_pilot",
        "g8_backbone_coverage_status": "reported_not_thresholded_until_builder_pilot_audit",
        "cpc_sample_weight_status": cpc_report.get("status"),
        "holdout_hits": stats.holdout_hits[:50],
        "max_token_id": stats.max_token_id,
        "failures": failures,
        "status": "pass" if not failures else "fail",
        "allow_full_shard_generation": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="build-pan66-shard")
    p.add_argument("--gfa", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument("--links", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--preflight", default=None)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--pilot-max-segments", type=int, default=None)
    p.add_argument("--pilot-token-limit", type=int, default=None)
    p.add_argument("--pilot-sampling", choices=("prefix", "uniform_blocks"), default="prefix")
    p.add_argument("--pilot-block-size", type=int, default=1000)
    p.add_argument("--n-fraction-skip", type=float, default=0.50)
    p.add_argument("--cpc-reweighted", default=None)
    p.add_argument("--expected-tokenizer-sha256", default=EXPECTED_TOKENIZER_SHA256)
    p.add_argument("--expected-vocab-size", type=int, default=EXPECTED_VOCAB_SIZE)
    p.add_argument("--holdout", action="append", default=list(DEFAULT_HOLDOUTS))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(build_pan66_shard(args), indent=2))


if __name__ == "__main__":
    main()
