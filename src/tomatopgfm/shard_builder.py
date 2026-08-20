from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .features import build_node_features, feature_table
from .graph import find_simple_bubbles, node_accession_map, parse_gfa
from .path_sampling import PathSample, samples_from_bubbles
from .shards import VTP_CLASSES, ShardExample, make_mlm_mask, write_jsonl_shard
from .tokenizer import RCKmerTokenizer

# Indel/SV size boundary (bp). Length deltas at/under this are indels; larger
# are structural variants. A coarse, sequence-derived split -- NOT a substitute
# for a real variant caller.
SV_MIN_BP = 50


def _interior_nodes(sample: PathSample, source: str, sink: str) -> list[str]:
    interior = [n for n in sample.node_ids if n not in (source, sink)]
    return interior or list(sample.node_ids)


def _branch_sequence(node_ids: list[str], segments: dict) -> str:
    parts = []
    for sid in node_ids:
        seg = segments.get(sid)
        if seg is not None and seg.seq not in ("", "*"):
            parts.append(seg.seq.upper())
    return "".join(parts)


def classify_variant(alt_seq: str, ref_seq: str, n_interior: int, is_reference: bool) -> str:
    """Type an allele by comparing its sequence to the bubble's reference allele.

    This is a real *sequence-level* comparison (length delta + identity), a clear
    improvement over a branch-length heuristic, but it is intentionally limited:
    it returns none/snp/small_indel/sv_ins/sv_del/sv_complex only. Transposable-
    element family labels (te_gypsy/te_copia/te_sine/te_dna) require RepeatMasker/
    trf on the A100 host and are NOT inferable from the graph alone, so they are
    never emitted here. Callers that have TE annotations should override.
    """
    if is_reference:
        return "none"
    delta = len(alt_seq) - len(ref_seq)
    if delta == 0:
        return "none" if alt_seq == ref_seq else "snp"
    if n_interior > 1:
        return "sv_complex"
    if abs(delta) <= SV_MIN_BP:
        return "small_indel"
    return "sv_ins" if delta > 0 else "sv_del"


def build_examples_from_gfa(
    gfa_path: Path,
    tokenizer: RCKmerTokenizer | None = None,
    reference_accession: str = "SL6.0",
    max_segments: int | None = None,
    mlm_seed: int = 0,
) -> tuple[list[ShardExample], dict[str, list[float]]]:
    """Assemble real ShardExamples + a node feature table from a graph GFA.

    The full chain that was previously missing: parse_gfa -> node_accession_map
    -> build_node_features/feature_table -> find bubbles -> samples_from_bubbles
    (with real per-node accession membership) -> tokenize each branch's segment
    sequence (tokens line up 1:1 with their graph node) -> sequence-derived VTP
    class -> ShardExample. Returns (examples, node_feature_table) so the table can
    be handed to ShardDataset for real per-node graph features.
    """
    gfa_path = Path(gfa_path)
    segments, edges = parse_gfa(gfa_path, max_segments=max_segments)
    node_acc = node_accession_map(gfa_path)
    feats = build_node_features(segments, edges, node_accessions=node_acc)
    ftable = feature_table(feats)

    bubbles = find_simple_bubbles(edges)
    samples = samples_from_bubbles(bubbles, reference_accession=reference_accession, node_accessions=node_acc or None)

    if tokenizer is None:
        tokenizer = RCKmerTokenizer(k=6)
        tokenizer.train([seg.seq for seg in segments.values() if seg.seq not in ("", "*")], vocab_size=4096)

    # Per-bubble reference allele sequence, for the variant comparison.
    by_bubble: dict[str, list[PathSample]] = defaultdict(list)
    for s in samples:
        by_bubble[s.bubble_id].append(s)
    bubble_src_sink = {b.bubble_id: (b.source, b.sink) for b in bubbles}
    ref_seq: dict[str, str] = {}
    for bid, group in by_bubble.items():
        src, sink = bubble_src_sink.get(bid, ("", ""))
        ref = next((g for g in group if g.role == "reference"), group[0])
        ref_seq[bid] = _branch_sequence(_interior_nodes(ref, src, sink), segments)

    examples: list[ShardExample] = []
    for i, s in enumerate(samples):
        src, sink = bubble_src_sink.get(s.bubble_id, ("", ""))
        # Tokenize segment-by-segment so every token carries the node it came from.
        input_ids: list[int] = []
        node_ids: list[str] = []
        for sid in s.node_ids:
            seg = segments.get(sid)
            if seg is None or seg.seq in ("", "*"):
                continue
            kids = tokenizer.encode_kmers(seg.seq)
            input_ids.extend(kids)
            node_ids.extend([sid] * len(kids))
        if not input_ids:
            continue
        # The path is a real walk; consecutive tokens are graph-adjacent (within a
        # segment trivially, across a boundary via a real GFA L-edge). Edges are
        # local token positions, matching what collate expects.
        real_dag_edges = [(j, j + 1) for j in range(len(input_ids) - 1)]
        n_interior = len(_interior_nodes(s, src, sink))
        vclass = classify_variant(_branch_sequence(_interior_nodes(s, src, sink), segments), ref_seq.get(s.bubble_id, ""), n_interior, s.role == "reference")
        examples.append(
            ShardExample(
                input_ids=input_ids,
                accession_id=s.accession_id,
                path_id=s.path_id,
                bubble_id=s.bubble_id,
                node_ids=node_ids,
                graph_edge_index=real_dag_edges,
                real_dag_edges=real_dag_edges,
                variant_class=VTP_CLASSES.get(vclass, 0),
                feature_mask=[1] * len(input_ids),
                mlm_mask=make_mlm_mask(len(input_ids), seed=mlm_seed + i),
                cpc_positive_path_id=s.positive_path_id,
                cpc_negative_path_ids=s.negative_path_ids,
            )
        )
    return examples, ftable


def build_shards(gfa_path: Path, out_shard: Path, reference_accession: str = "SL6.0", max_segments: int | None = None) -> dict:
    """Build real shards from a GFA and write them (+ a node-feature table) to disk."""
    import json

    examples, ftable = build_examples_from_gfa(gfa_path, reference_accession=reference_accession, max_segments=max_segments)
    out_shard = Path(out_shard)
    report = write_jsonl_shard(examples, out_shard)
    table_path = out_shard.with_suffix(".node_features.json")
    table_path.write_text(json.dumps(ftable), encoding="utf-8")
    report["node_feature_table"] = str(table_path)
    report["examples_built"] = len(examples)
    return report


# A tiny synthetic GFA with one real bubble (s1 -> {s2|s3} -> s4), two accessions
# carried on real P-lines. Used only to exercise the builder end-to-end on CPU;
# it does NOT stand in for a real minigraph tomato graph.
_SYNTHETIC_GFA = "\n".join(
    [
        "S\ts1\tACGTACGTACGTACGT\tSN:Z:SL6.0",
        "S\ts2\tTTTTAAAACCCC\tSN:Z:SL6.0",
        "S\ts3\tTTTTAAAACCCCGGGGTTTTAAAACCCCGGGG\tSN:Z:ACC1",
        "S\ts4\tGGGGCCCCAAAATTTT\tSN:Z:SL6.0",
        "L\ts1\t+\ts2\t+\t0M",
        "L\ts1\t+\ts3\t+\t0M",
        "L\ts2\t+\ts4\t+\t0M",
        "L\ts3\t+\ts4\t+\t0M",
        "P\tSL6.0#1#chr1\ts1+,s2+,s4+\t*",
        "P\tACC1#1#chr1\ts1+,s3+,s4+\t*",
    ]
)


def build_shards_smoke() -> dict:
    """End-to-end check: synthetic GFA -> real ShardExamples -> ShardDataset (with
    the real node-feature table) -> collate -> model forward. Verifies the bridge
    that was previously missing actually carries real per-node features and a
    real edge_index into the model. CPU-only; proves wiring, not biology.
    """
    import tempfile

    import torch

    from .dataset import FEATURE_DIM, ShardDataset, collate
    from .model import TomatoPGFMConfig, TomatoPGFM

    with tempfile.TemporaryDirectory() as tmp:
        gfa = Path(tmp) / "synthetic.gfa"
        gfa.write_text(_SYNTHETIC_GFA + "\n", encoding="utf-8")
        shard = Path(tmp) / "shard.jsonl"
        report = build_shards(gfa, shard)

        import json as _json

        table = _json.loads(Path(report["node_feature_table"]).read_text(encoding="utf-8"))
        ds = ShardDataset(shard, node_feature_table=table)
        batch = collate([ds[i] for i in range(len(ds))])

        vocab = max(int(batch["input_ids"].max()) + 1, 8)
        cfg = TomatoPGFMConfig(vocab_size=vocab, d_model=32, n_layers=3,
                               n_heads=4, graph_feature_dim=FEATURE_DIM,
                               max_seq_len=int(batch["input_ids"].shape[1]) + 2,
                               attn_window=8, use_real_mamba=False)
        model = TomatoPGFM(cfg)
        out = model(batch["input_ids"], batch["graph_features"], "on", edge_index=batch["edge_index"])

    graph_feat_nonzero = float(batch["graph_features"].abs().sum()) > 0
    edges_built = int(batch["edge_index"].shape[1]) > 0
    variant_classes = sorted({int(v) for v in batch["variant_class"].tolist()})
    ok = report["examples_built"] > 0 and graph_feat_nonzero and edges_built and out["mlm_logits"].isfinite().all().item()
    return {
        "examples_built": report["examples_built"],
        "non_reference_fraction": report["non_reference_fraction"],
        "graph_features_nonzero": graph_feat_nonzero,
        "edge_index_columns": int(batch["edge_index"].shape[1]),
        "variant_classes_present": variant_classes,
        "forward_finite": bool(out["mlm_logits"].isfinite().all().item()),
        "status": "pass" if ok else "fail",
    }
