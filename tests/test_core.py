from pathlib import Path

import pytest

from tomatopgfm.data import accession_from_name
from tomatopgfm.features import audit_node_features, build_node_features
from tomatopgfm.graph import Bubble, Edge, Segment, is_window_like
from tomatopgfm.path_sampling import audit_path_samples, samples_from_bubbles
from tomatopgfm.shards import VTP_CLASSES, validate_vtp_labels
from tomatopgfm.tokenizer import RCKmerTokenizer, rc
from tomatopgfm.training import TaskHealth

try:
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def test_accession_parsing():
    assert accession_from_name("LA1974_genome.fa.gz") == "LA1974"
    assert accession_from_name("S.pimpinellifolium.genomic.fa") == "S.pimpinellifolium"
    assert accession_from_name("TS623.gff3.gz") == "TS623"


def test_rc_tokenizer_symmetry():
    tok = RCKmerTokenizer(k=3)
    tok.train(["ACGTACGTACGT"], vocab_size=64)
    assert tok.audit_rc_symmetry()["level"] == "vocab_closure_legacy_not_production_gate"
    assert tok.audit_kmer_rc_consistency()["status"] == "pass"
    assert rc("ACGN") == "NCGT"


def test_tokenizer_encode_rc_consistency():
    tok = RCKmerTokenizer(k=3)
    tok.train(["ACGTACGTACGTTTTGGGCCCAAA"], vocab_size=128)
    report = tok.audit_encode_rc_consistency(["ACGTACGTACGT", "TTTGGGCCCAAA"])
    assert report["level"] == "encode_soft_consistency_frame_sensitive"
    assert report["consistent_fraction"] == 1.0


def test_vtp_range():
    assert validate_vtp_labels(range(len(VTP_CLASSES)))["status"] == "pass"
    assert validate_vtp_labels([len(VTP_CLASSES)])["status"] == "fail"


def test_path_sampling_synthetic_is_flagged():
    bubbles = [
        Bubble("b1", "s", "t", [["s", "a", "t"], ["s", "b", "t"]]),
        Bubble("b2", "u", "v", [["u", "c", "v"], ["u", "d", "v"]]),
    ]
    samples = samples_from_bubbles(bubbles)
    report = audit_path_samples(samples)
    assert report["bad_cpc_samples"] == []
    assert report["synthetic_accession"] is True
    assert report["status"] == "warn"  # placeholder accessions cannot certify non-ref fraction


def test_path_sampling_real_accessions():
    bubbles = [Bubble("b1", "s", "t", [["s", "a", "t"], ["s", "b", "t"]])]
    node_acc = {"s": {"SL6.0", "ACC1"}, "t": {"SL6.0", "ACC1"}, "a": {"SL6.0"}, "b": {"ACC1"}}
    samples = samples_from_bubbles(bubbles, node_accessions=node_acc)
    roles = {s.role for s in samples}
    assert roles == {"reference", "alternate"}
    ref = next(s for s in samples if s.role == "reference")
    assert ref.accession_id == "SL6.0"
    report = audit_path_samples(samples)
    assert report["synthetic_accession"] is False


def test_is_window_like_handles_segment_ids():
    window_edges = [Edge(f"s{i}", f"s{i+1}") for i in range(200)]
    assert is_window_like(window_edges) is True
    far_edges = [Edge("s1", "s9000"), Edge("s2", "s8000")]
    assert is_window_like(far_edges) is False


def test_collapse_detector_loss_near_zero():
    h = TaskHealth("mlm")
    for _ in range(3):
        h.update(1e-6)
    assert h.collapsed is True


def test_collapse_detector_metric_stalled():
    h = TaskHealth("vtp")
    for _ in range(3):
        h.update(2.0, metric=0.1)  # loss flat, metric flat -> silent collapse
    assert h.collapsed is True


def test_collapse_detector_healthy():
    h = TaskHealth("cpc")
    for i, loss in enumerate([3.0, 2.0, 1.0]):
        h.update(loss, metric=0.1 + 0.1 * i)
    assert h.collapsed is False


def test_node_features_not_random_only():
    segs = {
        "1": Segment("1", "ACGTACGT", 8, {"SN": "SL6.0"}),
        "2": Segment("2", "GGGGCCCC", 8, {"SN": "ACC1", "note": "repeat"}),
    }
    feats = build_node_features(segs, [Edge("1", "2")])
    assert audit_node_features(feats)["status"] == "pass"


@pytest.mark.skipif(not HAS_TORCH, reason="torch only available on the GPU host")
def test_core_smoke():
    from tomatopgfm.audit import core_smoke_audit

    assert core_smoke_audit()["status"] == "pass"


@pytest.mark.skipif(not HAS_TORCH, reason="torch only available on the GPU host")
def test_cpc_loss_is_finite_with_self_mask():
    import torch

    from tomatopgfm.training import cpc_loss

    proj = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    pos = torch.tensor([1, 0, 3, 2])
    loss = cpc_loss(proj, pos)
    assert torch.isfinite(loss)


def test_node_accession_map_preserves_dotted_accession(tmp_path):
    from tomatopgfm.graph import node_accession_map

    gfa = (
        "S\ts1\tACGT\tSN:Z:SL6.0\n"
        "P\tSL6.0#1#chr1\ts1+\t*\n"
        "P\tS.pimpinellifolium#1#chr1\ts1+\t*\n"
    )
    p = tmp_path / "g.gfa"
    p.write_text(gfa, encoding="utf-8")
    acc = node_accession_map(p)
    # PanSN accession before '#' must be kept intact, dots and all.
    assert acc["s1"] == {"SL6.0", "S.pimpinellifolium"}


def test_classify_variant_from_sequence():
    from tomatopgfm.shard_builder import classify_variant

    assert classify_variant("ACGT", "ACGT", 1, True) == "none"
    assert classify_variant("ACGT", "ACGA", 1, False) == "snp"
    assert classify_variant("ACGTACGT", "ACGT", 1, False) == "small_indel"
    assert classify_variant("A" * 200, "ACGT", 1, False) == "sv_ins"
    assert classify_variant("ACGT", "A" * 200, 1, False) == "sv_del"
    assert classify_variant("AC", "ACGTGGGG", 3, False) == "sv_complex"


@pytest.mark.skipif(not HAS_TORCH, reason="torch only available on the GPU host")
def test_shard_builder_end_to_end():
    from tomatopgfm.shard_builder import build_shards_smoke

    r = build_shards_smoke()
    assert r["status"] == "pass"
    assert r["examples_built"] == 2
    assert r["graph_features_nonzero"] is True
    assert r["edge_index_columns"] > 0
    assert r["non_reference_fraction"] == 0.5


@pytest.mark.skipif(not HAS_TORCH, reason="torch only available on the GPU host")
def test_dry_run_short():
    from tomatopgfm.trainer import dry_run

    result = dry_run(steps=20, batch_size=4, seq_len=32)
    assert result["status"] == "pass"
    assert result["steps"] == 20
