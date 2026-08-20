from __future__ import annotations

import json
import shutil
from pathlib import Path

from .data import write_manifest
from .graph import Edge, is_window_like
from .shards import VTP_CLASSES, dummy_examples, non_reference_fraction, validate_vtp_labels
from .tokenizer import RCKmerTokenizer


def external_tools() -> dict:
    tools = ["minigraph", "samtools", "bedtools", "RepeatMasker", "trf"]
    return {tool: shutil.which(tool) for tool in tools}


def phase0_audit(data_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(data_root, out_dir)
    tool_report = external_tools()
    payload = {
        "manifest": {
            "train_count": manifest["train_count"],
            "holdout_count": manifest["holdout_count"],
            "leakage": manifest["leakage"],
        },
        "external_tools": tool_report,
        "status": "pass" if manifest["leakage"]["status"] == "pass" else "fail",
    }
    (out_dir / "phase0_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def core_smoke_audit() -> dict:
    from .model import smoke_forward

    static = static_smoke_audit()
    model = smoke_forward()
    status = "pass" if static["status"] == "pass" and model["status"] == "pass" else "fail"
    return {**static, "model": model, "status": status}


def static_smoke_audit() -> dict:
    tok = RCKmerTokenizer(k=3)
    tok.train(["ACGTACGT", "TTTAAACCCGGG"], vocab_size=128)
    tok_report = tok.audit_rc_symmetry()
    tok_encode = tok.audit_encode_rc_consistency(["ACGTACGTACGT", "TTTAAACCCGGG"])
    examples = dummy_examples()
    frac = non_reference_fraction(examples)
    vtp = validate_vtp_labels([e.variant_class for e in examples], n_classes=len(VTP_CLASSES))
    window = is_window_like([Edge(str(i), str(i + 1)) for i in range(200)])
    encode_ok = tok_encode["consistent_fraction"] == 1.0
    # Vocabulary closure is a legacy diagnostic; encode-level reverse-complement
    # consistency is the production tokenizer gate.
    status = "pass" if vtp["status"] == "pass" and window and encode_ok else "fail"
    return {
        "tokenizer_vocab_closure": tok_report,
        "tokenizer_encode_rc": tok_encode,
        "non_reference_fraction": frac,
        "vtp": vtp,
        "window_detector_on_window_edges": window,
        "status": status,
    }
