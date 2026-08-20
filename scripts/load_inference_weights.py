"""Verify or strictly load the official TomatoPGFM inference weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from tomatopgfm.config import load_model_config
from tomatopgfm.model import TomatoPGFM


EXPECTED_BYTES = 1_916_843_008
EXPECTED_SHA256 = "462f3bfc7178c4558fe993ef579d925da6a54a7448c975b8488bf74d0ac36f7c"
EXPECTED_TENSORS = 604
EXPECTED_UNIQUE_PARAMETERS = 479_195_678
CANONICAL_TIED_KEY = "embed.weight"
OMITTED_TIED_ALIAS = "mlm_head.weight"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path: Path) -> dict:
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != EXPECTED_BYTES or actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"Inference-weight identity mismatch: bytes={actual_bytes}, sha256={actual_sha256}"
        )

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        metadata = handle.metadata()
        value_count = sum(
            int(torch.tensor(handle.get_slice(name).get_shape()).prod().item())
            for name in keys
        )
    if len(keys) != EXPECTED_TENSORS or value_count != EXPECTED_UNIQUE_PARAMETERS:
        raise RuntimeError(
            f"Unexpected safetensors contents: tensors={len(keys)}, values={value_count}"
        )
    if CANONICAL_TIED_KEY not in keys or OMITTED_TIED_ALIAS in keys:
        raise RuntimeError("Unexpected tied-weight representation in safetensors")
    if metadata.get("tied_weight") != f"{OMITTED_TIED_ALIAS}->{CANONICAL_TIED_KEY}":
        raise RuntimeError("Missing or incorrect tied-weight metadata")
    return {
        "status": "verified",
        "filename": path.name,
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "tensor_count": len(keys),
        "unique_parameters": value_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        "--checkpoint",
        dest="weights",
        type=Path,
        default=Path("model.safetensors"),
    )
    parser.add_argument("--config", type=Path, default=Path("model_final.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify identity and safetensors metadata without instantiating the model.",
    )
    args = parser.parse_args()

    report = verify_weights(args.weights)
    if args.verify_only:
        print(json.dumps(report, indent=2))
        return

    cfg = load_model_config(args.config)
    model = TomatoPGFM(cfg)
    state = load_file(args.weights, device="cpu")
    state[OMITTED_TIED_ALIAS] = state[CANONICAL_TIED_KEY]
    model.load_state_dict(state, strict=True)
    if model.embed.weight.data_ptr() != model.mlm_head.weight.data_ptr():
        raise RuntimeError("Embedding/output-head weight tying was not preserved")
    model.to(args.device).eval()
    report.update(
        {
            "status": "loaded",
            "model": "TomatoPGFM",
            "version": "v0.1.0",
            "device": args.device,
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
