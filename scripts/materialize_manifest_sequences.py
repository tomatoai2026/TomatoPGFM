#!/usr/bin/env python
"""Reconstruct 0-based half-open evaluation windows from a reference FASTA."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def open_text(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, list[str]] = {}
    name = None
    with open_text(path, "rt") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in sequences:
                    raise ValueError(f"Duplicate FASTA identifier: {name}")
                sequences[name] = []
            elif name is None:
                raise ValueError("FASTA sequence encountered before a header")
            else:
                sequences[name].append(line.upper())
    return {key: "".join(value) for key, value in sequences.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    genome = read_fasta(args.fasta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open_text(args.manifest, "rt") as source, open_text(args.out, "wt") as target:
        for line in source:
            row = json.loads(line)
            chrom = row["chrom"]
            start, end = int(row["start"]), int(row["end"])
            if chrom not in genome or not (0 <= start < end <= len(genome[chrom])):
                raise ValueError(f"Invalid interval: {chrom}:{start}-{end}")
            seq = genome[chrom][start:end]
            observed = hashlib.sha256(seq.encode("ascii")).hexdigest()
            expected = row.get("sequence_sha256")
            if expected and observed != expected:
                raise ValueError(f"Sequence checksum mismatch at row {rows}")
            row["seq"] = seq
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows += 1
    print(json.dumps({"rows": rows, "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
