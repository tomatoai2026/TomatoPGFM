from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


FASTA_RE = re.compile(r"(\.fa|\.fasta)(\.gz)?$|_genome\.fa\.gz$|genomic\.fa$", re.I)
GFF_RE = re.compile(r"\.gff3?(\.gz)?$", re.I)


@dataclass(frozen=True)
class GenomeRecord:
    accession: str
    fasta: str
    gff: str
    split: str
    size_fasta: int
    size_gff: int
    mtime_fasta: float
    mtime_gff: float
    sha256_fasta_head: str
    sha256_gff_head: str


def accession_from_name(name: str) -> str:
    base = name
    for suffix in [".fasta.gz", ".fa.gz", ".gff3.gz", ".gff.gz", ".fasta", ".fa", ".gff3", ".gff"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for suffix in ["_genome", ".genomic"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def sha256_head(path: Path, n_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(n_bytes))
    return h.hexdigest()


def scan_genomes(data_root: Path, holdout_names: Iterable[str] = ("LA1974",)) -> tuple[list[GenomeRecord], list[GenomeRecord]]:
    holdouts = set(holdout_names)
    train_files = [p for p in data_root.iterdir() if p.is_file()]
    holdout_dir = data_root / "holdout"
    holdout_files = [p for p in holdout_dir.iterdir() if p.is_file()] if holdout_dir.exists() else []

    def pair(files: list[Path], split: str) -> list[GenomeRecord]:
        fasta = {accession_from_name(p.name): p for p in files if FASTA_RE.search(p.name)}
        gff = {accession_from_name(p.name): p for p in files if GFF_RE.search(p.name)}
        records: list[GenomeRecord] = []
        missing_gff = sorted(set(fasta) - set(gff))
        missing_fasta = sorted(set(gff) - set(fasta))
        if missing_gff or missing_fasta:
            raise ValueError(f"Unpaired genome files in {split}: missing_gff={missing_gff}, missing_fasta={missing_fasta}")
        for acc in sorted(fasta):
            f = fasta[acc]
            g = gff[acc]
            fs = f.stat()
            gs = g.stat()
            records.append(
                GenomeRecord(
                    accession=acc,
                    fasta=str(f),
                    gff=str(g),
                    split=split,
                    size_fasta=fs.st_size,
                    size_gff=gs.st_size,
                    mtime_fasta=fs.st_mtime,
                    mtime_gff=gs.st_mtime,
                    sha256_fasta_head=sha256_head(f),
                    sha256_gff_head=sha256_head(g),
                )
            )
        return records

    train = pair(train_files, "train")
    holdout = pair(holdout_files, "holdout")
    leaked = [r.accession for r in train if r.accession in holdouts]
    if leaked:
        raise ValueError(f"Holdout accessions present in training root: {leaked}")
    missing_holdout = sorted(holdouts - {r.accession for r in holdout})
    if missing_holdout:
        raise ValueError(f"Expected holdout accessions missing from holdout/: {missing_holdout}")
    return train, holdout


def write_manifest(data_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, holdout = scan_genomes(data_root)
    payload = {
        "schema": "tomatopgfm_manifest_v1",
        "data_root": str(data_root),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "train_manifest": [asdict(r) for r in train],
        "holdout_manifest": [asdict(r) for r in holdout],
        "leakage": {
            "holdout_in_train": sorted(set(r.accession for r in train) & set(r.accession for r in holdout)),
            "status": "pass",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "train_accessions.txt").write_text("\n".join(r.accession for r in train) + "\n", encoding="utf-8")
    (out_dir / "holdout_accessions.txt").write_text("\n".join(r.accession for r in holdout) + "\n", encoding="utf-8")
    return payload


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    name = None
    parts: list[str] = []
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts).upper()
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if name is not None:
            yield name, "".join(parts).upper()


def pansn_name(accession: str, contig: str, hap: str = "1") -> str:
    return f"{accession}#{hap}#{contig}"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = write_manifest(Path(args.data_root), Path(args.out))
    print(json.dumps({"train_count": payload["train_count"], "holdout_count": payload["holdout_count"], "status": "pass"}))


if __name__ == "__main__":
    main()

