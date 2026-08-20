from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .data import iter_fasta, pansn_name, scan_genomes


def write_pansn_fasta(records, out_path: Path) -> int:
    """Write a PanSN-renamed FASTA (accession#hap#contig) for graph building.

    Only the records passed in are written, so holdout exclusion is enforced by
    the caller selecting train-only records.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_contigs = 0
    with out_path.open("wt", encoding="utf-8") as out:
        for rec in records:
            for contig, seq in iter_fasta(Path(rec.fasta)):
                out.write(f">{pansn_name(rec.accession, contig)}\n")
                for i in range(0, len(seq), 60):
                    out.write(seq[i : i + 60] + "\n")
                n_contigs += 1
    return n_contigs


def build_train_pansn(data_root: Path, out_path: Path, accessions: list[str] | None = None) -> dict:
    """Build a train-only PanSN FASTA. Holdout records are never included."""
    train, _holdout = scan_genomes(data_root)
    if accessions is not None:
        wanted = set(accessions)
        train = [r for r in train if r.accession in wanted]
    n_contigs = write_pansn_fasta(train, out_path)
    return {"accessions": [r.accession for r in train], "contigs": n_contigs, "fasta": str(out_path)}


def run_minigraph(
    fasta: Path,
    out_gfa: Path,
    reference: Path | None = None,
    preset: str = "ggs",
    threads: int = 16,
    extra_args: list[str] | None = None,
) -> dict:
    """Run minigraph to build a graph GFA from a PanSN FASTA.

    Requires minigraph on PATH (Linux A100 dependency). Raises if absent so the
    pipeline fails loudly rather than silently skipping graph construction.
    """
    exe = shutil.which("minigraph")
    if exe is None:
        raise RuntimeError("minigraph not found on PATH; install it on the A100 host before graph building")
    out_gfa.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-c", "-x", preset, "-t", str(threads)]
    if extra_args:
        cmd += extra_args
    if reference is not None:
        cmd.append(str(reference))
    cmd.append(str(fasta))
    with out_gfa.open("wt", encoding="utf-8") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"minigraph failed (code {proc.returncode}): {proc.stderr[-2000:]}")
    return {"gfa": str(out_gfa), "cmd": " ".join(cmd), "status": "pass"}
