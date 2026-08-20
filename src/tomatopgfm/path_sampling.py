from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .graph import Bubble


@dataclass(frozen=True)
class PathSample:
    sample_id: str
    accession_id: str
    path_id: str
    bubble_id: str
    role: str
    node_ids: list[str]
    positive_path_id: str
    negative_path_ids: list[str]
    variant_class: str
    accessions: list[str]
    synthetic_accession: bool


def _branch_accessions(branch: list[str], source: str, sink: str, node_accessions: dict[str, set[str]]) -> set[str]:
    """Accessions traversing a branch = intersection over its interior nodes.

    A genome path uses a branch only if it visits every interior node, so the
    set of accessions on the branch is the intersection of per-node accession
    membership (source/sink are shared by all branches and excluded).
    """
    interior = [n for n in branch if n not in (source, sink)]
    if not interior:
        interior = branch
    acc: set[str] | None = None
    for n in interior:
        members = node_accessions.get(n, set())
        acc = set(members) if acc is None else (acc & members)
    return acc or set()


def samples_from_bubbles(
    bubbles: list[Bubble],
    reference_accession: str = "SL6.0",
    node_accessions: dict[str, set[str]] | None = None,
) -> list[PathSample]:
    """Build CPC/path samples.

    If ``node_accessions`` (node id -> set of accessions, parsed from real GFA
    tags) is supplied, each branch's accession membership and reference/alternate
    role come from the graph itself. Without it, accession is a positional
    placeholder and ``synthetic_accession=True`` is flagged so the audit can
    refuse to certify the non-reference fraction.
    """
    synthetic = node_accessions is None
    samples: list[PathSample] = []
    all_path_ids: list[str] = []
    staged: list[tuple[Bubble, int, list[str], str]] = []
    for b in bubbles:
        for i, branch in enumerate(b.branches):
            path_id = f"{b.bubble_id}_path_{i}"
            all_path_ids.append(path_id)
            staged.append((b, i, branch, path_id))
    for b, i, branch, path_id in staged:
        sibling_ids = [f"{b.bubble_id}_path_{j}" for j in range(len(b.branches)) if j != i]
        positives = sibling_ids or [path_id]
        negatives = [pid for pid in all_path_ids if not pid.startswith(b.bubble_id)]
        if synthetic:
            accs = {reference_accession} if i == 0 else {f"ALT_{i}"}
        else:
            accs = _branch_accessions(branch, b.source, b.sink, node_accessions)
        is_ref = reference_accession in accs
        if is_ref:
            accession_id = reference_accession
        else:
            accession_id = sorted(accs)[0] if accs else f"ALT_{i}"
        samples.append(
            PathSample(
                sample_id=f"{b.bubble_id}_{i}",
                accession_id=accession_id,
                path_id=path_id,
                bubble_id=b.bubble_id,
                role="reference" if is_ref else "alternate",
                node_ids=branch,
                positive_path_id=positives[0],
                negative_path_ids=negatives[:8],
                variant_class="sv_complex" if len(branch) > 3 else "small_indel",
                accessions=sorted(accs),
                synthetic_accession=synthetic,
            )
        )
    return samples


def audit_path_samples(samples: list[PathSample], reference_accession: str = "SL6.0") -> dict:
    if not samples:
        return {"samples": 0, "status": "fail", "reason": "no path samples"}
    synthetic = any(s.synthetic_accession for s in samples)
    non_ref = sum(1 for s in samples if s.role != "reference") / len(samples)
    bad_cpc = [s.sample_id for s in samples if s.positive_path_id == s.path_id or not s.negative_path_ids]
    fraction_ok = 0.40 <= non_ref <= 0.60
    if synthetic:
        # Positional placeholder accessions cannot certify a real non-reference
        # token mix; the fraction is structurally ~0.5 and meaningless.
        status = "warn"
    elif fraction_ok and not bad_cpc:
        status = "pass"
    else:
        status = "warn"
    return {
        "samples": len(samples),
        "non_reference_fraction": non_ref,
        "synthetic_accession": synthetic,
        "bad_cpc_samples": bad_cpc[:20],
        "status": status,
    }


def write_path_samples(samples: list[PathSample], out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(asdict(s)) for s in samples) + "\n", encoding="utf-8")
    report = audit_path_samples(samples)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

