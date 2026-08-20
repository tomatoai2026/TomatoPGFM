from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Segment:
    sid: str
    seq: str
    length: int
    tags: dict[str, str]


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str


@dataclass(frozen=True)
class Bubble:
    bubble_id: str
    source: str
    sink: str
    branches: list[list[str]]


def parse_gfa(path: Path, max_segments: int | None = None) -> tuple[dict[str, Segment], list[Edge]]:
    segments: dict[str, Segment] = {}
    edges: list[Edge] = []
    with path.open("rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            rec = fields[0]
            if rec == "S":
                sid, seq = fields[1], fields[2]
                tags = {}
                for tag in fields[3:]:
                    parts = tag.split(":", 2)
                    if len(parts) == 3:
                        tags[parts[0]] = parts[2]
                segments[sid] = Segment(sid=sid, seq=seq, length=len(seq) if seq != "*" else 0, tags=tags)
                if max_segments and len(segments) >= max_segments:
                    break
            elif rec == "L":
                edges.append(Edge(src=fields[1], dst=fields[3]))
    return segments, edges


def adjacency(edges: Iterable[Edge]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e.src].add(e.dst)
        adj.setdefault(e.dst, set())
    return adj


_SEG_TOKEN_RE = re.compile(r"[<>]?([^<>,+\-]+)")


def node_accession_map(path: Path) -> dict[str, set[str]]:
    """Build node id -> set of accessions from GFA P-lines and W-lines.

    Real per-node accession membership is what makes the non-reference path
    fraction meaningful (vs. positional placeholders). P-lines name a path
    whose accession is taken as the part before '#' (PanSN) or '.'; W-lines
    carry the sample name in field 1 directly.
    """
    node_acc: dict[str, set[str]] = defaultdict(set)
    with path.open("rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            rec = fields[0]
            if rec == "P" and len(fields) >= 3:
                # PanSN path names are 'accession#hap#contig' -> accession is the
                # part before the first '#'. Only fall back to a '.'-split for
                # non-PanSN names; splitting on '.' unconditionally would corrupt
                # legitimately dotted accessions (e.g. 'SL6.0', 'S.pimpinellifolium').
                raw = fields[1]
                acc = raw.split("#")[0] if "#" in raw else raw.split(".")[0]
                for m in _SEG_TOKEN_RE.finditer(fields[2]):
                    node_acc[m.group(1)].add(acc)
            elif rec == "W" and len(fields) >= 7:
                acc = fields[1]
                for m in _SEG_TOKEN_RE.finditer(fields[6]):
                    node_acc[m.group(1)].add(acc)
    return dict(node_acc)


def condensation_dag(nodes: Iterable[str], edges: Iterable[Edge]) -> tuple[dict[str, int], list[Edge]]:
    """Iterative Tarjan SCC + condensation.

    Uses an explicit work stack instead of recursion so that real pangenome
    graphs (millions of segments, long chains) do not overflow Python's call
    stack the way the recursive form would.
    """
    node_list = list(nodes)
    adj = adjacency(edges)
    index = 0
    sccstack: list[str] = []
    on_stack: set[str] = set()
    idx: dict[str, int] = {}
    low: dict[str, int] = {}
    comp: dict[str, int] = {}
    comps: list[list[str]] = []

    for root in node_list:
        if root in idx:
            continue
        # work stack holds (vertex, iterator over its successors)
        work: list[tuple[str, "deque[str]"]] = [(root, deque(sorted(adj.get(root, ()))))]
        idx[root] = low[root] = index
        index += 1
        sccstack.append(root)
        on_stack.add(root)
        while work:
            v, succ = work[-1]
            advanced = False
            while succ:
                w = succ.popleft()
                if w not in idx:
                    idx[w] = low[w] = index
                    index += 1
                    sccstack.append(w)
                    on_stack.add(w)
                    work.append((w, deque(sorted(adj.get(w, ())))))
                    advanced = True
                    break
                elif w in on_stack:
                    low[v] = min(low[v], idx[w])
            if advanced:
                continue
            # all successors of v processed
            if low[v] == idx[v]:
                members = []
                while True:
                    w = sccstack.pop()
                    on_stack.discard(w)
                    comp[w] = len(comps)
                    members.append(w)
                    if w == v:
                        break
                comps.append(members)
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])

    dag_edges = sorted({(comp[e.src], comp[e.dst]) for e in edges if comp[e.src] != comp[e.dst]})
    return comp, [Edge(str(a), str(b)) for a, b in dag_edges]


def find_simple_bubbles(edges: Iterable[Edge], max_depth: int = 6, limit: int = 100000) -> list[Bubble]:
    adj = adjacency(edges)
    rev: dict[str, set[str]] = defaultdict(set)
    for s, outs in adj.items():
        for d in outs:
            rev[d].add(s)
    bubbles: list[Bubble] = []
    for source, outs in adj.items():
        if len(outs) < 2:
            continue
        branch_paths: list[list[str]] = []
        sink_counts: dict[str, int] = defaultdict(int)
        for start in sorted(outs):
            q = deque([(start, [source, start])])
            seen = {start}
            found = None
            while q:
                node, path = q.popleft()
                if len(path) > max_depth:
                    continue
                if len(rev[node]) > 1 and node != start:
                    found = path
                    break
                for nxt in sorted(adj.get(node, ())):
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, path + [nxt]))
            if found:
                branch_paths.append(found)
                sink_counts[found[-1]] += 1
        sinks = [s for s, c in sink_counts.items() if c >= 2]
        if sinks:
            sink = sorted(sinks)[0]
            branches = [p for p in branch_paths if p[-1] == sink]
            bubbles.append(Bubble(bubble_id=f"bub_{len(bubbles):08d}", source=source, sink=sink, branches=branches))
            if len(bubbles) >= limit:
                break
    return bubbles


def export_graph_report(gfa: Path, out_dir: Path, max_segments: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    segments, edges = parse_gfa(gfa, max_segments=max_segments)
    comp, dag_edges = condensation_dag(segments.keys(), edges)
    bubbles = find_simple_bubbles(dag_edges)
    payload = {
        "gfa": str(gfa),
        "segments": len(segments),
        "raw_edges": len(edges),
        "dag_nodes": len(set(comp.values())),
        "dag_edges": len(dag_edges),
        "bubbles": len(bubbles),
        "status": "pass" if dag_edges else "fail",
    }
    (out_dir / "graph_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "dag_edges.tsv").write_text("\n".join(f"{e.src}\t{e.dst}" for e in dag_edges) + "\n", encoding="utf-8")
    (out_dir / "bubbles.jsonl").write_text("\n".join(json.dumps(asdict(b)) for b in bubbles) + "\n", encoding="utf-8")
    return payload


_INT_RE = re.compile(r"\d+")


def _node_index(node_id: str) -> int | None:
    """Extract the first integer run from a segment id (e.g. 's123' -> 123).

    minigraph/gfatools name segments like ``s1``/``s2``; a naive ``int()`` would
    raise and silently disable the window-adjacency guard, so the audit must be
    able to read the numeric ordinal out of typical id schemes.
    """
    m = _INT_RE.search(node_id)
    return int(m.group()) if m else None


def is_window_like(edges: list[Edge], tolerance: float = 0.95, window: int = 128) -> bool:
    if not edges:
        return False
    checked = 0
    near = 0
    for e in edges:
        a, b = _node_index(e.src), _node_index(e.dst)
        if a is None or b is None:
            continue
        checked += 1
        if abs(a - b) <= window:
            near += 1
    return checked > 0 and near / checked >= tolerance

