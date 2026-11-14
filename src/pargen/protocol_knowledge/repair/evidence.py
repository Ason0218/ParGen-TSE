"""RetrieveEvidence: specification fragments linked to the objects implicated by a check report.

Each PKG node records the sections that support it in ``source_section_ids`` (indices into the section list).
For a report, the implicated objects are the nodes and edge endpoints named by its issues, plus their ancestors
and the endpoints of dependency edges that touch them.  The evidence is the union of their linked sections.
When no link resolves, all sections are returned so that repair is never left without evidence.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Sequence, Set

_INT_RE = re.compile(r"-?\d+")


def _tree_nodes_edges(tree: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload = tree.get("protocol_tree", tree) if isinstance(tree, dict) else {}
    nodes = [n for n in payload.get("nodes", []) or [] if isinstance(n, dict)]
    edges = [e for e in payload.get("edges", []) or [] if isinstance(e, dict)]
    return nodes, edges


def _ints(value: Any) -> List[int]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: List[int] = []
        for item in value:
            out.extend(_ints(item))
        return out
    return [int(m) for m in _INT_RE.findall(str(value))]


def implicated_node_ids(report: Any) -> Set[int]:
    issues = getattr(report, "issues", None)
    if isinstance(report, dict):
        issues = report.get("issues", issues)
    values: Iterable[Any] = issues.values() if isinstance(issues, dict) else (issues or [])
    out: Set[int] = set()
    for issue in values:
        target = getattr(issue, "target", None)
        if target is None and isinstance(issue, dict):
            target = issue.get("target")
        identifier = getattr(target, "identifier", None)
        if identifier is None and isinstance(target, dict):
            identifier = target.get("identifier") or target.get("id")
        out.update(_ints(identifier))
        node_id = getattr(issue, "node_id", None)
        if node_id is None and isinstance(issue, dict):
            node_id = issue.get("node_id")
        out.update(_ints(node_id))
    return out


def retrieve_evidence_sections(tree: Any, report: Any, sections: Sequence[Any]) -> List[Any]:
    sections = list(sections or [])
    if not sections or str(os.getenv("STEP2_RETRIEVE_EVIDENCE", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return sections
    nodes, edges = _tree_nodes_edges(tree)
    by_id = {}
    for node in nodes:
        ids = _ints(node.get("node_id"))
        if ids:
            by_id[ids[0]] = node
    implicated = {nid for nid in implicated_node_ids(report) if nid in by_id}
    if not implicated:
        return sections
    scope: Set[int] = set(implicated)
    for edge in edges:
        src, dst = _ints(edge.get("src")), _ints(edge.get("dst"))
        if (src and src[0] in implicated) or (dst and dst[0] in implicated):
            scope.update(x for x in src[:1] + dst[:1] if x in by_id)
    for nid in list(scope):
        parent = _ints(by_id[nid].get("parent_id"))
        seen = set()
        while parent and parent[0] in by_id and parent[0] not in seen:
            seen.add(parent[0]); scope.add(parent[0])
            parent = _ints(by_id[parent[0]].get("parent_id"))
    indices: Set[int] = set()
    for nid in scope:
        node = by_id[nid]
        for key in ("source_section_ids", "source_sections", "source_section_id", "section_ids", "section_index"):
            indices.update(i for i in _ints(node.get(key)) if 0 <= i < len(sections))
    if not indices:
        return sections
    return [sections[i] for i in sorted(indices)]
