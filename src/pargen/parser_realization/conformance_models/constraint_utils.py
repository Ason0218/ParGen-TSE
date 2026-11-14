from __future__ import annotations

"""Helpers for mapping format-tree constraints to stable IDs."""

from typing import Any, Dict, Optional, List


def constraint_id_for_node(node_name: str, raw_constraint: str) -> str:
    """Return a stable constraint ID for a node-level constraint string."""
    return f"node:{node_name}:{raw_constraint}".strip()


def constraint_id_for_edge(rel: str, src_name: str, dst_name: str, raw: Optional[str] = None) -> str:
    """Return a stable constraint ID for an edge relationship."""
    rel_norm = rel.replace(" ", "_").lower()
    suffix = f":{raw}" if raw else ""
    return f"edge:{rel_norm}:{src_name}->{dst_name}{suffix}"


def is_checksum_field(name: str) -> bool:
    lname = name.lower()
    return "checksum" in lname or "crc" in lname or lname.endswith("cs")


def summarize_constraints(format_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize node/edge constraints from the format tree into a prompt-friendly JSON.
    Captures:
    - Fields: name, bit_length, endianness, range/enum constraints.
    - Edges: length_of / condition_on relations (selector/length src, target, formula).
    """
    tree = format_tree.get("protocol_tree") or format_tree
    nodes = tree.get("nodes", []) or []
    edges = tree.get("edges", []) or []

    fields: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name") or node.get("node_id") or "")
        bit_length = node.get("bit_length") or node.get("size_bits")
        try:
            bit_length = int(bit_length)
        except Exception:
            bit_length = None
        endianness = (
            node.get("endianness")
            or node.get("byte_order")
            or (node.get("encoding") or {}).get("endianness")
            or "big"
        )
        constraints = []
        for c in node.get("constraints", []) or []:
            if isinstance(c, str) and (c.lower().startswith("range") or c.lower().startswith("enum")):
                constraints.append(c)
        fields.append(
            {
                "name": name,
                "bit_length": bit_length,
                "endianness": endianness,
                "constraints": constraints,
            }
        )

    rels: List[Dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        rel = str(edge.get("rel") or "").lower()
        if rel not in ("length_of", "condition_on"):
            continue
        rels.append(
            {
                "rel": rel,
                "src": edge.get("src"),
                "dst": edge.get("dst"),
                "src_name": edge.get("src_name") or edge.get("src"),
                "dst_name": edge.get("dst_name") or edge.get("dst"),
                "formula": edge.get("formula"),
            }
        )

    return {
        "fields": fields,
        "relations": rels,
    }
