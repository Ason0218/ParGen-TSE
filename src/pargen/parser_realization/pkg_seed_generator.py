from __future__ import annotations

"""PKG-path-derived positive seed generation for constraint suites."""

import ast
import copy
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from pargen.protocol_knowledge.tree_utils import (
    parse_numeric_calculated_upper_range_constraint_text,
    parse_numeric_enum_constraint_text,
    parse_numeric_range_constraint_text,
)

from .conformance_models.constraint_utils import constraint_id_for_edge, constraint_id_for_node
from .conformance_models.models import FieldBoundary, NegativeSample, PositiveSample
from .conformance_models.negative_sampler import ConstraintGuidedNegativeSampler


@dataclass(frozen=True)
class PkgParsingPath:
    path_id: str
    node_ids: Tuple[str, ...]
    variant_id: Optional[str] = None
    variant_name: Optional[str] = None
    sequence_items: Tuple[str, ...] = ()
    covered_branch_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SkippedPkgPath:
    path_id: str
    reason: str


@dataclass(frozen=True)
class PkgSeedGenerationResult:
    paths_total: int
    positive_samples: Tuple[PositiveSample, ...]
    negative_samples: Tuple[NegativeSample, ...]
    paths: Tuple[PkgParsingPath, ...]
    skipped_paths: Tuple[SkippedPkgPath, ...] = ()
    branch_constraints_total: int = 0
    branch_constraints_covered: int = 0
    positive_only_branch_constraints: Tuple[str, ...] = ()
    negative_obligations_activated: int = 0
    negative_obligations_mutatable: int = 0
    negative_obligations_covered: int = 0
    positive_only_node_constraints: Tuple[str, ...] = ()
    unsupported_negative_constraints: Tuple[str, ...] = ()
    condition_constraints_total: int = 0
    condition_constraints_covered: int = 0
    replay_validated_paths: int = 0
    replay_unverified_paths: Tuple[SkippedPkgPath, ...] = ()
    paths_enumerated_total: int = 0
    paths_truncated: bool = False
    negative_qualifications: Tuple[Dict[str, Any], ...] = ()
    negative_construction_audit: Tuple[Dict[str, Any], ...] = ()


def pkg_constraint_id_for_node(node_id: Any, node_name: str, raw_constraint: str) -> str:
    """Return a node-scoped ID so duplicate field names remain distinct obligations."""

    return f"node:{_node_key(node_id)}:{str(node_name)}:{str(raw_constraint).strip()}"


def pkg_constraint_id_for_edge(
    rel: str,
    src_id: Any,
    src_name: str,
    dst_id: Any,
    dst_name: str,
    raw: Optional[str] = None,
    message_type: Optional[str] = None,
) -> str:
    """Return an edge-scoped ID that maps unambiguously to a codegen contract tag."""

    rel_norm = str(rel or "").replace(" ", "_").lower()
    message_type_norm = str(message_type or "").strip().lower() or "bidirectional"
    direction_suffix = (
        f":message_type={message_type_norm}"
        if rel_norm == "length_of" and message_type_norm != "bidirectional"
        else ""
    )
    suffix = f":{raw}" if raw else ""
    return (
        f"edge:{rel_norm}:{_node_key(src_id)}:{str(src_name)}"
        f"->{_node_key(dst_id)}:{str(dst_name)}{direction_suffix}{suffix}"
    )


def build_pkg_constraints_catalog(format_tree: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a complete, node-scoped catalog for tree-only conformance suites."""

    from .constraint_contract import build_contract, tag_for
    from .parse_ir import build_parse_ir

    tree = _tree_payload(format_tree)
    nodes = [dict(node) for node in tree.get("nodes", []) or [] if isinstance(node, Mapping)]
    edges = [dict(edge) for edge in tree.get("edges", []) or [] if isinstance(edge, Mapping)]
    node_map = {
        _node_key(node.get("node_id")): node
        for node in nodes
        if node.get("node_id") is not None
    }
    catalog: Dict[str, Dict[str, Any]] = {}

    for node_id in sorted(node_map, key=_stable_node_id_key):
        node = node_map[node_id]
        name = _node_name(node, node_id)
        size_bits = _coerce_optional_int(node.get("size_bits"))
        bit_start = _coerce_optional_int(node.get("bit_start"))
        for raw in node.get("constraints") or []:
            if not isinstance(raw, str):
                continue
            cid = pkg_constraint_id_for_node(node_id, name, raw)
            catalog[cid] = {
                "kind": "node_constraint",
                "node_id": _json_node_id(node_id),
                "node_name": name,
                "node_type": _node_type(node),
                "constraint": raw,
                "endianness": _node_endianness(node),
                "size_bits": size_bits,
                "bit_start": bit_start,
                "present_in_tree": True,
                "legacy_constraint_id": constraint_id_for_node(name, raw),
            }
    for edge in edges:
        rel = str(edge.get("rel") or "").strip().lower()
        if rel not in {"length_of", "condition_on"}:
            continue
        src = _node_key(edge.get("src"))
        dst = _node_key(edge.get("dst"))
        src_name = _node_name(node_map.get(src, {}), src)
        dst_name = _node_name(node_map.get(dst, {}), dst)
        message_type = str(edge.get("message_type") or "").strip().lower() or "bidirectional"
        cid = pkg_constraint_id_for_edge(
            rel,
            src,
            src_name,
            dst,
            dst_name,
            edge.get("formula"),
            message_type,
        )
        catalog[cid] = {
            "kind": "edge_constraint",
            "src": _json_node_id(src),
            "dst": _json_node_id(dst),
            "src_name": src_name,
            "dst_name": dst_name,
            "rel": rel,
            "formula": edge.get("formula"),
            "message_type": message_type,
            "endianness": _node_endianness(node_map.get(src, {})),
            "present_in_tree": True,
            "legacy_constraint_id": constraint_id_for_edge(rel, src_name, dst_name, edge.get("formula")),
        }

    contract_by_node: Dict[Tuple[str, str], Any] = {}
    contract_by_edge: Dict[Tuple[str, str, str, str, str], Any] = {}
    try:
        for item in build_contract(build_parse_ir(tree)):
            if item.kind == "node" and item.node_id is not None and item.clause is not None:
                contract_by_node[
                    (_node_key(item.node_id), _normalize_contract_text(item.clause))
                ] = item
            elif item.kind == "edge" and item.src is not None and item.dst is not None:
                contract_by_edge[
                    (
                        str(item.rel or "").strip().lower(),
                        _node_key(item.src),
                        _node_key(item.dst),
                        str(item.message_type or "").strip().lower() or "bidirectional",
                        _normalize_contract_text(item.formula),
                    )
                ] = item
    except Exception:
        # The raw catalog remains useful for diagnostics, but generated targets
        # without a matching executable contract are rejected by the suite builder.
        contract_by_node = {}
        contract_by_edge = {}

    for spec in catalog.values():
        item = None
        if spec.get("kind") == "node_constraint":
            item = contract_by_node.get(
                (_node_key(spec.get("node_id")), _normalize_contract_text(spec.get("constraint")))
            )
        elif spec.get("kind") == "edge_constraint":
            item = contract_by_edge.get(
                (
                    str(spec.get("rel") or "").strip().lower(),
                    _node_key(spec.get("src")),
                    _node_key(spec.get("dst")),
                    str(spec.get("message_type") or "").strip().lower() or "bidirectional",
                    _normalize_contract_text(spec.get("formula")),
                )
            )
        spec["present_in_codegen_contract"] = item is not None
        if item is not None:
            spec["contract_id"] = item.id
            spec["contract_tag"] = tag_for(item.id)
    return catalog


def generate_pkg_path_seed_suite(
    format_tree: Mapping[str, Any],
    *,
    max_negatives_per_positive: int = 2,
    max_paths: Optional[int] = None,
    max_sequence_items: int = 3,
    isolate_negatives: bool = True,
    boundary_positives: bool = True,
) -> PkgSeedGenerationResult:
    """Generate PKG-derived positive seeds and mutate them into negatives.

    This helper is intentionally side-effect free. It materializes synthetic
    packet bytes from feasible PKG paths and returns in-memory samples that can
    be fed into existing exp-constraint machinery.
    """

    tree = _tree_payload(format_tree)
    ctx = _TreeContext(tree)
    executable_condition_ids = {
        cid
        for cid, spec in build_pkg_constraints_catalog(format_tree).items()
        if str(spec.get("rel") or "").strip().lower() == "condition_on"
        and bool(spec.get("present_in_codegen_contract"))
    }
    enumerated_paths = tuple(ctx.enumerate_paths(max_sequence_items=max_sequence_items))
    paths = enumerated_paths
    if max_paths is not None:
        paths = paths[: max(0, int(max_paths))]

    positives = []
    materialized_paths = []
    skipped = []
    for idx, path in enumerate(paths):
        try:
            positives.append(ctx.materialize_positive(path, packet_id=f"pkg_path_{idx}"))
            materialized_paths.append(path)
        except Exception as exc:
            skipped.append(SkippedPkgPath(path_id=path.path_id, reason=f"{type(exc).__name__}: {exc}"))
            continue
        if boundary_positives:
            # The same path at the upper boundary of its value rules; kept only if it replays and differs.
            upper_id = f"pkg_path_{idx}{UPPER_BOUNDARY_SUFFIX}"
            ctx._value_side = "upper"
            try:
                upper = ctx.materialize_positive(path, packet_id=upper_id)
                if upper.raw_bytes != positives[-1].raw_bytes:
                    positives.append(upper)
            except Exception:
                ctx.replay_failures.pop(upper_id, None)
            finally:
                ctx._value_side = "lower"

    negative_candidates = []
    if max_negatives_per_positive > 0 and positives:
        sampler = ConstraintGuidedNegativeSampler(dict(format_tree))
        for sample in positives:
            sampler_sample = replace(
                sample,
                activated_constraints=ctx.legacy_activated_constraints(sample),
            )
            for neg in sampler.generate_for_positive(
                sampler_sample,
                max_per_constraint=max(1, int(max_negatives_per_positive)),
            ):
                canonical = ctx.canonicalize_negative_targets(sample, neg)
                if canonical is None:
                    continue
                if not ctx.negative_keeps_path_reachable(sample, canonical):
                    continue
                if not ctx.negative_targets_supported_constraint(sample, canonical):
                    continue
                proven = ctx.prove_negative_targets(sample, canonical)
                if proven is not None:
                    negative_candidates.append(proven)
            negative_candidates.extend(
                negative
                for negative in ctx.generate_condition_body_negatives(sample)
                if all(
                    str(cid) in executable_condition_ids
                    for cid in negative.expected_violated_constraints
                )
            )
            negative_candidates.extend(ctx.generate_length_truncation_negatives(sample))

    negatives = _deduplicate_and_limit_negatives(
        negative_candidates,
        max_per_constraint=max(1, int(max_negatives_per_positive)),
    )

    from pargen.protocol_knowledge.message_validation.interpreter import DynamicTreeInterpreter
    from .pkg_negative_qualification import qualify_negative
    positive_by_id = {sample.packet_id: sample for sample in positives}
    replays = {
        sample.packet_id: DynamicTreeInterpreter(copy.deepcopy(ctx.tree),
            target_message_type=ctx._target_message_type_for_sample(sample)).parse(sample.raw_bytes)
        for sample in positives
    }
    construction_audit = ()
    if isolate_negatives:
        from .pkg_negative_construction import isolate_selected_negatives
        negatives, qualifications, construction_audit = isolate_selected_negatives(
            ctx, positive_by_id, negatives, replays)
    else:
        qualifications = tuple(qualify_negative(ctx, positive_by_id[negative.base_packet_id], negative,
            replay=replays[negative.base_packet_id]) for negative in negatives)

    covered_condition_constraints = {
        cid
        for sample in positives
        for cid in sample.activated_constraints
        if str(cid).startswith("edge:condition_on:")
    }
    condition_constraints = ctx.branch_constraint_ids()
    branch_variants = ctx.branch_variant_ids()
    covered_branch_variants = {
        branch_id
        for path in materialized_paths
        for branch_id in path.covered_branch_ids
    }
    activated_negative_obligations = {
        str(cid)
        for sample in positives
        for cid in sample.activated_constraints
        if not str(cid).startswith("edge:condition_on:")
    }
    covered_negative_obligations = {
        str(cid)
        for negative in negatives
        for cid in negative.expected_violated_constraints
        if not str(cid).startswith("edge:condition_on:")
    }
    positive_only_nodes = {
        cid
        for cid in activated_negative_obligations
        if all(
            ctx.constraint_is_tautological_for_sample(sample, cid)
            for sample in positives
            if cid in sample.activated_constraints
        )
    }
    mutatable_negative_obligations = activated_negative_obligations - positive_only_nodes
    unsupported_negative_constraints = mutatable_negative_obligations - covered_negative_obligations
    return PkgSeedGenerationResult(
        paths_total=len(paths),
        positive_samples=tuple(positives),
        negative_samples=tuple(negatives),
        paths=paths,
        skipped_paths=tuple(skipped),
        branch_constraints_total=len(branch_variants),
        branch_constraints_covered=len(covered_branch_variants.intersection(branch_variants)),
        positive_only_branch_constraints=ctx.positive_only_branch_constraint_ids(),
        negative_obligations_activated=len(activated_negative_obligations),
        negative_obligations_mutatable=len(mutatable_negative_obligations),
        negative_obligations_covered=len(covered_negative_obligations),
        positive_only_node_constraints=tuple(sorted(positive_only_nodes)),
        unsupported_negative_constraints=tuple(sorted(unsupported_negative_constraints)),
        condition_constraints_total=len(condition_constraints),
        condition_constraints_covered=len(covered_condition_constraints.intersection(condition_constraints)),
        replay_validated_paths=len({pid for pid in ctx.replay_successes if not is_upper_boundary_id(pid)}),
        replay_unverified_paths=tuple(
            SkippedPkgPath(path_id=path_id, reason=reason)
            for path_id, reason in ctx.replay_failures.values()
        ),
        paths_enumerated_total=len(enumerated_paths),
        paths_truncated=len(paths) < len(enumerated_paths),
        negative_qualifications=qualifications,
        negative_construction_audit=construction_audit,
    )


UPPER_BOUNDARY_SUFFIX = "_upper"


def is_upper_boundary_id(packet_id: Any) -> bool:
    """Packet id of the extra positive that materializes a path at the upper boundary of its value rules."""
    return str(packet_id).endswith(UPPER_BOUNDARY_SUFFIX)


def _deduplicate_and_limit_negatives(
    candidates: Sequence[NegativeSample],
    *,
    max_per_constraint: int,
) -> list[NegativeSample]:
    """De-duplicate physical mutations per target, then apply an obligation cap.

    A fail-fast parser can emit only the first violated contract tag.  Keeping
    targets as separate cases prevents one physical packet from claiming that
    several distinct tags were observed at runtime.
    """

    grouped: Dict[Tuple[str, bytes, str], Dict[str, Any]] = {}
    for candidate in candidates:
        for cid in candidate.expected_violated_constraints:
            cid_s = str(cid)
            if not cid_s:
                continue
            key = (str(candidate.base_packet_id), bytes(candidate.raw_bytes), cid_s)
            entry = grouped.setdefault(key, {"notes": []})
            note = str(candidate.note or "").strip()
            if note and note not in entry["notes"]:
                entry["notes"].append(note)

    emitted: Dict[str, int] = {}
    output: list[NegativeSample] = []
    cap = max(1, int(max_per_constraint))
    for (base_packet_id, raw_bytes, cid), entry in grouped.items():
        if emitted.get(cid, 0) >= cap:
            continue
        output.append(
            NegativeSample(
                base_packet_id=base_packet_id,
                raw_bytes=raw_bytes,
                expected_violated_constraints=[cid],
                note="; ".join(entry["notes"]),
            )
        )
        emitted[cid] = emitted.get(cid, 0) + 1
    return output


def _tree_payload(format_tree: Mapping[str, Any]) -> Dict[str, Any]:
    payload = format_tree.get("protocol_tree") if isinstance(format_tree, Mapping) else None
    if isinstance(payload, Mapping):
        return copy.deepcopy(dict(payload))
    return copy.deepcopy(dict(format_tree))


class _TreeContext:
    FIELD_TYPES = {"field", "selector", "payload", "length", "type", "checksum", "count"}
    DESCENDABLE_TYPES = {"protocol", "container", "variant", "header", "payload", "selector"}

    def __init__(self, tree: Mapping[str, Any]) -> None:
        self._value_side = "lower"  # boundary used for unconstrained-by-path values: lower | upper
        self.tree = copy.deepcopy(dict(tree))
        self.nodes = {
            _node_key(node.get("node_id")): dict(node)
            for node in (self.tree.get("nodes") or [])
            if isinstance(node, Mapping) and node.get("node_id") is not None
        }
        self.children: Dict[str, Tuple[str, ...]] = {}
        for node_id, node in self.nodes.items():
            raw_children = node.get("children_ids") or []
            self.children[node_id] = tuple(_node_key(ch) for ch in raw_children)
        self.edges = tuple(edge for edge in (self.tree.get("edges") or []) if isinstance(edge, Mapping))
        self.replay_successes: set[str] = set()
        self.replay_failures: Dict[str, Tuple[str, str]] = {}
        self._legacy_by_canonical: Dict[str, str] = {}
        self._constraint_specs_by_canonical = self._build_constraint_specs()
        self._node_boundaries_by_packet: Dict[str, Dict[str, Tuple[FieldBoundary, ...]]] = {}

    def _build_constraint_specs(self) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}
        for node_id in sorted(self.nodes, key=_stable_node_id_key):
            node = self.nodes[node_id]
            name = _node_name(node, node_id)
            for raw in node.get("constraints") or []:
                if not isinstance(raw, str):
                    continue
                text = raw.strip()
                kind = None
                if _parse_enum_constraint(text):
                    kind = "enum"
                elif _parse_range_constraint(text) is not None:
                    kind = "range"
                elif parse_numeric_calculated_upper_range_constraint_text(text) is not None:
                    kind = "range_lower"
                elif text.lower().startswith("formula:"):
                    kind = "formula"
                if kind is None:
                    continue
                specs[pkg_constraint_id_for_node(node_id, name, text)] = {
                    "kind": kind,
                    "node_id": node_id,
                    "node_name": name,
                    "constraint": text,
                }
        for edge in self.edges:
            rel = str(edge.get("rel") or "").strip().lower()
            if rel != "length_of":
                continue
            src = _node_key(edge.get("src"))
            dst = _node_key(edge.get("dst"))
            formula = str(edge.get("formula") or "")
            cid = pkg_constraint_id_for_edge(
                rel,
                src,
                _node_name(self.nodes.get(src, {}), src),
                dst,
                _node_name(self.nodes.get(dst, {}), dst),
                edge.get("formula"),
                edge.get("message_type"),
            )
            specs[cid] = {
                "kind": "length_of",
                "src": src,
                "dst": dst,
                "formula": formula,
                "message_type": (
                    str(edge.get("message_type") or "").strip().lower()
                    or "bidirectional"
                ),
            }
        return specs

    def enumerate_paths(self, *, max_sequence_items: int = 3) -> Iterable[PkgParsingPath]:
        variants = [
            (node_id, node)
            for node_id, node in self.nodes.items()
            if _node_type(node) == "variant"
            and (self._fixed_fields_under(node_id) or self._incoming_condition_edges(node_id))
        ]
        nested_variant_ancestors = {
            ancestor_id
            for variant_id, _node in variants
            for ancestor_id in self._ancestors(variant_id)[1:]
            if _node_type(self.nodes.get(ancestor_id, {})) == "variant"
        }
        variants = [
            (variant_id, variant)
            for variant_id, variant in variants
            if variant_id not in nested_variant_ancestors
        ]
        if not variants:
            node_ids = tuple(self._sorted_fields(self.nodes.keys()))
            if node_ids:
                yield PkgParsingPath(path_id="PKG_Path", node_ids=node_ids)
            return

        condition_sources = {
            _node_key(edge.get("src"))
            for edge in self.edges
            if str(edge.get("rel") or "").lower() == "condition_on" and edge.get("src") is not None
        }

        base_path_by_variant: Dict[str, PkgParsingPath] = {}
        for variant_id, variant in variants:
            base_path_by_variant[variant_id] = self._path_for_variant(
                variant_id,
                variant,
                condition_sources=condition_sources,
            )

        emitted = set()
        grouped = self._tlv_sequence_variant_groups(tuple(base_path_by_variant))
        grouped_variants = {variant_id for group in grouped for variant_id in group[1]}
        for seq_id, group_variants in grouped:
            empty = self._empty_sequence_path(seq_id, group_variants, base_path_by_variant=base_path_by_variant)
            if empty is not None:
                key = (empty.variant_id, empty.path_id, empty.sequence_items)
                if key not in emitted:
                    emitted.add(key)
                    yield empty
            for path in self._representative_tlv_paths(
                seq_id,
                group_variants,
                base_path_by_variant=base_path_by_variant,
                max_sequence_items=max_sequence_items,
            ):
                key = (path.variant_id, path.path_id, path.sequence_items)
                if key not in emitted:
                    emitted.add(key)
                    yield path

        for variant_id, _variant in variants:
            if variant_id in grouped_variants:
                continue
            path = base_path_by_variant[variant_id]
            key = (path.variant_id, path.path_id, path.sequence_items)
            if key not in emitted:
                emitted.add(key)
                yield path

    def _path_for_variant(
        self,
        variant_id: str,
        variant: Mapping[str, Any],
        *,
        condition_sources: set[str],
        include_common_suffix: bool = True,
    ) -> PkgParsingPath:
        variant_start = self._resolve_bit_start(variant_id)
        variant_size = self._resolve_size_bits(variant_id)
        variant_end = (
            int(variant_start) + int(variant_size)
            if variant_start is not None and variant_size is not None and variant_size >= 0
            else None
        )
        if variant_end is None:
            descendant_ends = []
            for field_id in self._fixed_fields_under(variant_id):
                field_start = self._resolve_bit_start(field_id)
                field_size = self._resolve_size_bits(field_id)
                if field_start is not None and field_size is not None and field_size >= 0:
                    descendant_ends.append(int(field_start) + int(field_size))
            if descendant_ends:
                variant_end = max(descendant_ends)
        active_variants = {
            ancestor_id
            for ancestor_id in self._ancestors(variant_id)
            if _node_type(self.nodes.get(ancestor_id, {})) == "variant"
        }
        active_fields = []
        for node_id, node in self.nodes.items():
            if _node_type(node) not in self.FIELD_TYPES:
                continue
            field_variants = {
                ancestor_id
                for ancestor_id in self._ancestors(node_id)[1:]
                if _node_type(self.nodes.get(ancestor_id, {})) == "variant"
            }
            if not field_variants.issubset(active_variants):
                continue
            bit_start = self._resolve_bit_start(node_id)
            is_common_suffix = (
                include_common_suffix
                and not field_variants
                and variant_end is not None
                and bit_start is not None
                and bit_start >= variant_end
            )
            if (
                is_common_suffix
                or field_variants
                or node_id in condition_sources
                or variant_start is None
                or bit_start is None
                or bit_start < variant_start
            ):
                active_fields.append(node_id)

        node_ids = tuple(self._sorted_fields(active_fields))
        variant_name = str(variant.get("name") or f"variant_{variant_id}")
        return PkgParsingPath(
            path_id=variant_name,
            node_ids=node_ids,
            variant_id=variant_id,
            variant_name=variant_name,
            covered_branch_ids=tuple(
                ancestor_id
                for ancestor_id in reversed(self._ancestors(variant_id))
                if _node_type(self.nodes.get(ancestor_id, {})) == "variant"
            ),
        )

    def _incoming_condition_edges(self, variant_id: str) -> Tuple[Mapping[str, Any], ...]:
        descendants = set(self._descendants(variant_id))
        return tuple(
            edge
            for edge in self.edges
            if str(edge.get("rel") or "").strip().lower() == "condition_on"
            and _node_key(edge.get("dst")) in descendants
        )

    def _tlv_sequence_ancestor(self, node_id: str) -> Optional[str]:
        seen = set()
        current: Optional[str] = node_id
        while current is not None and current not in seen:
            seen.add(current)
            node = self.nodes.get(current)
            if not node:
                return None
            if _node_type(node) == "tlv_seq":
                return current
            parent = node.get("parent_id")
            current = None if parent is None else _node_key(parent)
        return None

    def _subtree_ids(self, root_id: str) -> set[str]:
        out: set[str] = set()
        stack = [root_id]
        while stack:
            current = stack.pop()
            if current in out:
                continue
            out.add(current)
            stack.extend(_node_key(child) for child in (self.nodes.get(current, {}).get("children_ids") or []))
        return out

    def _empty_sequence_path(
        self,
        seq_id: str,
        variant_ids: Sequence[str],
        *,
        base_path_by_variant: Mapping[str, PkgParsingPath],
    ) -> Optional[PkgParsingPath]:
        """The path on which a sequence region holds no items; replay decides whether the PKG allows it."""
        if not variant_ids:
            return None
        inside = self._subtree_ids(seq_id)
        base = base_path_by_variant[variant_ids[0]]
        node_ids = tuple(node_id for node_id in base.node_ids if node_id not in inside)
        if not node_ids:
            return None
        name = f"{self._variant_name(seq_id)}__empty"
        return PkgParsingPath(path_id=name, node_ids=node_ids, variant_name=name)

    def _enclosing_length_sources(self, seq_id: str) -> Tuple[Tuple[str, str, str], ...]:
        """length_of edges whose destination encloses the sequence, innermost container first."""
        enclosing = self._ancestors(seq_id)[1:]
        rank = {node_id: idx for idx, node_id in enumerate(enclosing)}
        found = []
        for edge in self.edges:
            if str(edge.get("rel") or "").strip().lower() != "length_of":
                continue
            src, dst = _node_key(edge.get("src")), _node_key(edge.get("dst"))
            if dst in rank and src not in rank:
                found.append((rank[dst], src, dst, str(edge.get("formula") or "")))
        return tuple((src, dst, formula) for _rank, src, dst, formula in sorted(found))

    def _trailing_padding_child(self, container_id: str) -> Optional[str]:
        """A variable-size child that may hold zero padding (named pad/padding or constrained to zero bytes)."""
        for child in reversed(self.nodes.get(container_id, {}).get("children_ids") or []):
            child_id = _node_key(child)
            node = self.nodes.get(child_id, {})
            if str(node.get("size_bits") or "").strip().lower() not in {"variable", "var", ""} and not isinstance(
                node.get("size_bits"), str
            ):
                continue
            name = str(node.get("name") or "")
            constraints = " ".join(str(item) for item in (node.get("constraints") or []))
            if re.search(r"(^|[_\s-])(pad|padding)([_\s-]|$)", name, re.I) or re.search(
                r"bytes\s*==\s*0x0*0\b", constraints, re.I
            ):
                return child_id
        return None

    def _align_sequence_enclosures(
        self,
        raw: bytes,
        length_sources: Sequence[Tuple[str, str, str]],
        first_bounds: Mapping[str, Sequence[FieldBoundary]],
    ) -> Tuple[bytes, Dict[str, int], Optional[Tuple[str, FieldBoundary]]]:
        updates: Dict[str, int] = {}
        padding: Optional[Tuple[str, FieldBoundary]] = None
        for src, dst, formula in length_sources:
            src_bounds = tuple(first_bounds.get(src) or ())
            if len(src_bounds) != 1:
                raise ValueError(f"length source {src} of enclosing container {dst} has no unique boundary")
            dst_start = self._resolve_bit_start(dst, {})
            if dst_start is None:
                raise ValueError(f"cannot resolve the start of enclosing container {dst}")
            f0, f1, f2 = (_eval_formula_with_values(formula, {src: k}) for k in (0, 1, 2))
            if None in (f0, f1, f2) or f1 - f0 <= 0 or f2 - f1 != f1 - f0:
                raise ValueError(f"length relation {src}->{dst} is not affine in its source")
            unit, offset = int(f1 - f0), int(f0)
            needed = len(raw) * 8 - int(dst_start)
            value = max(0, -(-(needed - offset) // unit))
            extra = unit * value + offset - needed
            if extra:
                pad_id = self._trailing_padding_child(dst)
                if extra % 8 or pad_id is None:
                    raise ValueError(f"content of {dst} does not fill its length unit and no padding field exists")
                padding = (pad_id, FieldBoundary(start_bit=len(raw) * 8, bit_length=int(extra)))
                raw = raw + bytes(int(extra) // 8)
            boundary = src_bounds[0]
            raw = _patch_field_value(
                raw, boundary.start_bit, boundary.bit_length, int(value), _node_endianness(self.nodes.get(src, {}))
            )
            updates[src] = int(value)
        return raw, updates, padding

    def _tlv_sequence_variant_groups(
        self,
        variant_ids: Sequence[str],
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        groups: Dict[str, list[str]] = {}
        for variant_id in variant_ids:
            seq_id = self._tlv_sequence_ancestor(variant_id)
            if seq_id is None:
                continue
            groups.setdefault(seq_id, []).append(variant_id)
        return tuple((seq_id, tuple(items)) for seq_id, items in groups.items())

    def _representative_tlv_paths(
        self,
        seq_id: str,
        variant_ids: Sequence[str],
        *,
        base_path_by_variant: Mapping[str, PkgParsingPath],
        max_sequence_items: int,
    ) -> Iterable[PkgParsingPath]:
        cap = max(1, int(max_sequence_items))
        tree_cap = _coerce_optional_int(self.nodes.get(seq_id, {}).get("max_items"))
        if tree_cap is not None:
            cap = min(cap, max(1, tree_cap))
        zero_body = [variant_id for variant_id in variant_ids if not self._fixed_fields_under(variant_id)]
        terminal = self._terminal_variant_from_stop_condition(seq_id, variant_ids)
        if terminal is None:
            terminal = next(
                (
                    variant_id
                    for variant_id in zero_body
                    if re.search(r"(^|[_\s-])(end|stop|eol|terminator)([_\s-]|$)", self._variant_name(variant_id), re.I)
                ),
                None,
            )
        pad = next(
            (
                variant_id
                for variant_id in zero_body
                if re.search(r"(^|[_\s-])(pad|padding)([_\s-]|$)", self._variant_name(variant_id), re.I)
            ),
            None,
        )
        ordinary = [variant_id for variant_id in variant_ids if variant_id not in {terminal, pad}]

        sequences: list[Tuple[str, ...]] = []
        if terminal is not None:
            sequences.append((terminal,))
        if pad is not None and (terminal is None or cap >= 2):
            sequences.append((pad, terminal) if terminal is not None else (pad,))
        for variant_id in ordinary:
            if terminal is not None and cap < 2:
                continue
            sequences.append((variant_id, terminal) if terminal is not None else (variant_id,))

        representative = ordinary[0] if ordinary else None
        if representative is not None and terminal is not None and cap >= 3:
            sequences.append((representative, representative, terminal))
            if pad is not None:
                sequences.append((pad, representative, terminal))

        for items in sequences:
            bounded = tuple(items[:cap])
            if not bounded:
                continue
            component_paths = [base_path_by_variant[item] for item in bounded]
            names = [str(path.variant_name or path.path_id) for path in component_paths]
            node_ids = tuple(
                self._sorted_fields(
                    node_id
                    for component in component_paths
                    for node_id in component.node_ids
                )
            )
            yield PkgParsingPath(
                path_id="__".join(names),
                node_ids=node_ids,
                variant_id=bounded[0],
                variant_name="__".join(names),
                sequence_items=bounded,
                covered_branch_ids=tuple(dict.fromkeys(bounded)),
            )

    def _terminal_variant_from_stop_condition(
        self,
        seq_id: str,
        variant_ids: Sequence[str],
    ) -> Optional[str]:
        stop_formula = str(
            self.nodes.get(seq_id, {}).get("stop_condition")
            or self.nodes.get(seq_id, {}).get("repeat_until")
            or ""
        ).strip()
        if not stop_formula:
            return None
        for variant_id in variant_ids:
            for edge in self._incoming_condition_edges(variant_id):
                selector_id = _node_key(edge.get("src"))
                selector = self.nodes.get(selector_id, {})
                witness = _selector_value_from_formula(
                    str(edge.get("formula") or ""),
                    selector_id=selector_id,
                    bit_length=self._resolve_size_bits(selector_id) or 8,
                    selector_node=selector,
                )
                if witness is not None and _formula_is_true(stop_formula, {selector_id: witness}):
                    return variant_id
        return None

    def _variant_name(self, variant_id: str) -> str:
        return str(self.nodes.get(variant_id, {}).get("name") or f"variant_{variant_id}")

    def branch_constraint_ids(self) -> set[str]:
        return {
            self._condition_constraint_id(edge)
            for edge in self.edges
            if str(edge.get("rel") or "").strip().lower() == "condition_on"
        }

    def branch_variant_ids(self) -> set[str]:
        return {
            _node_key(edge.get("dst"))
            for edge in self.edges
            if str(edge.get("rel") or "").strip().lower() == "condition_on"
            and _node_type(self.nodes.get(_node_key(edge.get("dst")), {})) == "variant"
        }

    def positive_only_branch_constraint_ids(self) -> Tuple[str, ...]:
        # A zero-width branch has no body whose absence or inconsistency can be
        # turned into a meaningful negative packet.  Other condition_on edges
        # remain runtime obligations in the Stage-C contract: their legal
        # selector witnesses provide positive coverage, but do not replace a
        # branch-body counterexample.
        out: list[str] = []
        for edge in self.edges:
            if str(edge.get("rel") or "").strip().lower() != "condition_on":
                continue
            dst = _node_key(edge.get("dst"))
            if self._resolve_size_bits(dst) != 0:
                continue
            has_nonzero_leaf = False
            for node_id in self._descendants(dst):
                if node_id == dst:
                    continue
                node = self.nodes.get(node_id)
                if not node or _node_type(node) not in self.FIELD_TYPES:
                    continue
                size_bits = self._resolve_size_bits(node_id)
                if size_bits is None or size_bits > 0:
                    has_nonzero_leaf = True
                    break
            if not has_nonzero_leaf:
                out.append(self._condition_constraint_id(edge))
        return tuple(sorted(out))

    def _condition_constraint_id(self, edge: Mapping[str, Any]) -> str:
        src = _node_key(edge.get("src"))
        dst = _node_key(edge.get("dst"))
        canonical = pkg_constraint_id_for_edge(
            "condition_on",
            src,
            _node_name(self.nodes.get(src, {}), src),
            dst,
            _node_name(self.nodes.get(dst, {}), dst),
            edge.get("formula"),
            edge.get("message_type"),
        )
        self._legacy_by_canonical[canonical] = constraint_id_for_edge(
            "condition_on",
            _node_name(self.nodes.get(src, {}), src),
            _node_name(self.nodes.get(dst, {}), dst),
            edge.get("formula"),
        )
        return canonical

    def materialize_positive(self, path: PkgParsingPath, *, packet_id: str) -> PositiveSample:
        if len(path.sequence_items) > 1:
            return self._materialize_sequence_positive(path, packet_id=packet_id)
        try:
            sample = self._materialize_single_positive(path, packet_id=packet_id)
            self._replay_validate(
                sample,
                expected_variant_ids=path.sequence_items or path.covered_branch_ids,
                exact_sequence=bool(path.sequence_items),
                strict=True,
            )
            return sample
        except ValueError as first_error:
            if not path.sequence_items:
                raise
            # A one-item sequence may need its enclosing length fields and padding recomputed.
            try:
                sample = self._materialize_sequence_positive(path, packet_id=packet_id)
            except Exception:
                raise first_error
            self.replay_failures.pop(packet_id, None)
            return sample

    def _materialize_single_positive(self, path: PkgParsingPath, *, packet_id: str) -> PositiveSample:
        forced_values = self._path_forced_values(path)
        fields = []
        values_by_id: Dict[str, int] = dict(forced_values)
        for node_id in path.node_ids:
            node = self.nodes.get(node_id)
            if not node or _node_type(node) not in self.FIELD_TYPES:
                continue
            start_bit = self._resolve_bit_start(node_id, values_by_id)
            bit_length = self._resolve_size_bits(node_id, values_by_id)
            if start_bit is None or bit_length is None or bit_length <= 0:
                continue
            name = _node_name(node, node_id)
            value = values_by_id.get(node_id)
            if value is None:
                value = self._choose_value(node)
                values_by_id[node_id] = int(value)
            fields.append((int(start_bit), int(bit_length), name, node, int(value), node_id))

        if not fields:
            raise ValueError("path has no materializable fixed-width fields")

        total_bits = max(start + size for start, size, _name, _node, _value, _node_id in fields)
        raw = bytes((total_bits + 7) // 8)
        field_values: Dict[str, Any] = {}
        boundaries: Dict[str, FieldBoundary] = {}
        materialized_ids = set()
        for start_bit, bit_length, name, node, value, node_id in sorted(fields, key=lambda item: (item[0], item[5])):
            raw = _patch_field_value(raw, start_bit, bit_length, value, _node_endianness(node))
            field_values[name] = value
            boundaries[name] = FieldBoundary(start_bit=start_bit, bit_length=bit_length)
            values_by_id[node_id] = int(value)
            materialized_ids.add(node_id)

        selected_node_ids = set(path.node_ids)
        for selected in (*path.node_ids, *(path.covered_branch_ids or ())):
            selected_node_ids.update(self._ancestors(selected))
        raw, field_values, values_by_id = self._repair_positive_values(
            raw=raw,
            field_values=field_values,
            values_by_id=values_by_id,
            fields=fields,
            selected_node_ids=selected_node_ids,
        )
        sample = PositiveSample(
            packet_id=packet_id,
            raw_bytes=raw,
            path_id=path.path_id,
            variant_id=path.variant_id,
            field_values=field_values,
            field_boundaries=boundaries,
            activated_constraints=self._activated_constraints(
                materialized_ids,
                values_by_id=values_by_id,
                path=path,
            ),
        )
        valid, reason = self._validate_positive(
            sample,
            values_by_id=values_by_id,
            materialized_ids=materialized_ids,
        )
        if not valid:
            raise ValueError(f"positive seed violates PKG constraints: {reason}")
        by_node: Dict[str, list[FieldBoundary]] = {}
        for start_bit, bit_length, _name, _node, _value, node_id in fields:
            by_node.setdefault(node_id, []).append(
                FieldBoundary(start_bit=int(start_bit), bit_length=int(bit_length))
            )
        self._node_boundaries_by_packet[packet_id] = {
            node_id: tuple(items)
            for node_id, items in by_node.items()
        }
        return sample

    def _materialize_sequence_positive(self, path: PkgParsingPath, *, packet_id: str) -> PositiveSample:
        seq_ids = {self._tlv_sequence_ancestor(item) for item in path.sequence_items}
        if len(seq_ids) != 1 or None in seq_ids:
            raise ValueError("sequence path items do not share one tlv_seq ancestor")
        seq_id = next(iter(seq_ids))
        assert seq_id is not None
        seq_start = self._resolve_bit_start(seq_id)
        if seq_start is None or seq_start < 0 or seq_start % 8:
            raise ValueError("tlv_seq start must resolve to a non-negative byte boundary")

        condition_sources = {
            _node_key(edge.get("src"))
            for edge in self.edges
            if str(edge.get("rel") or "").strip().lower() == "condition_on" and edge.get("src") is not None
        }
        component_samples = []
        for occurrence, variant_id in enumerate(path.sequence_items):
            component_path = self._path_for_variant(
                variant_id,
                self.nodes[variant_id],
                condition_sources=condition_sources,
                include_common_suffix=False,
            )
            component_samples.append(
                self._materialize_single_positive(
                    component_path,
                    packet_id=f"{packet_id}_item_{occurrence}",
                )
            )

        start_byte = seq_start // 8
        length_sources = self._enclosing_length_sources(seq_id)
        first_bounds = self._node_boundaries_by_packet.get(component_samples[0].packet_id, {})
        masked_ranges = [
            (boundary.start_bit, boundary.bit_length)
            for src, _dst, _formula in length_sources
            for boundary in first_bounds.get(src, ())
            if boundary.start_bit + boundary.bit_length <= start_byte * 8
        ]

        def _without_length_fields(data: bytes) -> bytes:
            for start, length in masked_ranges:
                data = _patch_field_value(data, start, length, 0, "big")
            return data

        prefix = component_samples[0].raw_bytes[:start_byte]
        for component in component_samples[1:]:
            if _without_length_fields(component.raw_bytes[:start_byte]) != _without_length_fields(prefix):
                raise ValueError("tlv_seq component prefixes are inconsistent")

        chunks = [component.raw_bytes[start_byte:] for component in component_samples]
        raw = prefix + b"".join(chunks)
        raw, length_updates, padding = self._align_sequence_enclosures(raw, length_sources, first_bounds)
        field_values: Dict[str, Any] = {}
        boundaries: Dict[str, FieldBoundary] = {}
        activated: list[str] = []

        def add_occurrence(name: str, value: Any, boundary: FieldBoundary) -> None:
            key = name
            suffix = 2
            while key in boundaries:
                key = f"{name}#{suffix}"
                suffix += 1
            field_values[key] = value
            boundaries[key] = boundary

        for name, boundary in component_samples[0].field_boundaries.items():
            if boundary.start_bit < seq_start:
                add_occurrence(name, component_samples[0].field_values.get(name), boundary)

        cursor_bits = seq_start
        for component in component_samples:
            for name, boundary in component.field_boundaries.items():
                if boundary.start_bit < seq_start:
                    continue
                shifted = FieldBoundary(
                    start_bit=cursor_bits + boundary.start_bit - seq_start,
                    bit_length=boundary.bit_length,
                )
                add_occurrence(name, component.field_values.get(name), shifted)
            cursor_bits += max(0, len(component.raw_bytes) * 8 - seq_start)
            activated.extend(component.activated_constraints)

        for src_id, value in length_updates.items():
            src_name = _node_name(self.nodes.get(src_id, {}), src_id)
            if src_name in field_values:
                field_values[src_name] = value
        if padding is not None:
            pad_id, pad_boundary = padding
            add_occurrence(_node_name(self.nodes.get(pad_id, {}), pad_id), 0, pad_boundary)

        sample = PositiveSample(
            packet_id=packet_id,
            raw_bytes=raw,
            path_id=path.path_id,
            variant_id=path.variant_id,
            field_values=field_values,
            field_boundaries=boundaries,
            activated_constraints=list(dict.fromkeys(activated)),
        )
        combined_by_node: Dict[str, list[FieldBoundary]] = {}
        first_component_bounds = self._node_boundaries_by_packet.get(component_samples[0].packet_id, {})
        for node_id, node_boundaries in first_component_bounds.items():
            for boundary in node_boundaries:
                if boundary.start_bit < seq_start:
                    combined_by_node.setdefault(node_id, []).append(boundary)
        cursor_bits = seq_start
        for component in component_samples:
            component_bounds = self._node_boundaries_by_packet.get(component.packet_id, {})
            for node_id, node_boundaries in component_bounds.items():
                for boundary in node_boundaries:
                    if boundary.start_bit < seq_start:
                        continue
                    combined_by_node.setdefault(node_id, []).append(
                        FieldBoundary(
                            start_bit=cursor_bits + boundary.start_bit - seq_start,
                            bit_length=boundary.bit_length,
                        )
                    )
            cursor_bits += max(0, len(component.raw_bytes) * 8 - seq_start)
        if padding is not None:
            combined_by_node.setdefault(padding[0], []).append(padding[1])
        self._node_boundaries_by_packet[packet_id] = {
            node_id: tuple(items)
            for node_id, items in combined_by_node.items()
        }
        self._replay_validate(
            sample,
            expected_variant_ids=path.sequence_items,
            exact_sequence=True,
            strict=True,
        )
        return sample

    def _replay_validate(
        self,
        sample: PositiveSample,
        *,
        expected_variant_ids: Sequence[str],
        exact_sequence: bool = False,
        strict: bool,
    ) -> None:
        from pargen.protocol_knowledge.message_validation.interpreter import DynamicTreeInterpreter

        target_message_type = None
        if sample.variant_id:
            candidate = str(self.nodes.get(sample.variant_id, {}).get("message_type") or "").strip().lower()
            if candidate in {"request", "response"}:
                target_message_type = candidate
        ok, context, error = DynamicTreeInterpreter(
            copy.deepcopy(self.tree),
            target_message_type=target_message_type,
        ).parse(sample.raw_bytes)
        if not ok or context is None:
            detail = f"{type(error).__name__}: {error}" if error is not None else "unknown parse failure"
            reason = f"tree replay rejected synthetic positive: {detail}"
        else:
            if exact_sequence and expected_variant_ids:
                seq_id = self._tlv_sequence_ancestor(_node_key(expected_variant_ids[0]))
                seq_variants = {
                    node_id
                    for node_id, node in self.nodes.items()
                    if _node_type(node) == "variant" and self._tlv_sequence_ancestor(node_id) == seq_id
                }
                actual = tuple(
                    node_id
                    for raw_node_id, _parent in context.visit_log
                    for node_id in (_node_key(raw_node_id),)
                    if node_id in seq_variants
                )
                expected = tuple(_node_key(node_id) for node_id in expected_variant_ids)
                # Trailing zero-body items (e.g. zero padding read as EOL options) leave the intended
                # branches covered; the reference output is the interpreter's own parse.
                covered = actual == expected or (
                    actual[: len(expected)] == expected
                    and all(not self._fixed_fields_under(item) for item in actual[len(expected):])
                )
                reason = (
                    ""
                    if covered
                    else f"tree replay branch sequence mismatch: expected {expected}, observed {actual}"
                )
            else:
                visited = {_node_key(node_id) for node_id in context.visited_nodes}
                missing = [variant_id for variant_id in expected_variant_ids if variant_id not in visited]
                reason = (
                    f"tree replay did not visit expected branch(es): {', '.join(missing)}"
                    if missing
                    else ""
                )
        if reason:
            self.replay_failures[sample.packet_id] = (sample.path_id, reason)
            if strict:
                raise ValueError(reason)
            return
        self.replay_successes.add(sample.packet_id)

    def generate_condition_body_negatives(self, sample: PositiveSample) -> Tuple[NegativeSample, ...]:
        """Construct tree-derived ``condition_on`` truncation counterexamples.

        A selector value that names a sibling branch is still a valid packet, so
        selector flipping alone cannot violate ``condition_on``.  Instead, a
        replay-valid path supplies an actually selected selector/target occurrence:
        the mutation keeps its legal prefix and ends immediately before the
        selected branch's first positive-width body node.

        Every returned sample is replayed through ``DynamicTreeInterpreter``.
        Its note contains a deterministic certificate recording the source and
        target spans, the selected edge, and the exact descendant whose read
        crossed EOF.  No protocol, field name, or selector value is assumed.
        """

        from pargen.protocol_knowledge.message_validation.interpreter import DynamicTreeInterpreter

        target_message_type = self._target_message_type_for_sample(sample)
        ok, context, _error = DynamicTreeInterpreter(
            copy.deepcopy(self.tree),
            target_message_type=target_message_type,
        ).parse(sample.raw_bytes)
        if not ok or context is None:
            return ()

        spans = [
            dict(span)
            for span in (getattr(context, "node_spans", ()) or ())
            if isinstance(span, Mapping)
        ]
        if not spans:
            return ()

        activated = {str(cid) for cid in sample.activated_constraints}
        generated: list[NegativeSample] = []
        for edge in self.edges:
            if str(edge.get("rel") or "").strip().lower() != "condition_on":
                continue
            cid = self._condition_constraint_id(edge)
            if cid not in activated:
                continue

            src = _node_key(edge.get("src"))
            dst = _node_key(edge.get("dst"))
            direct = self._direct_condition_body_negative(
                sample,
                edge=edge,
                cid=cid,
                src=src,
                dst=dst,
                context=context,
                spans=spans,
                target_message_type=target_message_type,
            )
            if direct is not None:
                generated.append(direct)

        return tuple(generated)

    def _direct_condition_body_negative(
        self,
        sample: PositiveSample,
        *,
        edge: Mapping[str, Any],
        cid: str,
        src: str,
        dst: str,
        context: Any,
        spans: Sequence[Mapping[str, Any]],
        target_message_type: Optional[str],
    ) -> Optional[NegativeSample]:
        """Cut a selected occurrence immediately before its first body leaf."""

        from pargen.protocol_knowledge.message_validation.interpreter import DynamicTreeInterpreter

        target_descendants = set(self._descendants(dst))
        target_spans = [
            span
            for span in spans
            if _node_key(span.get("node_id")) == dst
            and self._span_int(span, "wire_size_bits") > 0
        ]
        selector_spans = [
            span
            for span in spans
            if _node_key(span.get("node_id")) == src
            and span.get("value") is not None
            and self._span_int(span, "wire_size_bits") > 0
        ]
        if not target_spans or not selector_spans:
            return None

        if _node_type(self.nodes.get(dst, {})) == "variant":
            active_targets = {
                _node_key(active_dst)
                for raw_selector, destinations in (
                    getattr(context, "active_variants", {}) or {}
                ).items()
                if _node_key(raw_selector) == src
                for active_dst in destinations
            }
            if dst not in active_targets:
                return None

        seq_id = self._tlv_sequence_ancestor(dst)
        item_id = (
            self._sequence_item_root(src, seq_id)
            if seq_id is not None and self._tlv_sequence_ancestor(src) == seq_id
            else None
        )
        proposals: list[
            Tuple[
                Tuple[int, Tuple[int, Any], int],
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, Any],
            ]
        ] = []
        for target_span in target_spans:
            target_start = self._span_int(target_span, "bit_start")
            target_end = target_start + self._span_int(target_span, "wire_size_bits")
            if target_end <= target_start:
                continue
            body_spans = [
                span
                for span in spans
                if _node_key(span.get("node_id")) in target_descendants
                and _node_type(self.nodes.get(_node_key(span.get("node_id")), {})) in self.FIELD_TYPES
                and self._span_int(span, "wire_size_bits") > 0
                and self._span_int(span, "bit_start") >= target_start
                and self._span_int(span, "bit_start") + self._span_int(span, "wire_size_bits") <= target_end
            ]
            if _node_type(self.nodes.get(dst, {})) in self.FIELD_TYPES:
                body_spans.append(target_span)
            body_spans = sorted(
                {
                    (
                        _node_key(span.get("node_id")),
                        self._span_int(span, "bit_start"),
                        self._span_int(span, "wire_size_bits"),
                    ): span
                    for span in body_spans
                }.values(),
                key=lambda span: (
                    self._span_int(span, "bit_start"),
                    _stable_node_id_key(_node_key(span.get("node_id"))),
                ),
            )
            if not body_spans:
                continue
            first_body = body_spans[0]
            body_start = self._span_int(first_body, "bit_start")

            for selector_span in selector_spans:
                selector_start = self._span_int(selector_span, "bit_start")
                selector_end = selector_start + self._span_int(selector_span, "wire_size_bits")
                if selector_end > target_start or selector_end > body_start:
                    continue
                values = self._occurrence_values_before(spans, target_start)
                try:
                    values[src] = int(selector_span.get("value"))
                except (TypeError, ValueError):
                    continue
                if _try_formula_truth(str(edge.get("formula") or ""), values) is not True:
                    continue
                if item_id is not None:
                    selector_item = self._containing_occurrence(spans, item_id, selector_span)
                    target_item = self._containing_occurrence(spans, item_id, target_span)
                    if selector_item is None or target_item is None:
                        continue
                    if (
                        self._span_int(selector_item, "bit_start"),
                        self._span_int(selector_item, "wire_size_bits"),
                    ) != (
                        self._span_int(target_item, "bit_start"),
                        self._span_int(target_item, "wire_size_bits"),
                    ):
                        continue
                proposals.append(
                    (
                        (
                            body_start,
                            _stable_node_id_key(_node_key(first_body.get("node_id"))),
                            -selector_start,
                        ),
                        selector_span,
                        target_span,
                        first_body,
                    )
                )

        for _rank, selector_span, target_span, body_span in sorted(
            proposals,
            key=lambda item: item[0],
        ):
            selector_start = self._span_int(selector_span, "bit_start")
            selector_size = self._span_int(selector_span, "wire_size_bits")
            selector_end = selector_start + selector_size
            target_start = self._span_int(target_span, "bit_start")
            target_size = self._span_int(target_span, "wire_size_bits")
            body_start = self._span_int(body_span, "bit_start")
            if body_start <= 0 or body_start % 8 or selector_start % 8 or selector_end % 8:
                continue
            if body_start < selector_end or body_start >= len(sample.raw_bytes) * 8:
                continue
            raw = sample.raw_bytes[: body_start // 8]
            if not raw or raw == sample.raw_bytes:
                continue

            ok_mutated, mutated_context, mutated_error = DynamicTreeInterpreter(
                copy.deepcopy(self.tree),
                target_message_type=target_message_type,
            ).parse(raw)
            if ok_mutated or mutated_context is None or mutated_error is None:
                continue
            replay_selector = next(
                (
                    span
                    for span in (getattr(mutated_context, "node_spans", ()) or ())
                    if isinstance(span, Mapping)
                    and _node_key(span.get("node_id")) == src
                    and self._span_int(span, "bit_start") == selector_start
                    and span.get("value") is not None
                ),
                None,
            )
            if replay_selector is None:
                continue
            replay_spans = [
                dict(span)
                for span in (getattr(mutated_context, "node_spans", ()) or ())
                if isinstance(span, Mapping)
            ]
            replay_values = self._occurrence_values_before(replay_spans, body_start)
            try:
                selected_value = int(replay_selector.get("value"))
            except (TypeError, ValueError):
                continue
            replay_values[src] = selected_value
            if _try_formula_truth(str(edge.get("formula") or ""), replay_values) is not True:
                continue

            if _node_type(self.nodes.get(dst, {})) == "variant":
                replay_targets = {
                    _node_key(active_dst)
                    for raw_selector, destinations in (
                        getattr(mutated_context, "active_variants", {}) or {}
                    ).items()
                    if _node_key(raw_selector) == src
                    for active_dst in destinations
                }
                if dst not in replay_targets:
                    continue

            missing_id = self._descendant_oob_error_node(mutated_error, target_descendants)
            if missing_id is None:
                continue
            overlaps = self._coincident_constraints_for_missing_node(sample, missing_id)
            certificate = {
                "certificate": "tree_condition_on_missing_body_v1",
                "mutation": "direct_prefix_truncation",
                "target": cid,
                "source_packet_id": sample.packet_id,
                "sequence_node_id": _json_node_id(seq_id) if seq_id is not None else None,
                "sequence_stop_condition": (
                    str(
                        self.nodes.get(seq_id, {}).get("stop_condition")
                        or self.nodes.get(seq_id, {}).get("repeat_until")
                        or ""
                    ).strip()
                    if seq_id is not None
                    else None
                ),
                "item_node_id": _json_node_id(item_id) if item_id is not None else None,
                "selector_node_id": _json_node_id(src),
                "target_node_id": _json_node_id(dst),
                "condition_formula": str(edge.get("formula") or ""),
                "selected_selector_value": selected_value,
                "selector_occurrence_span_bits": [selector_start, selector_size],
                "target_occurrence_span_bits": [target_start, target_size],
                "truncated_at_bit": body_start,
                "missing_node_id": _json_node_id(missing_id),
                "reference_error_type": type(mutated_error).__name__,
                "exact_descendant_oob": True,
                "target_selected": True,
                "missing_body": True,
                "coincident_constraints": list(overlaps),
            }
            return NegativeSample(
                base_packet_id=sample.packet_id,
                raw_bytes=bytes(raw),
                expected_violated_constraints=[cid],
                note="condition_on_certificate="
                + json.dumps(certificate, sort_keys=True, separators=(",", ":")),
            )
        return None

    @staticmethod
    def _span_int(span: Mapping[str, Any], key: str) -> int:
        try:
            return int(span.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _sequence_item_root(self, node_id: str, seq_id: str) -> Optional[str]:
        for ancestor_id in self._ancestors(node_id):
            parent_id = self.nodes.get(ancestor_id, {}).get("parent_id")
            if parent_id is not None and _node_key(parent_id) == seq_id:
                return ancestor_id
        return None

    def _containing_occurrence(
        self,
        spans: Sequence[Mapping[str, Any]],
        node_id: str,
        nested_span: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        nested_start = self._span_int(nested_span, "bit_start")
        nested_end = nested_start + self._span_int(nested_span, "wire_size_bits")
        candidates = []
        for span in spans:
            if _node_key(span.get("node_id")) != node_id:
                continue
            start = self._span_int(span, "bit_start")
            end = start + self._span_int(span, "wire_size_bits")
            if start <= nested_start and nested_end <= end and end > start:
                candidates.append(span)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda span: (
                self._span_int(span, "wire_size_bits"),
                self._span_int(span, "bit_start"),
            ),
        )

    def _coincident_constraints_for_missing_node(
        self,
        sample: PositiveSample,
        missing_node_id: str,
    ) -> Tuple[str, ...]:
        overlaps = []
        for raw_cid in sample.activated_constraints:
            cid = str(raw_cid)
            if cid.startswith("edge:condition_on:"):
                continue
            spec = self._constraint_specs_by_canonical.get(cid)
            if spec is None:
                continue
            kind = str(spec.get("kind") or "")
            if kind in {"enum", "range", "formula", "range_lower"}:
                matches = _node_key(spec.get("node_id")) == missing_node_id
            elif kind == "length_of":
                # A missing descendant can also truncate any enclosing length
                # destination, even when the destination is not the leaf itself.
                matches = _node_key(spec.get("dst")) in self._ancestors(missing_node_id)
            else:
                matches = False
            if matches:
                overlaps.append(cid)
        return tuple(sorted(dict.fromkeys(overlaps)))

    @staticmethod
    def _descendant_oob_error_node(
        error: BaseException,
        target_descendants: set[str],
    ) -> Optional[str]:
        """Return the exact target-descendant whose read crossed packet EOF."""

        from pargen.protocol_knowledge.message_validation.interpreter import (
            IncompleteStreamError,
            NodeParseError,
            RoutingError,
        )

        seen: set[int] = set()

        def visit(current: Any) -> Optional[str]:
            if current is None or id(current) in seen:
                return None
            seen.add(id(current))
            if isinstance(current, NodeParseError):
                node_id = _node_key(current.node_id)
                if (
                    node_id in target_descendants
                    and isinstance(getattr(current, "cause", None), IncompleteStreamError)
                ):
                    return node_id
                return visit(getattr(current, "cause", None))
            if isinstance(current, RoutingError):
                for variant_error in current.variant_errors:
                    nested = (
                        variant_error[1]
                        if isinstance(variant_error, tuple) and len(variant_error) >= 2
                        else variant_error
                    )
                    matched = visit(nested)
                    if matched is not None:
                        return matched
            return None

        return visit(error)

    def _target_message_type_for_sample(self, sample: PositiveSample) -> Optional[str]:
        if not sample.variant_id:
            return None
        return self._message_type_for_variant_scope(sample.variant_id)

    def _message_type_for_variant_scope(self, variant_id: str) -> Optional[str]:
        candidates = {
            str(self.nodes.get(node_id, {}).get("message_type") or "").strip().lower()
            for node_id in self._ancestors(variant_id)
        }
        candidates.intersection_update({"request", "response"})
        return next(iter(candidates)) if len(candidates) == 1 else None

    def generate_length_truncation_negatives(self, sample: PositiveSample) -> Tuple[NegativeSample, ...]:
        """Construct protocol-independent short-body cases for active ``length_of`` edges.

        The source value and destination occurrence come from a successful tree
        replay.  The packet is cut at the destination start, after the source has
        been decoded.  If the current formula evaluates to zero, a deterministic
        node-constraint-aware search may choose another legal source value whose
        formula requires a positive destination span.  The reference interpreter
        must then report the exact ``(src, dst, formula)`` length edge.
        """

        from pargen.protocol_knowledge.message_validation.interpreter import DynamicTreeInterpreter

        target_message_type = self._target_message_type_for_sample(sample)
        ok, context, _error = DynamicTreeInterpreter(
            copy.deepcopy(self.tree),
            target_message_type=target_message_type,
        ).parse(sample.raw_bytes)
        if not ok or context is None:
            return ()

        spans = sorted(
            (
                dict(span)
                for span in (getattr(context, "node_spans", ()) or ())
                if isinstance(span, Mapping)
            ),
            key=lambda span: (
                self._span_int(span, "bit_start"),
                _stable_node_id_key(_node_key(span.get("node_id"))),
            ),
        )
        activated = {str(cid) for cid in sample.activated_constraints}
        generated: list[NegativeSample] = []
        for edge in self.edges:
            if str(edge.get("rel") or "").strip().lower() != "length_of":
                continue
            src = _node_key(edge.get("src"))
            dst = _node_key(edge.get("dst"))
            cid = pkg_constraint_id_for_edge(
                "length_of",
                src,
                _node_name(self.nodes.get(src, {}), src),
                dst,
                _node_name(self.nodes.get(dst, {}), dst),
                edge.get("formula"),
                edge.get("message_type"),
            )
            if cid not in activated:
                continue
            candidate = self._length_truncation_negative(
                sample,
                edge=edge,
                cid=cid,
                src=src,
                dst=dst,
                spans=spans,
            )
            if candidate is not None:
                generated.append(candidate)
        return tuple(generated)

    def _length_truncation_negative(
        self,
        sample: PositiveSample,
        *,
        edge: Mapping[str, Any],
        cid: str,
        src: str,
        dst: str,
        spans: Sequence[Mapping[str, Any]],
    ) -> Optional[NegativeSample]:
        formula = str(edge.get("formula") or "").strip()
        if not formula:
            return None
        source_spans = [
            span
            for span in spans
            if _node_key(span.get("node_id")) == src
            and span.get("value") is not None
            and self._span_int(span, "wire_size_bits") > 0
        ]
        target_spans = [
            span
            for span in spans
            if _node_key(span.get("node_id")) == dst
            and self._span_int(span, "wire_size_bits") > 0
        ]
        if not source_spans:
            return None

        proposals: list[Tuple[int, Mapping[str, Any], Optional[Mapping[str, Any]]]] = []
        if target_spans:
            for target_span in target_spans:
                target_start = self._span_int(target_span, "bit_start")
                preceding = [
                    span
                    for span in source_spans
                    if self._span_int(span, "bit_start")
                    + self._span_int(span, "wire_size_bits")
                    <= target_start
                ]
                if preceding:
                    proposals.append((target_start, max(preceding, key=lambda s: self._span_int(s, "bit_start")), target_span))
        else:
            for source_span in source_spans:
                values = self._occurrence_values_before(spans, self._span_int(source_span, "bit_start"))
                try:
                    values[src] = int(source_span.get("value"))
                except (TypeError, ValueError):
                    continue
                src_template = self._resolve_bit_start(src, values)
                dst_template = self._resolve_bit_start(dst, values)
                if src_template is None or dst_template is None:
                    continue
                target_start = self._span_int(source_span, "bit_start") + int(dst_template) - int(src_template)
                proposals.append((target_start, source_span, None))

        for target_start, source_span, target_span in sorted(
            proposals,
            key=lambda item: (item[0], -self._span_int(item[1], "bit_start")),
        ):
            if target_start <= 0 or target_start % 8 or target_start > len(sample.raw_bytes) * 8:
                continue
            source_start = self._span_int(source_span, "bit_start")
            source_size = self._span_int(source_span, "wire_size_bits")
            if source_start < 0 or source_size <= 0 or source_start + source_size > target_start:
                continue
            try:
                current_value = int(source_span.get("value"))
            except (TypeError, ValueError):
                continue

            values = self._occurrence_values_before(spans, target_start)
            values[src] = current_value
            expected_bits = _eval_formula_with_values(formula, values)
            selected_value = current_value
            patched = sample.raw_bytes
            if expected_bits is None or int(expected_bits) <= 0:
                selected_value = self._legal_length_source_value(
                    src,
                    current=current_value,
                    values_by_id=values,
                    formula=formula,
                )
                if selected_value is None:
                    continue
                values[src] = int(selected_value)
                expected_bits = _eval_formula_with_values(formula, values)
                if expected_bits is None or int(expected_bits) <= 0:
                    continue
                patched = _patch_field_value(
                    patched,
                    source_start,
                    source_size,
                    int(selected_value),
                    _node_endianness(self.nodes.get(src, {})),
                )

            raw = patched[: target_start // 8]
            if not raw or raw == sample.raw_bytes:
                continue
            if not self._reference_tree_rejects_for_target(sample, raw, cid):
                continue

            certificate = {
                "certificate": "tree_length_of_truncation_v1",
                "target": cid,
                "source_packet_id": sample.packet_id,
                "source_node_id": _json_node_id(src),
                "target_node_id": _json_node_id(dst),
                "formula": formula,
                "source_occurrence_span_bits": [source_start, source_size],
                "target_occurrence_span_bits": (
                    [target_start, self._span_int(target_span, "wire_size_bits")]
                    if target_span is not None
                    else [target_start, 0]
                ),
                "source_value_before": current_value,
                "source_value_after": int(selected_value),
                "expected_bits": int(expected_bits),
                "available_bits": 0,
                "truncated_at_bit": target_start,
                "reference_error_type": "LengthMismatchError",
            }
            return NegativeSample(
                base_packet_id=sample.packet_id,
                raw_bytes=bytes(raw),
                expected_violated_constraints=[cid],
                note="length_of_certificate="
                + json.dumps(certificate, sort_keys=True, separators=(",", ":")),
            )
        return None

    def _occurrence_values_before(
        self,
        spans: Sequence[Mapping[str, Any]],
        bit_position: int,
    ) -> Dict[str, int]:
        values: Dict[str, int] = {}
        for span in spans:
            if self._span_int(span, "bit_start") > int(bit_position):
                break
            if span.get("value") is None:
                continue
            try:
                values[_node_key(span.get("node_id"))] = int(span.get("value"))
            except (TypeError, ValueError):
                continue
        return values

    def _legal_length_source_value(
        self,
        node_id: str,
        *,
        current: int,
        values_by_id: Mapping[str, int],
        formula: str,
    ) -> Optional[int]:
        node = self.nodes.get(node_id, {})
        width = self._resolve_size_bits(node_id, values_by_id)
        if width is None or int(width) <= 0:
            return None
        domain_hi = (1 << int(width)) - 1
        candidates: list[int] = [int(current) + 1, 1, int(current) - 1, 0, domain_hi]
        for raw in node.get("constraints") or []:
            text = str(raw)
            candidates.extend(_parse_enum_constraint(text))
            bounds = _parse_range_constraint(text)
            if bounds is not None and _range_constraint_target(text) == "value":
                lo, hi = (int(bounds[0]), int(bounds[1]))
                candidates.extend((lo, lo + 1, hi - 1, hi))
            lower = _parse_range_lower_bound(text)
            if lower is not None:
                candidates.extend((int(lower), int(lower) + 1))
            for token in re.findall(r"(?<![A-Za-z0-9_])(0x[0-9a-fA-F]+|0b[01]+|-?\d+)", text):
                try:
                    value = _parse_int_token(token)
                except Exception:
                    continue
                candidates.extend((value - 1, value, value + 1))

        for candidate in dict.fromkeys(int(value) for value in candidates):
            if candidate == int(current) or candidate < 0 or candidate > domain_hi:
                continue
            values = dict(values_by_id)
            values[node_id] = candidate
            if not self._node_value_satisfies_constraints(node_id, candidate, values):
                continue
            expected = _eval_formula_with_values(formula, values)
            if expected is not None and int(expected) > 0:
                return candidate
        return None

    def _node_value_satisfies_constraints(
        self,
        node_id: str,
        value: int,
        values_by_id: Mapping[str, int],
    ) -> bool:
        node = self.nodes.get(node_id, {})
        for raw in node.get("constraints") or []:
            text = str(raw).strip()
            enum_values = _parse_enum_constraint(text)
            if enum_values:
                if int(value) not in set(enum_values):
                    return False
                continue
            bounds = _parse_range_constraint(text)
            if bounds is not None:
                if _range_constraint_target(text) == "value" and not (
                    int(bounds[0]) <= int(value) <= int(bounds[1])
                ):
                    return False
                continue
            lower = _parse_range_lower_bound(text)
            if lower is not None:
                if int(value) < int(lower):
                    return False
                continue
            if text.lower().startswith("formula:"):
                expression = text.split(":", 1)[1].strip()
                expression = re.sub(r"\bvalue\b", f"val({node_id})", expression)
                if _try_formula_truth(expression, values_by_id) is not True:
                    return False
                continue
            # A source mutation is unsafe when an executable-looking clause is
            # present but this generator cannot prove it remains satisfied.
            if text:
                return False
        return True

    def negative_keeps_path_reachable(self, sample: PositiveSample, negative: NegativeSample) -> bool:
        """Reject route switches, but retain exact selector-domain violations."""

        boundaries_by_node = self._node_boundaries_by_packet.get(sample.packet_id, {})
        for selector_id in self._condition_selector_ids():
            selector = self.nodes.get(selector_id)
            if not selector:
                continue
            for boundary in boundaries_by_node.get(selector_id, ()):
                if (
                    boundary.start_bit < 0
                    or boundary.start_bit + boundary.bit_length > len(sample.raw_bytes) * 8
                    or boundary.start_bit + boundary.bit_length > len(negative.raw_bytes) * 8
                ):
                    return False
                before = _extract_field_value(sample.raw_bytes, boundary, _node_endianness(selector))
                after = _extract_field_value(negative.raw_bytes, boundary, _node_endianness(selector))
                if before == after:
                    continue
                if not self._selector_change_is_target_domain_violation(
                    selector_id,
                    before=before,
                    after=after,
                    target_ids=negative.expected_violated_constraints,
                ):
                    return False
        return True

    def _selector_change_is_target_domain_violation(
        self,
        selector_id: str,
        *,
        before: int,
        after: int,
        target_ids: Sequence[str],
    ) -> bool:
        for cid in target_ids:
            spec = self._constraint_specs_by_canonical.get(str(cid))
            if spec is None or _node_key(spec.get("node_id")) != selector_id:
                continue
            raw_constraint = str(spec.get("constraint") or "")
            kind = str(spec.get("kind") or "")
            if kind == "enum":
                allowed = set(_parse_enum_constraint(raw_constraint))
                if allowed and int(before) in allowed and int(after) not in allowed:
                    return True
            elif kind == "range" and _range_constraint_target(raw_constraint) == "value":
                bounds = _parse_range_constraint(raw_constraint)
                if bounds is not None:
                    lo, hi = (int(bounds[0]), int(bounds[1]))
                    if lo <= int(before) <= hi and not (lo <= int(after) <= hi):
                        return True
        return False

    def negative_targets_supported_constraint(self, sample: PositiveSample, negative: NegativeSample) -> bool:
        constraints = tuple(str(item) for item in negative.expected_violated_constraints if str(item).strip())
        if not constraints:
            return False
        # Branch selectors are exercised as positive path coverage. Flipping a
        # selector merely chooses a sibling path; it is not a malformed packet.
        if any(cid.startswith("edge:condition_on:") for cid in constraints):
            return False
        return not any(self.constraint_is_tautological_for_sample(sample, cid) for cid in constraints)

    def legacy_activated_constraints(self, sample: PositiveSample) -> list[str]:
        """Translate scoped IDs only for the legacy byte mutator's internal lookup."""

        return list(
            dict.fromkeys(
                self._legacy_by_canonical[cid]
                for cid in sample.activated_constraints
                if cid in self._legacy_by_canonical
            )
        )

    def canonicalize_negative_targets(
        self,
        sample: PositiveSample,
        negative: NegativeSample,
    ) -> Optional[NegativeSample]:
        canonical_targets: list[str] = []
        for legacy_id in negative.expected_violated_constraints:
            matches = [
                cid
                for cid in sample.activated_constraints
                if self._legacy_by_canonical.get(cid) == str(legacy_id)
            ]
            # A name-only legacy mutation cannot identify which simultaneously
            # active duplicate node it changed. Reject ambiguity instead of
            # assigning a potentially wrong contract tag.
            if len(matches) != 1:
                return None
            canonical_targets.append(matches[0])
        if not canonical_targets:
            return None
        return NegativeSample(
            base_packet_id=negative.base_packet_id,
            raw_bytes=bytes(negative.raw_bytes),
            expected_violated_constraints=list(dict.fromkeys(canonical_targets)),
            note=negative.note,
        )

    def prove_negative_targets(
        self,
        sample: PositiveSample,
        negative: NegativeSample,
    ) -> Optional[NegativeSample]:
        """Keep only targets with a direct, node-scoped mutation certificate."""

        proven: list[str] = []
        boundaries_by_node = self._node_boundaries_by_packet.get(sample.packet_id, {})
        before_values = self._values_from_node_boundaries(sample.raw_bytes, boundaries_by_node)
        after_values = self._values_from_node_boundaries(negative.raw_bytes, boundaries_by_node)
        for cid in negative.expected_violated_constraints:
            cid_s = str(cid)
            spec = self._constraint_specs_by_canonical.get(cid_s)
            if spec is None:
                continue
            kind = str(spec.get("kind") or "")
            if kind in {"enum", "range", "formula"}:
                node_id = _node_key(spec.get("node_id"))
                node_boundaries = boundaries_by_node.get(node_id, ())
                changed = [
                    boundary
                    for boundary in node_boundaries
                    if self._mutation_is_confined_to_boundary(sample.raw_bytes, negative.raw_bytes, boundary)
                ]
                if len(changed) != 1:
                    continue
                boundary = changed[0]
                node = self.nodes.get(node_id, {})
                before_value = _extract_field_value(sample.raw_bytes, boundary, _node_endianness(node))
                after_value = _extract_field_value(negative.raw_bytes, boundary, _node_endianness(node))
                if before_value == after_value:
                    continue
                raw_constraint = str(spec.get("constraint") or "")
                if kind == "enum":
                    allowed = _parse_enum_constraint(raw_constraint)
                    if not allowed or int(after_value) in set(allowed):
                        continue
                elif kind == "range":
                    bounds = _parse_range_constraint(raw_constraint)
                    if bounds is None or _range_constraint_target(raw_constraint) != "value":
                        continue
                    if int(bounds[0]) <= int(after_value) <= int(bounds[1]):
                        continue
                else:
                    expression = raw_constraint.split(":", 1)[1].strip() if ":" in raw_constraint else raw_constraint
                    expression = re.sub(r"\bvalue\b", f"val({node_id})", expression)
                    before_ok = _try_formula_truth(expression, before_values)
                    after_ok = _try_formula_truth(expression, after_values)
                    if before_ok is not True or after_ok is not False:
                        continue
                if self._earlier_activated_constraint_is_violated(
                    sample,
                    cid_s,
                    before_values=before_values,
                    after_values=after_values,
                    boundaries_by_node=boundaries_by_node,
                ):
                    continue
                if not self._reference_tree_rejects_for_target(sample, negative.raw_bytes, cid_s):
                    continue
                proven.append(cid_s)
                continue

            if kind == "length_of":
                src = _node_key(spec.get("src"))
                source_boundaries = boundaries_by_node.get(src, ())
                changed = [
                    boundary
                    for boundary in source_boundaries
                    if self._mutation_is_confined_to_boundary(sample.raw_bytes, negative.raw_bytes, boundary)
                ]
                if len(changed) != 1:
                    continue
                before_expected = _eval_formula_with_values(str(spec.get("formula") or ""), before_values)
                after_expected = _eval_formula_with_values(str(spec.get("formula") or ""), after_values)
                if before_expected is None or after_expected is None or before_expected == after_expected:
                    continue
                if self._earlier_activated_constraint_is_violated(
                    sample,
                    cid_s,
                    before_values=before_values,
                    after_values=after_values,
                    boundaries_by_node=boundaries_by_node,
                ):
                    continue
                if not self._reference_tree_rejects_for_target(sample, negative.raw_bytes, cid_s):
                    continue
                proven.append(cid_s)

        proven = list(dict.fromkeys(proven))
        if not proven:
            return None
        return NegativeSample(
            base_packet_id=negative.base_packet_id,
            raw_bytes=bytes(negative.raw_bytes),
            expected_violated_constraints=proven,
            note=negative.note,
        )

    def _reference_tree_rejects_for_target(self, sample: PositiveSample, raw: bytes, cid: str) -> bool:
        """Require the tree interpreter's structured failure to identify this target."""

        from pargen.protocol_knowledge.message_validation.interpreter import (
            ConstraintViolationError,
            DynamicTreeInterpreter,
            LengthMismatchError,
        )

        spec = self._constraint_specs_by_canonical.get(str(cid))
        if spec is None:
            return False
        target_message_type = None
        if sample.variant_id:
            candidate = str(self.nodes.get(sample.variant_id, {}).get("message_type") or "").strip().lower()
            if candidate in {"request", "response"}:
                target_message_type = candidate
        ok, _context, error = DynamicTreeInterpreter(
            copy.deepcopy(self.tree),
            target_message_type=target_message_type,
        ).parse(bytes(raw))
        if ok or error is None:
            return False

        kind = str(spec.get("kind") or "")
        if kind in {"enum", "range", "formula"}:
            if not isinstance(error, ConstraintViolationError):
                return False
            if _node_key(error.node_id) != _node_key(spec.get("node_id")):
                return False
            raw_constraint = str(spec.get("constraint") or "")
            if kind == "enum":
                expected = f"enum {list(_parse_enum_constraint(raw_constraint))}"
                return (
                    str(getattr(error, "kind", "")) == "enum"
                    and _normalize_contract_text(error.constraint_text) == _normalize_contract_text(expected)
                )
            if kind == "range":
                bounds = _parse_range_constraint(raw_constraint)
                if bounds is None:
                    return False
                expected = f"range {bounds[0]}..{bounds[1]} (value)"
                return (
                    str(getattr(error, "kind", "")) == "range"
                    and _normalize_contract_text(error.constraint_text) == _normalize_contract_text(expected)
                )
            return (
                str(getattr(error, "kind", "")) == "expr"
                and _normalize_contract_text(error.constraint_text) == _normalize_contract_text(raw_constraint)
            )

        if kind == "length_of" and isinstance(error, LengthMismatchError):
            expected_message_type = (
                str(spec.get("message_type") or "").strip().lower()
                or "bidirectional"
            )
            return any(
                _node_key(mismatch.src) == _node_key(spec.get("src"))
                and _node_key(mismatch.dst) == _node_key(spec.get("dst"))
                and _normalize_contract_text(mismatch.formula) == _normalize_contract_text(spec.get("formula"))
                and (
                    str(mismatch.message_type or "").strip().lower()
                    or "bidirectional"
                )
                == expected_message_type
                for mismatch in error.mismatches
            )
        return False

    def _earlier_activated_constraint_is_violated(
        self,
        sample: PositiveSample,
        target_cid: str,
        *,
        before_values: Mapping[str, int],
        after_values: Mapping[str, int],
        boundaries_by_node: Mapping[str, Sequence[FieldBoundary]],
    ) -> bool:
        """Reject targets shadowed by an earlier fail-fast tree obligation."""

        for cid in sample.activated_constraints:
            cid_s = str(cid)
            if cid_s == target_cid:
                return False
            if cid_s.startswith("edge:condition_on:"):
                continue
            if self._constraint_is_violated_for_values(
                cid_s,
                before_values=before_values,
                after_values=after_values,
                boundaries_by_node=boundaries_by_node,
            ):
                return True
        return False

    def _constraint_is_violated_for_values(
        self,
        cid: str,
        *,
        before_values: Mapping[str, int],
        after_values: Mapping[str, int],
        boundaries_by_node: Mapping[str, Sequence[FieldBoundary]],
    ) -> bool:
        spec = self._constraint_specs_by_canonical.get(str(cid))
        if spec is None:
            return False
        kind = str(spec.get("kind") or "")
        if kind in {"enum", "range", "formula"}:
            node_id = _node_key(spec.get("node_id"))
            if node_id not in after_values or not boundaries_by_node.get(node_id):
                return False
            raw_constraint = str(spec.get("constraint") or "")
            after_value = int(after_values[node_id])
            if kind == "enum":
                allowed = _parse_enum_constraint(raw_constraint)
                return bool(allowed) and after_value not in set(allowed)
            if kind == "range":
                bounds = _parse_range_constraint(raw_constraint)
                if bounds is None or _range_constraint_target(raw_constraint) != "value":
                    return False
                return not (int(bounds[0]) <= after_value <= int(bounds[1]))
            expression = raw_constraint.split(":", 1)[1].strip() if ":" in raw_constraint else raw_constraint
            expression = re.sub(r"\bvalue\b", f"val({node_id})", expression)
            return (
                _try_formula_truth(expression, before_values) is True
                and _try_formula_truth(expression, after_values) is False
            )
        if kind == "length_of":
            formula = str(spec.get("formula") or "")
            before_expected = _eval_formula_with_values(formula, before_values)
            after_expected = _eval_formula_with_values(formula, after_values)
            return (
                before_expected is not None
                and after_expected is not None
                and before_expected != after_expected
            )
        return False

    def _values_from_node_boundaries(
        self,
        raw: bytes,
        boundaries_by_node: Mapping[str, Sequence[FieldBoundary]],
    ) -> Dict[str, int]:
        values: Dict[str, int] = {}
        for node_id, boundaries in boundaries_by_node.items():
            if len(boundaries) != 1:
                continue
            node = self.nodes.get(node_id, {})
            boundary = boundaries[0]
            if boundary.start_bit + boundary.bit_length > len(raw) * 8:
                continue
            values[node_id] = _extract_field_value(raw, boundary, _node_endianness(node))
        return values

    @staticmethod
    def _mutation_is_confined_to_boundary(before: bytes, after: bytes, boundary: FieldBoundary) -> bool:
        if len(before) != len(after) or before == after:
            return False
        if boundary.start_bit < 0 or boundary.start_bit + boundary.bit_length > len(before) * 8:
            return False
        original = _extract_field_value(before, boundary, "big")
        try:
            restored = _patch_field_value(
                after,
                boundary.start_bit,
                boundary.bit_length,
                original,
                "big",
            )
        except ValueError:
            return False
        return restored == before

    def constraint_is_tautological_for_sample(self, sample: PositiveSample, cid: str) -> bool:
        spec = self._constraint_specs_by_canonical.get(str(cid))
        if spec is None or str(spec.get("kind") or "") not in {"enum", "range"}:
            return False
        node_id = _node_key(spec.get("node_id"))
        boundaries = self._node_boundaries_by_packet.get(sample.packet_id, {}).get(node_id, ())
        if not boundaries:
            return False
        domain_highs = {(1 << int(boundary.bit_length)) - 1 for boundary in boundaries if boundary.bit_length > 0}
        if len(domain_highs) != 1:
            return False
        domain_hi = next(iter(domain_highs))
        raw_constraint = str(spec.get("constraint") or "")
        if _range_constraint_target(raw_constraint) != "value":
            # This constrains the static tree layout, not a mutable on-wire value.
            return True
        bounds = _parse_range_constraint(raw_constraint)
        if bounds is not None:
            lo, hi = bounds
            return int(lo) <= 0 and int(hi) >= domain_hi
        enum_values = set(_parse_enum_constraint(raw_constraint))
        return (
            len(enum_values) == domain_hi + 1
            and min(enum_values, default=0) == 0
            and max(enum_values, default=-1) == domain_hi
        )

    def _condition_selector_ids(self) -> Tuple[str, ...]:
        out = []
        for edge in self.edges:
            if str(edge.get("rel") or "").lower() == "condition_on" and edge.get("src") is not None:
                out.append(_node_key(edge.get("src")))
        return tuple(dict.fromkeys(out))

    def _repair_positive_values(
        self,
        *,
        raw: bytes,
        field_values: Dict[str, Any],
        values_by_id: Dict[str, int],
        fields: Sequence[Tuple[int, int, str, Mapping[str, Any], int, str]],
        selected_node_ids: Optional[set[str]] = None,
    ) -> Tuple[bytes, Dict[str, Any], Dict[str, int]]:
        repaired = bytes(raw)
        repaired_values = dict(field_values)
        repaired_by_id = dict(values_by_id)

        for _ in range(3):
            changed = False
            for start_bit, bit_length, name, node, _value, node_id in fields:
                next_value = self._derived_value_for_node(
                    node_id,
                    repaired,
                    repaired_by_id,
                    fields=fields,
                    selected_node_ids=selected_node_ids,
                )
                if next_value is None:
                    continue
                next_values = {**repaired_by_id, node_id: int(next_value)}
                if (not 0 <= int(next_value) < 1 << int(bit_length)
                        or not self._node_value_satisfies_constraints(node_id, int(next_value), next_values)):
                    continue
                if repaired_by_id.get(node_id) == int(next_value):
                    continue
                repaired_by_id[node_id] = int(next_value)
                repaired_values[name] = int(next_value)
                repaired = _patch_field_value(
                    repaired,
                    int(start_bit),
                    int(bit_length),
                    int(next_value),
                    _node_endianness(node),
                )
                changed = True
            if not changed:
                break

        return repaired, repaired_values, repaired_by_id

    def _derived_value_for_node(
        self,
        node_id: str,
        raw: bytes,
        values_by_id: Mapping[str, int],
        *,
        fields: Sequence[Tuple[int, int, str, Mapping[str, Any], int, str]],
        selected_node_ids: Optional[set[str]] = None,
    ) -> Optional[int]:
        node = self.nodes.get(node_id)
        if not node:
            return None

        for raw_constraint in node.get("constraints") or []:
            value = _solve_formula_constraint(str(raw_constraint), target_id=node_id, values_by_id=values_by_id)
            if value is not None:
                return value

        for edge in self.edges:
            if str(edge.get("rel") or "").strip().lower() != "length_of":
                continue
            if _node_key(edge.get("src")) != node_id:
                continue
            if selected_node_ids is not None and _node_key(edge.get("dst")) not in selected_node_ids:
                continue
            target_bits = self._materialized_target_span_bits(
                _node_key(edge.get("dst")),
                raw=raw,
                values_by_id=values_by_id,
                fields=fields,
                selected_node_ids=selected_node_ids,
            )
            if target_bits is None:
                continue
            solved = _solve_formula_for_value(
                str(edge.get("formula") or ""),
                target_id=node_id,
                target_result=target_bits,
                values_by_id=values_by_id,
            )
            if solved is not None and _eval_formula_with_values(
                    str(edge.get("formula") or ""), {**values_by_id, node_id: solved}) == target_bits:
                return solved

        return None

    def _materialized_target_span_bits(
        self,
        target_id: str,
        *,
        raw: bytes,
        values_by_id: Mapping[str, int],
        fields: Sequence[Tuple[int, int, str, Mapping[str, Any], int, str]],
        selected_node_ids: Optional[set[str]] = None,
    ) -> Optional[int]:
        target = self.nodes.get(target_id)
        if not target:
            return None
        if _node_type(target) == "protocol" or target.get("parent_id") is None:
            return len(raw) * 8

        target_start = self._resolve_bit_start(target_id, values_by_id)
        if target_start is None:
            return None
        descendants = set(self._descendants(target_id))
        ends = [
            int(start_bit) + int(bit_length)
            for start_bit, bit_length, _name, _node, _value, field_id in fields
            if field_id in descendants
        ]
        if not ends:
            # A selected empty suffix has a measured width of zero even when
            # the initial source value makes its formula negative. Solve that
            # formula to zero; do not depend on the replay clamping negatives.
            if (selected_node_ids is not None and target_id in selected_node_ids
                    and int(target_start) == len(raw) * 8):
                return 0
            declared = self._resolve_size_bits(target_id, values_by_id)
            return int(declared) if declared is not None and declared >= 0 else None
        return max(0, max(ends) - int(target_start))

    def _validate_positive(
        self,
        sample: PositiveSample,
        *,
        values_by_id: Mapping[str, int],
        materialized_ids: set[str],
    ) -> Tuple[bool, str]:
        for node_id in self._sorted_fields(materialized_ids):
            node = self.nodes.get(node_id)
            if not node or node_id not in values_by_id:
                continue
            name = _node_name(node, node_id)
            value = int(values_by_id[node_id])
            for raw in node.get("constraints") or []:
                text = str(raw).strip()
                enum_values = _parse_enum_constraint(text)
                if enum_values and value not in enum_values:
                    return False, f"{name}={value} violates {text}"
                bounds = _parse_range_constraint(text)
                if bounds is not None:
                    lo, hi = bounds
                    range_target = _range_constraint_target(text)
                    if range_target == "size_bits":
                        constrained_value = self._resolve_size_bits(node_id, values_by_id)
                    elif range_target == "bit_start":
                        constrained_value = self._resolve_bit_start(node_id, values_by_id)
                    else:
                        constrained_value = value
                    if constrained_value is None:
                        return False, f"cannot resolve {name}.{range_target} for {text}"
                    if int(constrained_value) < lo or int(constrained_value) > hi:
                        return False, f"{name}.{range_target}={constrained_value} violates {text}"

        for edge in self.edges:
            rel = str(edge.get("rel") or "").lower()
            if rel != "length_of":
                continue
            src = _node_key(edge.get("src"))
            dst = _node_key(edge.get("dst"))
            src_node = self.nodes.get(src)
            dst_node = self.nodes.get(dst)
            if not src_node or not dst_node:
                continue
            src_name = _node_name(src_node, src)
            dst_name = _node_name(dst_node, dst)
            if src not in materialized_ids or dst not in materialized_ids:
                continue
            expected = _eval_formula_with_values(str(edge.get("formula") or ""), values_by_id)
            dst_size = self._resolve_size_bits(dst, values_by_id)
            if expected is not None and dst_size is not None and int(expected) != int(dst_size):
                return False, f"{edge.get('rel')} {src_name}->{dst_name} expected {expected} bits"

        return True, ""

    def _fixed_fields_under(self, start_id: str) -> Tuple[str, ...]:
        out = []
        for node_id in self._descendants(start_id):
            node = self.nodes.get(node_id)
            if not node or _node_type(node) not in self.FIELD_TYPES:
                continue
            if self._resolve_bit_start(node_id) is None:
                continue
            if self._resolve_size_bits(node_id) is None and "val(" not in str(node.get("size_bits") or ""):
                continue
            out.append(node_id)
        return tuple(self._sorted_fields(out))

    def _descendants(self, start_id: str) -> Tuple[str, ...]:
        out = []
        stack = [start_id]
        seen = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            out.append(node_id)
            stack.extend(reversed(self.children.get(node_id, ())))
        return tuple(out)

    def _ancestors(self, start_id: str) -> Tuple[str, ...]:
        out = []
        seen = set()
        current: Optional[str] = start_id
        while current is not None and current not in seen:
            seen.add(current)
            out.append(current)
            node = self.nodes.get(current)
            parent = node.get("parent_id") if node else None
            current = None if parent is None else _node_key(parent)
        return tuple(out)

    def _sorted_fields(self, node_ids: Iterable[str]) -> Tuple[str, ...]:
        unique = tuple(dict.fromkeys(node_ids))
        return tuple(
            sorted(
                unique,
                key=lambda node_id: (
                    self._resolve_bit_start(node_id) if self._resolve_bit_start(node_id) is not None else 10**12,
                    node_id,
                ),
            )
        )

    def _path_forced_values(self, path: PkgParsingPath) -> Dict[str, int]:
        forced: Dict[str, int] = {}
        path_set = set(path.node_ids)
        if path.variant_id:
            variant = self.nodes.get(path.variant_id)
            if variant:
                selector_value = _selector_value_from_variant_constraints(variant)
                selector_id = _selector_id_from_variant_constraints(variant)
                if selector_value is not None and selector_id is not None:
                    forced[selector_id] = selector_value

        selected_scope = (
            set(self._descendants(path.variant_id)) | set(self._ancestors(path.variant_id))
            if path.variant_id
            else set()
        )
        for edge in self.edges:
            rel = str(edge.get("rel") or "").lower()
            dst = _node_key(edge.get("dst"))
            if path.variant_id and dst not in selected_scope:
                continue
            src = _node_key(edge.get("src"))
            if src not in path_set:
                continue
            if rel == "repeat_count":
                forced[src] = max(1, forced.get(src, 0))
                continue
            if rel != "condition_on":
                continue
            selector = self.nodes.get(src, {})
            bit_length = self._resolve_size_bits(src) or 8
            value = _selector_value_from_formula(
                str(edge.get("formula") or ""),
                selector_id=src,
                bit_length=bit_length,
                selector_node=selector,
            )
            if value is not None:
                forced[src] = value
        return forced

    def _choose_value(self, node: Mapping[str, Any]) -> int:
        upper = getattr(self, "_value_side", "lower") == "upper"
        for raw in node.get("constraints") or []:
            enum_values = _parse_enum_constraint(str(raw))
            if enum_values:
                return int(max(enum_values) if upper else min(enum_values))
        for raw in node.get("constraints") or []:
            text = str(raw)
            if _range_constraint_target(text) != "value":
                continue
            bounds = _parse_range_constraint(text)
            if bounds is not None:
                return int(bounds[1] if upper else bounds[0])
            lower = _parse_range_lower_bound(text)
            if lower is not None:
                return int(lower)
        return 0

    def _activated_constraints(
        self,
        materialized_ids: set[str],
        *,
        values_by_id: Mapping[str, int],
        path: PkgParsingPath,
    ) -> list[str]:
        out: list[str] = []
        for node_id in self._sorted_fields(materialized_ids):
            node = self.nodes.get(node_id)
            if not node:
                continue
            name = _node_name(node, node_id)
            for raw in node.get("constraints") or []:
                text = str(raw).strip()
                if (
                    _parse_enum_constraint(text)
                    or _parse_range_constraint(text)
                    or parse_numeric_calculated_upper_range_constraint_text(text) is not None
                    or text.lower().startswith("formula:")
                ):
                    canonical = pkg_constraint_id_for_node(node_id, name, text)
                    self._legacy_by_canonical[canonical] = constraint_id_for_node(name, text)
                    out.append(canonical)
        active_structure_ids = set(materialized_ids)
        for node_id in tuple(materialized_ids):
            active_structure_ids.update(self._ancestors(node_id))
        for branch_id in path.covered_branch_ids:
            active_structure_ids.update(self._ancestors(branch_id))
            active_structure_ids.update(self._descendants(branch_id))

        for edge in self.edges:
            rel = str(edge.get("rel") or "").lower()
            if rel not in {"length_of", "condition_on"}:
                continue
            src = _node_key(edge.get("src"))
            dst = _node_key(edge.get("dst"))
            if rel == "length_of" and path.variant_id:
                path_message_type = self._message_type_for_variant_scope(path.variant_id)
                edge_message_type = (
                    str(edge.get("message_type") or "").strip().lower()
                    or "bidirectional"
                )
                if (
                    path_message_type is not None
                    and edge_message_type not in {"bidirectional", path_message_type}
                ):
                    continue
            if rel == "length_of" and not (src in materialized_ids and dst in active_structure_ids):
                continue
            if rel == "condition_on":
                covered_branches = set(path.covered_branch_ids or ((path.variant_id,) if path.variant_id else ()))
                active_targets = {
                    target
                    for branch_id in covered_branches
                    for target in self._ancestors(branch_id)
                }
                if dst not in active_targets:
                    continue
                if not _formula_is_true(str(edge.get("formula") or ""), values_by_id):
                    continue
            src_name = _node_name(self.nodes.get(src, {}), src)
            dst_name = _node_name(self.nodes.get(dst, {}), dst)
            canonical = pkg_constraint_id_for_edge(
                rel,
                src,
                src_name,
                dst,
                dst_name,
                edge.get("formula"),
                edge.get("message_type"),
            )
            self._legacy_by_canonical[canonical] = constraint_id_for_edge(
                rel,
                src_name,
                dst_name,
                edge.get("formula"),
            )
            out.append(canonical)
        return list(dict.fromkeys(out))

    def _resolve_bit_start(self, node_id: str, values_by_id: Optional[Mapping[str, int]] = None) -> Optional[int]:
        return self._resolve_attr(node_id, "bit_start", set(), values_by_id or {})

    def _resolve_size_bits(self, node_id: str, values_by_id: Optional[Mapping[str, int]] = None) -> Optional[int]:
        return self._resolve_attr(node_id, "size_bits", set(), values_by_id or {})

    def _resolve_attr(
        self,
        node_id: str,
        attr: str,
        seen: set[tuple[str, str]],
        values_by_id: Mapping[str, int],
    ) -> Optional[int]:
        key = (node_id, attr)
        if key in seen:
            return None
        seen.add(key)
        node = self.nodes.get(node_id)
        if not node:
            return None
        raw = node.get(attr)
        if isinstance(raw, int):
            return int(raw)
        text = str(raw or "").strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        if "val(" in text and not values_by_id:
            return None

        def repl(match: re.Match[str]) -> str:
            ref_id = _node_key(match.group(1))
            ref_attr = match.group(2)
            value = self._resolve_attr(ref_id, ref_attr, set(seen), values_by_id)
            if value is None:
                raise ValueError(f"unresolved {ref_id}.{ref_attr}")
            return str(value)

        try:
            expr = re.sub(r"([A-Za-z0-9_]+)\.(bit_start|size_bits)", repl, text)
            expr = _replace_val_refs(expr, values_by_id)
            return _eval_int_expression(expr)
        except Exception:
            return None


def _node_key(raw: Any) -> str:
    return str(raw)


def _stable_node_id_key(raw: Any) -> Tuple[int, Any]:
    text = _node_key(raw)
    try:
        return (0, int(text))
    except Exception:
        return (1, text)


def _json_node_id(raw: Any) -> Any:
    text = _node_key(raw)
    try:
        return int(text)
    except Exception:
        return text


def _coerce_optional_int(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _normalize_contract_text(raw: Any) -> str:
    text = str(raw or "").strip()
    text = text.replace("：", ":").replace("≤", "<=").replace("≥", ">=").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _node_type(node: Mapping[str, Any]) -> str:
    return str(node.get("node_type") or "").strip().lower()


def _node_name(node: Mapping[str, Any], node_id: str) -> str:
    return str(node.get("name") or f"node_{node_id}")


def _node_endianness(node: Mapping[str, Any]) -> str:
    enc = node.get("encoding") if isinstance(node.get("encoding"), Mapping) else {}
    return str(node.get("endianness") or node.get("byte_order") or enc.get("endianness") or "big")


def _parse_int_token(raw: str) -> int:
    text = str(raw).strip()
    try:
        return int(text, 0)
    except ValueError:
        return int(text, 10)


def _parse_enum_constraint(raw: str) -> Tuple[int, ...]:
    parsed = parse_numeric_enum_constraint_text(str(raw or "").strip())
    return tuple(parsed or ())


def _parse_range_constraint(raw: str) -> Optional[Tuple[int, int]]:
    return parse_numeric_range_constraint_text(str(raw or "").strip())


def _range_constraint_target(raw: str) -> str:
    text = str(raw or "").replace("：", ":").lower()
    if re.search(r"\bsize_bits\b", text):
        return "size_bits"
    if re.search(r"\bbit_start\b", text):
        return "bit_start"
    return "value"


def _parse_range_lower_bound(raw: str) -> Optional[int]:
    text = str(raw or "").strip()
    if not text.lower().startswith("range:"):
        return None
    match = re.search(r"range:\s*(0x[0-9a-fA-F]+|-?\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return _parse_int_token(match.group(1))
    except Exception:
        return None


def _selector_value_from_formula(
    formula: str,
    *,
    selector_id: Optional[str] = None,
    bit_length: int = 8,
    selector_node: Optional[Mapping[str, Any]] = None,
) -> Optional[int]:
    text = str(formula or "")
    if selector_id is None:
        ids = tuple(dict.fromkeys(re.findall(r"val\(([^)]+)\)", text)))
        if len(ids) != 1:
            return None
        selector_id = _node_key(ids[0])

    width = max(1, int(bit_length or 8))
    exhaustive_width = min(width, 16)
    exhaustive_hi = (1 << exhaustive_width) - 1
    candidates: list[int] = []
    node = selector_node or {}
    match = re.search(
        rf"val\(\s*{re.escape(str(selector_id))}\s*\)\s*==\s*(0x[0-9a-fA-F]+|-?\d+)",
        text,
    )
    if match:
        candidates.append(_parse_int_token(match.group(1)))
    match = re.search(
        rf"val\(\s*{re.escape(str(selector_id))}\s*\)\s+in\s+\[?\s*(0x[0-9a-fA-F]+|-?\d+)",
        text,
    )
    if match:
        candidates.append(_parse_int_token(match.group(1)))
    for raw in node.get("constraints") or []:
        enum_values = _parse_enum_constraint(str(raw))
        if enum_values:
            candidates.extend(int(value) for value in enum_values)
        bounds = (
            _parse_range_constraint(str(raw))
            if _range_constraint_target(str(raw)) == "value"
            else None
        )
        if bounds is not None:
            lo, hi = bounds
            candidates.extend((int(lo), int(hi)))
    formula_constants = re.sub(r"val\([^)]+\)", "value", text)
    for token in re.findall(r"(?<![A-Za-z0-9_])(0x[0-9a-fA-F]+|-?\d+)", formula_constants):
        try:
            value = _parse_int_token(token)
        except Exception:
            continue
        candidates.extend((value, value - 1, value + 1))
    candidates.extend(range(exhaustive_hi + 1))
    for candidate in dict.fromkeys(
        value
        for value in candidates
        if int(value) >= 0 and int(value).bit_length() <= width
    ):
        candidate_int = int(candidate)
        allowed = True
        for raw in node.get("constraints") or []:
            enum_values = _parse_enum_constraint(str(raw))
            if enum_values and candidate_int not in enum_values:
                allowed = False
                break
            bounds = (
                _parse_range_constraint(str(raw))
                if _range_constraint_target(str(raw)) == "value"
                else None
            )
            if bounds is not None and not (int(bounds[0]) <= candidate_int <= int(bounds[1])):
                allowed = False
                break
        if allowed and _formula_is_true(text, {str(selector_id): candidate_int}):
            return candidate_int
    return None


def _selector_value_from_variant_constraints(variant: Mapping[str, Any]) -> Optional[int]:
    for raw in variant.get("constraints") or []:
        payload = _constraint_ir_payload(str(raw))
        if not payload:
            continue
        expr = str(payload.get("expr") or "")
        match = re.search(r"value\(s\)\s+(0x[0-9a-fA-F]+|-?\d+)", expr, flags=re.IGNORECASE)
        if match:
            return _parse_int_token(match.group(1))
    return None


def _selector_id_from_variant_constraints(variant: Mapping[str, Any]) -> Optional[str]:
    variant_id = _node_key(variant.get("node_id"))
    for raw in variant.get("constraints") or []:
        payload = _constraint_ir_payload(str(raw))
        if not payload:
            continue
        field_ids = payload.get("field_ids") if isinstance(payload.get("field_ids"), Sequence) else []
        normalized = [_node_key(item) for item in field_ids]
        if len(normalized) >= 2 and normalized[0] == variant_id:
            return normalized[1]
        if len(normalized) >= 2:
            return normalized[0]
    return None


def _constraint_ir_payload(raw: str) -> Optional[dict[str, Any]]:
    text = str(raw or "").strip()
    if not text.lower().startswith("constraint_ir:"):
        return None
    try:
        payload = json.loads(text.split(":", 1)[1].strip())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _replace_val_refs(expr: str, values_by_id: Mapping[str, int]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = _node_key(match.group(1))
        if key not in values_by_id:
            raise ValueError(f"missing val({key})")
        return str(int(values_by_id[key]))

    return re.sub(r"val\(([^)]+)\)", repl, str(expr or ""))


def _eval_formula_with_values(formula: str, values_by_id: Mapping[str, int]) -> Optional[int]:
    text = str(formula or "").strip()
    if not text:
        return None
    try:
        return _eval_int_expression(_replace_val_refs(text, values_by_id))
    except Exception:
        return None


def _normalize_boolean_formula(expr: str) -> str:
    text = str(expr or "")
    text = text.replace("&&", " and ").replace("||", " or ")
    return re.sub(r"!(?!=)", " not ", text)


def _formula_is_true(formula: str, values_by_id: Mapping[str, int]) -> bool:
    return _try_formula_truth(formula, values_by_id) is True


def _try_formula_truth(formula: str, values_by_id: Mapping[str, int]) -> Optional[bool]:
    text = str(formula or "").strip()
    if not text:
        return None
    try:
        replaced = _replace_val_refs(_normalize_boolean_formula(text), values_by_id)
        parsed = ast.parse(replaced, mode="eval")
        return bool(_eval_ast_scalar(parsed.body))
    except Exception:
        return None


def _solve_formula_constraint(
    raw_constraint: str,
    *,
    target_id: str,
    values_by_id: Mapping[str, int],
) -> Optional[int]:
    text = str(raw_constraint or "").strip()
    if not text.lower().startswith("formula:"):
        return None
    formula = text.split(":", 1)[1].strip()
    match = re.match(rf"val\(\s*{re.escape(str(target_id))}\s*\)\s*==\s*(?P<rhs>.+)$", formula)
    if not match:
        return None
    return _eval_formula_with_values(match.group("rhs"), values_by_id)


def _solve_formula_for_value(
    formula: str,
    *,
    target_id: str,
    target_result: int,
    values_by_id: Mapping[str, int],
) -> Optional[int]:
    """Invert a one-variable affine length formula using two safe evaluations."""

    text = str(formula or "").strip()
    if not text or not re.search(rf"val\(\s*{re.escape(str(target_id))}\s*\)", text):
        return None

    def evaluate(candidate: int) -> Optional[int]:
        values = dict(values_by_id)
        values[str(target_id)] = int(candidate)
        return _eval_formula_with_values(text, values)

    at_zero = evaluate(0)
    at_one = evaluate(1)
    if at_zero is None or at_one is None:
        return None
    coefficient = int(at_one) - int(at_zero)
    if coefficient == 0:
        return None
    numerator = int(target_result) - int(at_zero)
    if numerator % coefficient:
        return None
    value = numerator // coefficient
    return int(value) if value >= 0 else None


def _extract_field_value(raw: bytes, boundary: FieldBoundary, endianness: str) -> int:
    start_bit = int(boundary.start_bit)
    bit_length = int(boundary.bit_length)
    if bit_length <= 0 or start_bit < 0 or start_bit + bit_length > len(raw) * 8:
        return 0
    data_int = int.from_bytes(raw, byteorder="big", signed=False)
    shift = len(raw) * 8 - (start_bit + bit_length)
    value = (data_int >> shift) & ((1 << bit_length) - 1)
    if str(endianness or "big").lower().startswith("little") and bit_length % 8 == 0:
        byte_len = bit_length // 8
        value = int.from_bytes(value.to_bytes(byte_len, "big", signed=False)[::-1], "big", signed=False)
    return int(value)


def _encode_field_value(value: int, bit_length: int, endianness: str) -> int:
    value = int(value) & ((1 << int(bit_length)) - 1)
    if str(endianness or "big").lower().startswith("little") and int(bit_length) % 8 == 0:
        byte_len = int(bit_length) // 8
        return int.from_bytes(value.to_bytes(byte_len, "big", signed=False)[::-1], "big", signed=False)
    return value


def _patch_field_value(raw: bytes, start_bit: int, bit_length: int, value: int, endianness: str) -> bytes:
    total_bits = len(raw) * 8
    if bit_length <= 0 or start_bit < 0 or start_bit + bit_length > total_bits:
        raise ValueError("field boundary is outside packet bytes")
    encoded = _encode_field_value(value, bit_length, endianness)
    data_int = int.from_bytes(raw, byteorder="big", signed=False)
    shift = total_bits - (start_bit + bit_length)
    mask = ((1 << bit_length) - 1) << shift
    new_int = (data_int & ~mask) | ((encoded & ((1 << bit_length) - 1)) << shift)
    return new_int.to_bytes(len(raw), byteorder="big", signed=False)


def _eval_int_expression(expr: str) -> int:
    parsed = ast.parse(expr, mode="eval")
    value = _eval_ast_int(parsed.body)
    if not isinstance(value, int):
        raise ValueError(f"expression is not integral: {expr}")
    return int(value)


def _eval_ast_int(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast_int(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div)):
        left = _eval_ast_int(node.left)
        right = _eval_ast_int(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ValueError("division by zero")
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if left % right != 0:
            raise ValueError("non-integral division")
        return left // right
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _eval_ast_scalar(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_eval_ast_scalar(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.Set):
            return set(values)
        return values
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast_scalar(node.operand)
        if isinstance(node.op, ast.Not):
            return not bool(value)
        if isinstance(node.op, ast.UAdd):
            return +int(value)
        if isinstance(node.op, ast.USub):
            return -int(value)
        if isinstance(node.op, ast.Invert):
            return ~int(value)
    if isinstance(node, ast.BoolOp):
        values = [_eval_ast_scalar(value) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(value) for value in values)
        if isinstance(node.op, ast.Or):
            return any(bool(value) for value in values)
    if isinstance(node, ast.BinOp):
        left = int(_eval_ast_scalar(node.left))
        right = int(_eval_ast_scalar(node.right))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if right == 0:
                raise ValueError("division by zero")
            if isinstance(node.op, ast.Div) and left % right:
                raise ValueError("non-integral division")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ValueError("modulo by zero")
            return left % right
        if isinstance(node.op, ast.BitAnd):
            return left & right
        if isinstance(node.op, ast.BitOr):
            return left | right
        if isinstance(node.op, ast.BitXor):
            return left ^ right
        if isinstance(node.op, ast.LShift):
            return left << right
        if isinstance(node.op, ast.RShift):
            return left >> right
    if isinstance(node, ast.Compare):
        left = _eval_ast_scalar(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast_scalar(comparator)
            if isinstance(op, ast.Eq):
                result = left == right
            elif isinstance(op, ast.NotEq):
                result = left != right
            elif isinstance(op, ast.Lt):
                result = left < right
            elif isinstance(op, ast.LtE):
                result = left <= right
            elif isinstance(op, ast.Gt):
                result = left > right
            elif isinstance(op, ast.GtE):
                result = left >= right
            elif isinstance(op, ast.In):
                result = left in right
            elif isinstance(op, ast.NotIn):
                result = left not in right
            else:
                raise ValueError(f"unsupported comparison: {type(op).__name__}")
            if not result:
                return False
            left = right
        return True
    raise ValueError(f"unsupported scalar expression node: {type(node).__name__}")
