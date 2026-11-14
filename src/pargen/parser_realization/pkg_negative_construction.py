"""Bounded tree-only attempts to isolate a target without touching a parser.

Candidates are selected by reference-tree qualification, never host performance.
Unsuccessful attempts retain the original negative and its coverage denominator.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from .pkg_negative_qualification import _ProjectedPath, qualify_negative

POLICY = "pkg_targeted_isolation_construction_v1"
MAX_PREFIX_CANDIDATES = 32
MAX_LENGTH_PASSES = 8
MAX_PACKET_BYTES = 4096


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _values(projected):
    return {node: value for node in projected.by_node
            if (value := projected.value(node)) is not None}


def _balance_lengths(ctx, sample, raw, targets, replay):
    """Solve enclosing length equations; all writes have checked wire widths.

    A proposed inverse is only a candidate: exact re-evaluation and the complete
    isolation audit below must succeed. No approximate inverse is proof.
    """
    from .pkg_seed_generator import _node_endianness, _patch_field_value, _solve_formula_for_value
    spans = list(replay[1].node_spans)
    edits, seen = [], set()
    for _ in range(MAX_LENGTH_PASSES):
        if raw in seen:
            return None
        seen.add(raw)
        projected = _ProjectedPath(ctx, sample, raw, spans)
        changed = False
        for cid in sample.activated_constraints:
            spec = ctx._constraint_specs_by_canonical.get(str(cid), {})
            if cid in targets or spec.get("kind") != "length_of":
                continue
            src, dst = str(spec["src"]), str(spec["dst"])
            extent = projected.wire_extent(dst)
            expected = projected.formula(spec["formula"])
            if extent is None or expected is None or expected == extent[1]:
                continue
            start, size = projected.attr(src, "bit_start"), projected.attr(src, "size_bits")
            if (start is None or size is None or not 0 < size <= 64
                    or start < 0 or start+size > len(raw)*8):
                return None
            values = _values(projected)
            value = _solve_formula_for_value(spec["formula"], target_id=src,
                target_result=extent[1], values_by_id=values)
            if value is None or not 0 <= value < 1 << size:
                return None
            values[src] = value
            if not ctx._node_value_satisfies_constraints(src, value, values):
                return None
            patched = _patch_field_value(raw, start, size, value, _node_endianness(ctx.nodes[src]))
            candidate = _ProjectedPath(ctx, sample, patched, spans)
            if candidate.formula(spec["formula"]) != extent[1]:
                return None
            edits.append(dict(constraint_id=str(cid), source_node_id=src,
                bit_start=start, size_bits=size, before=projected.value(src), after=value,
                expected_bits=extent[1]))
            raw = patched
            projected = candidate
            changed = True
        if not changed:
            return raw, edits
    return None


def _terminal_resize(ctx, sample, negative, replay):
    """Keep an opaque terminal body consistent with a mutated scalar count.

    Only a unique, unconstrained leaf at capture end can be resized. Containers,
    constrained bodies, trailers and large allocations are deliberately refused.
    """
    targets = negative.expected_violated_constraints
    if len(targets) != 1:
        return None
    spec = ctx._constraint_specs_by_canonical.get(targets[0], {})
    if spec.get("kind") not in {"enum", "range", "formula"}:
        return None
    source = str(spec.get("node_id"))
    projected = _ProjectedPath(ctx, sample, negative.raw_bytes, replay[1].node_spans)
    relations = [v for cid, v in ctx._constraint_specs_by_canonical.items()
                 if cid in sample.activated_constraints and v.get("kind") == "length_of"
                 and str(v.get("src")) == source]
    if len(relations) != 1:
        return None
    relation = relations[0]
    dst = str(relation["dst"])
    node, span = ctx.nodes[dst], projected.span(dst)
    if (span is None or ctx.children.get(dst) or node.get("constraints")
            or int(span["bit_start"])+int(span["wire_size_bits"]) != len(sample.raw_bytes)*8):
        return None
    start, expected = projected.attr(dst, "bit_start"), projected.formula(relation["formula"])
    if start is None or expected is None or start < 0 or expected < 0 or start % 8 or expected % 8:
        return None
    end = (start+expected)//8
    if not start//8 <= len(negative.raw_bytes) or end > MAX_PACKET_BYTES:
        return None
    raw = negative.raw_bytes[:end] + b"\0" * max(0, end-len(negative.raw_bytes))
    if raw == negative.raw_bytes:
        return None
    return raw, dict(kind="resize_unconstrained_terminal_leaf", node_id=dst,
        before_bytes=len(negative.raw_bytes), after_bytes=len(raw), fill_byte=0)


def _condition_prefixes(ctx, sample, negative, replay):
    cid = negative.expected_violated_constraints[0]
    edges = [e for e in ctx.edges if str(e.get("rel", "")).lower() == "condition_on"
             and ctx._condition_constraint_id(e) == cid]
    if len(edges) != 1:
        return [], 0
    dst, src = str(edges[0]["dst"]), str(edges[0]["src"])
    spans = replay[1].node_spans
    projected = _ProjectedPath(ctx, sample, sample.raw_bytes, spans)
    selector = projected.span(src)
    if selector is None:
        return [], 0
    selector_end = int(selector["bit_start"])+int(selector["wire_size_bits"])
    descendants = set(ctx._descendants(dst))
    cuts = set()
    for span in spans:
        node = str(span["node_id"])
        if node not in descendants or str(ctx.nodes[node].get("node_type", "")).lower() not in ctx.FIELD_TYPES:
            continue
        start, size = int(span["bit_start"]), int(span["wire_size_bits"])
        for cut in (start, start+size-8):
            if selector_end <= cut < len(sample.raw_bytes)*8 and cut > 0 and cut % 8 == 0:
                cuts.add(cut//8)
    # Later cuts preserve more prerequisite fields. The budget is fixed before
    # inspecting any parser output; a cap is recorded rather than silently lifted.
    return ([(sample.raw_bytes[:cut], dict(kind="selected_body_suffix_truncation", cut_bytes=cut))
             for cut in sorted(cuts, reverse=True)[:MAX_PREFIX_CANDIDATES]], len(cuts))


def isolate_selected_negatives(ctx, positives, negatives, replays):
    """One bounded search per selected target, with a loss-accounted receipt."""
    output, qualifications, receipts = [], [], []
    for original in negatives:
        sample, replay = positives[original.base_packet_id], replays[original.base_packet_id]
        current = original
        qualification = qualify_negative(ctx, sample, current, replay=replay)
        original_status = qualification["status"]
        attempts = []
        proposals = []
        available = 0
        if not qualification["independent_target_test"] and replay[0]:
            if original.expected_violated_constraints[0].startswith("edge:condition_on:"):
                proposals, available = _condition_prefixes(ctx, sample, original, replay)
            else:
                resized = _terminal_resize(ctx, sample, original, replay)
                if resized is not None:
                    proposals = [resized]
                    available = 1
        for raw, mutation in proposals:
            if len(raw) > MAX_PACKET_BYTES:
                attempts.append(dict(mutation=mutation, status="candidate_exceeds_packet_budget"))
                continue
            balanced = _balance_lengths(ctx, sample, raw, set(original.expected_violated_constraints), replay)
            if balanced is None:
                attempts.append(dict(mutation=mutation, status="length_rebalance_unverified"))
                continue
            raw, edits = balanced
            if raw == original.raw_bytes:
                attempts.append(dict(mutation=mutation, status="unchanged_candidate"))
                continue
            receipt = dict(policy=POLICY, original_payload_sha256=_digest(original.raw_bytes),
                mutation=mutation, length_updates=edits)
            candidate = replace(original, raw_bytes=raw,
                note="isolated_construction="+json.dumps(receipt, sort_keys=True, separators=(",", ":")))
            q = qualify_negative(ctx, sample, candidate, replay=replay)
            attempts.append(dict(mutation=mutation, status=q["status"], payload_sha256=_digest(raw),
                target_violation_proven=q["target_violation_proven"],
                coincident_constraints=q["coincident_constraints"], unresolved_constraints=q["unresolved_constraints"]))
            if q["independent_target_test"]:
                cid = original.expected_violated_constraints[0]
                if cid.startswith("edge:condition_on:"):
                    import copy
                    from pargen.protocol_knowledge.message_validation.interpreter import DynamicTreeInterpreter
                    edge = next(e for e in ctx.edges if str(e.get("rel", "")).lower() == "condition_on"
                                and ctx._condition_constraint_id(e) == cid)
                    ok, _, error = DynamicTreeInterpreter(copy.deepcopy(ctx.tree),
                        target_message_type=ctx._target_message_type_for_sample(sample)).parse(raw)
                    rejected = not ok and ctx._descendant_oob_error_node(error, set(ctx._descendants(str(edge["dst"])))) is not None
                else:
                    rejected = ctx._reference_tree_rejects_for_target(sample, raw, cid)
                attempts[-1]["reference_target_reproduced"] = rejected
                if rejected:
                    current, qualification = candidate, q
                    break
        receipts.append(dict(policy=POLICY, target_constraints=list(original.expected_violated_constraints),
            base_packet_id=original.base_packet_id, original_payload_sha256=_digest(original.raw_bytes),
            original_status=original_status, selected_payload_sha256=_digest(current.raw_bytes),
            selected_status=qualification["status"], replaced=current.raw_bytes != original.raw_bytes,
            budgets=dict(prefix_candidates=MAX_PREFIX_CANDIDATES, length_passes=MAX_LENGTH_PASSES,
                         max_packet_bytes=MAX_PACKET_BYTES),
            candidates_available=available, candidates_truncated_by_budget=max(0, available-len(proposals)),
            search_stop_reason=("original_already_isolated" if original_status == "isolated" else
                "isolated_candidate_found" if current.raw_bytes != original.raw_bytes else
                "candidate_budget_exhausted" if available > len(proposals) else
                "bounded_candidates_exhausted" if proposals else "no_supported_construction"),
            attempts=attempts))
        output.append(current)
        qualifications.append(qualification)
    return tuple(output), tuple(qualifications), tuple(receipts)
