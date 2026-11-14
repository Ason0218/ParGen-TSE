"""Conservative, protocol-neutral isolation audit of tree-generated mutations.

A fail-fast replay proves a target violation, not that other constraints hold.
This audit projects the positive replay's occurrences onto the mutated packet
using explicit tree expressions and checks its activated obligations. Unsupported
or ambiguous layouts remain unverified, not independent-coverage credit.
"""
from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import replace

POLICY = "pkg_negative_isolation_v2"


class _ProjectedPath:
    """Re-evaluate explicit layout expressions, never assume shifted offsets.

    Only unique occurrences are supported. A positive span anchors an expression
    only if that expression agrees with the replay. Unknown expressions, cycles,
    implicit layouts after resizing, and repeated occurrences fail closed.
    """

    def __init__(self, ctx, sample, raw, spans):
        self.ctx, self.sample, self.raw = ctx, sample, bytes(raw)
        self.by_node = {}
        for span in spans:
            self.by_node.setdefault(str(span.get("node_id")), []).append(span)
        self.cache, self.busy = {}, set()

    def span(self, node):
        spans = self.by_node.get(str(node), ())
        return spans[0] if len(spans) == 1 else None

    def value(self, node, *, original=False):
        from .conformance_models.models import FieldBoundary
        from .pkg_seed_generator import _extract_field_value, _node_endianness
        node = str(node)
        span = self.span(node)
        if span is None or span.get("value") is None:
            return None
        if original:
            try:
                return int(span["value"])
            except (TypeError, ValueError):
                return None
        start, size = self.attr(node, "bit_start"), self.attr(node, "size_bits")
        if start is None or size is None or not 0 < size <= 64 or start < 0 or start + size > len(self.raw)*8:
            return None
        return _extract_field_value(self.raw, FieldBoundary(start, size),
                                    _node_endianness(self.ctx.nodes[node]))

    def formula(self, text, *, original=False, truth=False):
        from .pkg_seed_generator import _eval_formula_with_values, _try_formula_truth
        values = {}
        for node in re.findall(r"val\(\s*([^)]+?)\s*\)", str(text)):
            value = self.value(node, original=original)
            if value is None:
                return None
            values[node] = value
        return (_try_formula_truth if truth else _eval_formula_with_values)(text, values)

    def attr(self, node, attr, *, original=False):
        from .pkg_seed_generator import _eval_int_expression
        node = str(node)
        key = (node, attr, original)
        if key in self.cache:
            return self.cache[key]
        if key in self.busy:
            return None
        self.busy.add(key)
        result = None
        try:
            span = self.span(node)
            if span is None:
                return None
            observed = int(span["bit_start" if attr == "bit_start" else "wire_size_bits"])
            if original:
                result = observed
                return result
            expr = self.ctx.nodes.get(node, {}).get(attr)
            if str(expr).strip().lower() == "variable" and attr == "size_bits":
                children = [ch for ch in self.ctx.children.get(node, ()) if ch in self.by_node]
                start = self.attr(node, "bit_start")
                if children and start is not None:
                    ends = [(self.attr(ch, "bit_start"), self.attr(ch, "size_bits")) for ch in children]
                    if all(a is not None and b is not None for a, b in ends):
                        result = max(a+b for a, b in ends) - start
                elif not self.ctx.children.get(node) and observed == 0:
                    result = 0
                return result
            if expr is None:
                return None

            def evaluate(before):
                def ref(match):
                    value = self.attr(match[1], match[2], original=before)
                    if value is None:
                        raise ValueError("unresolved layout reference")
                    return str(value)
                text = re.sub(r"([A-Za-z0-9_]+)\.(bit_start|size_bits)", ref, str(expr))
                def val(match):
                    value = self.value(match[1].strip(), original=before)
                    if value is None:
                        raise ValueError("unresolved layout value")
                    return str(value)
                return _eval_int_expression(re.sub(r"val\(([^)]+)\)", val, text))

            if evaluate(True) == observed:
                result = evaluate(False)
                if result < 0:
                    result = None
            return result
        except (TypeError, ValueError, SyntaxError, KeyError, OverflowError):
            return None
        finally:
            self.busy.remove(key)
            self.cache[key] = result

    def wire_extent(self, node):
        """Independent physical boundary: terminal capture or stable next extent.

        A length formula cannot certify itself by defining its own destination
        size. Terminal destinations use captured bytes; non-terminal ones require
        an unchanged measured size and offset. Overlapping shifted fields remain
        unresolved, rather than borrowing bytes from a following field.
        """
        span = self.span(node)
        start = self.attr(node, "bit_start")
        if span is None or start is None or start < 0 or start > len(self.raw)*8:
            return None
        old_start, old_size = int(span["bit_start"]), int(span["wire_size_bits"])
        if old_start + old_size == len(self.sample.raw_bytes)*8:
            return start, len(self.raw)*8-start
        size = self.attr(node, "size_bits")
        if start != old_start or size != old_size:
            return None
        return start, min(old_size, max(0, len(self.raw)*8-start))


def qualify_negative(ctx, sample, negative, *, replay=None, _check_source=True):
    from pargen.protocol_knowledge.message_validation.interpreter import DynamicTreeInterpreter
    from .pkg_seed_generator import (
        _node_key, _parse_enum_constraint, _parse_range_constraint,
        _range_constraint_target,
    )

    if replay is None:
        replay = DynamicTreeInterpreter(copy.deepcopy(ctx.tree),
            target_message_type=ctx._target_message_type_for_sample(sample)).parse(sample.raw_bytes)
    ok, context, _ = replay
    raw = bytes(negative.raw_bytes)
    targets = set(map(str, negative.expected_violated_constraints))
    active = set(map(str, sample.activated_constraints)) | targets
    spans = list(getattr(context, "node_spans", ()) or ()) if ok else []
    projected = _ProjectedPath(ctx, sample, raw, spans)
    by_node = projected.by_node
    # Reusing the positive path is unsafe if a previously inactive route becomes
    # selected. Do not silently omit that route's obligations from the audit.
    route_changed = False
    route_unknown = False
    changed_edges, after_routes = [], []
    message_type = ctx._target_message_type_for_sample(sample)
    for edge in ctx.edges:
        if str(edge.get("rel", "")).lower() != "condition_on":
            continue
        dst = _node_key(edge.get("dst"))
        direction = str(ctx.nodes.get(dst, {}).get("message_type") or "bidirectional").lower()
        if message_type and direction not in {message_type, "bidirectional"}:
            continue
        expr = str(edge.get("formula") or "")
        before = projected.formula(expr, original=True, truth=True)
        after = projected.formula(expr, truth=True)
        if str(edge.get("src")) in by_node and (before is None or after is None):
            route_unknown = True
        after_routes.append((str(edge.get("src")), after))
        if before is not None and after is not None and before != after:
            route_changed = True
            changed_edges.append((edge, before, after))
    empty_routes_removed = set()
    if route_changed and all(before is True and after is False
                             and not ctx.children.get(str(edge["dst"]))
                             and projected.span(str(edge["dst"])) is not None
                             and projected.attr(str(edge["dst"]), "size_bits") == 0
                             and all(truth is False for src, truth in after_routes
                                     if src == str(edge["src"]))
                             for edge, before, after in changed_edges):
        # A false implication with no on-wire body cannot break body consistency.
        # This exception is not safe for nonempty routes or newly selected routes.
        empty_routes_removed = {ctx._condition_constraint_id(edge) for edge, _, _ in changed_edges}
        route_changed = False
    checks = []
    for cid in sorted(active):
        check = dict(constraint_id=cid, status="unresolved", reason="unsupported_or_ambiguous_obligation")
        spec = ctx._constraint_specs_by_canonical.get(cid, {})
        kind = spec.get("kind")
        if not ok:
            check["reason"] = "positive_replay_failed"
        elif cid.startswith("edge:condition_on:"):
            edges = [e for e in ctx.edges if str(e.get("rel", "")).lower() == "condition_on"
                     and ctx._condition_constraint_id(e) == cid]
            if len(edges) == 1:
                edge = edges[0]
                expr = str(edge.get("formula") or "")
                selected = projected.formula(expr, truth=True)
                dst = _node_key(edge.get("dst"))
                body = [node for node in ctx._descendants(dst)
                        if node in by_node and str(ctx.nodes[node].get("node_type", "")).lower() in ctx.FIELD_TYPES]
                invalid_lengths = [key for key in active
                    if (s := ctx._constraint_specs_by_canonical.get(key, {})).get("kind") == "length_of"
                    and str(s.get("dst")) in ctx._descendants(dst)
                    and (value := projected.formula(s.get("formula", ""))) is not None and value < 0]
                if selected is True and body and not route_changed:
                    extents = [(node, projected.attr(node, "bit_start"), projected.attr(node, "size_bits"))
                               for node in body]
                    missing = [node for node, start, size in extents
                               if start is not None and size is not None and start+size > len(raw)*8]
                    if missing or all(start is not None and size is not None for _, start, size in extents):
                        check.update(status="violated" if missing else "preserved",
                            reason="selected_body_truncated" if missing else "selected_body_present",
                            missing_node_ids=sorted(missing))
                elif (selected is True and not route_changed and not ctx.children.get(dst)
                      and projected.span(dst) is not None and projected.attr(dst, "size_bits") == 0):
                    check.update(status="preserved", reason="selected_empty_body")
                elif selected is False:
                    if cid in empty_routes_removed:
                        check.update(status="preserved", reason="unselected_empty_body_implication")
                    else:
                        check["reason"] = "selector_changed_branch_not_an_independent_violation"
                if selected is True and invalid_lengths and not route_changed:
                    check.update(status="violated", reason="selected_body_length_invalid",
                                 invalid_length_constraints=sorted(invalid_lengths))
        elif kind == "length_of":
            dst = _node_key(spec.get("dst"))
            expected = projected.formula(str(spec.get("formula") or ""))
            extent = projected.wire_extent(dst)
            if expected is not None and extent is not None and not route_changed:
                start, available = extent
                check.update(status="violated" if int(expected) != available else "preserved",
                    reason="length_relation", expected_bits=int(expected), available_bits=available,
                    destination_start_bit=start,
                    source_node_id=_node_key(spec.get("src")),
                    source_value=projected.value(_node_key(spec.get("src"))))
        elif kind in {"enum", "range", "formula", "range_lower"}:
            node = _node_key(spec.get("node_id"))
            text = str(spec.get("constraint") or "")
            if projected.span(node) is not None:
                start, size = projected.attr(node, "bit_start"), projected.attr(node, "size_bits")
                value = projected.value(node)
                truth = None
                if start is not None and size is not None and start >= 0 and start + size <= len(raw) * 8:
                    if kind == "enum" and value is not None:
                        truth = value in _parse_enum_constraint(text)
                    elif kind == "range":
                        bounds = _parse_range_constraint(text)
                        measured = value if _range_constraint_target(text) == "value" else None
                        if bounds is not None and measured is not None:
                            truth = int(bounds[0]) <= measured <= int(bounds[1])
                    elif kind == "formula":
                        expr = re.sub(r"\bvalue\b", f"val({node})", text.split(":", 1)[-1].strip())
                        truth = projected.formula(expr, truth=True)
                else:
                    check["reason"] = "constrained_field_absent_after_mutation"
                if truth is not None:
                    check.update(status="preserved" if truth else "violated", reason="node_constraint",
                                 node_id=node, value=value)
        checks.append(check)

    if route_changed or route_unknown:
        # Even an individually decidable scalar target cannot receive an
        # isolation certificate when the rest of the route was not audited.
        checks.append(dict(constraint_id="audit:route_stability", status="unresolved",
                           reason="routing_expression_unverified" if route_unknown else
                                  "selector_changed_route_requires_new_path_audit"))

    leaves = [node for node in by_node if node in ctx.nodes and not ctx.children.get(node)
              and str(ctx.nodes[node].get("node_type", "")).lower() in ctx.FIELD_TYPES]
    for index, left in enumerate(leaves):
        a = projected.span(left)
        a_start, a_size = projected.attr(left, "bit_start"), projected.attr(left, "size_bits")
        if a is None or a_start is None or a_size is None or a_size <= 0:
            continue
        for right in leaves[index+1:]:
            b = projected.span(right)
            b_start, b_size = projected.attr(right, "bit_start"), projected.attr(right, "size_bits")
            if b is None or b_start is None or b_size is None or b_size <= 0:
                continue
            previously_disjoint = (int(a["bit_start"])+int(a["wire_size_bits"]) <= int(b["bit_start"])
                or int(b["bit_start"])+int(b["wire_size_bits"]) <= int(a["bit_start"]))
            if previously_disjoint and max(a_start, b_start) < min(a_start+a_size, b_start+b_size):
                checks.append(dict(constraint_id=f"audit:layout_overlap:{left}:{right}",
                    status="unresolved", reason="mutation_overlaps_previously_disjoint_fields",
                    node_ids=[left, right]))

    source_gaps = []
    if _check_source:
        source_audit = qualify_negative(ctx, sample,
            replace(negative, raw_bytes=sample.raw_bytes, expected_violated_constraints=[]),
            replay=replay, _check_source=False)
        source_gaps = [check for check in source_audit["checks"] if check["status"] != "preserved"]
        if source_gaps:
            checks.append(dict(constraint_id="audit:positive_path_validity", status="unresolved",
                reason="source_positive_has_unverified_or_violated_obligations",
                source_checks=source_gaps))

    violated = {c["constraint_id"] for c in checks if c["status"] == "violated"}
    unresolved = {c["constraint_id"] for c in checks if c["status"] == "unresolved"}
    coincident = violated - targets
    isolated = bool(targets) and targets <= violated and not coincident and not unresolved
    return dict(policy=POLICY, scope="all_activated_positive_path_obligations",
        status="isolated" if isolated else "multi_violation" if coincident else "unverified",
        independent_target_test=isolated, target_violation_proven=bool(targets) and targets <= violated,
        target_constraints=sorted(targets), known_violated_constraints=sorted(violated),
        coincident_constraints=sorted(coincident), unresolved_constraints=sorted(unresolved),
        source_payload_sha256=hashlib.sha256(sample.raw_bytes).hexdigest(),
        payload_sha256=hashlib.sha256(raw).hexdigest(), checks=checks,
        source_positive_qualified=not source_gaps if _check_source else None,
        all_tree_paths_verified=False)
