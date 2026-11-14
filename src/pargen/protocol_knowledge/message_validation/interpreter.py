"""
Dynamic Protocol Tree Interpreter for Traffic Verification.
"""

import copy
import logging
import os
import struct
import math
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import re

from ..constraint_ir import is_constraint_ir_text
from ..tree_utils import (
    desugar_implication,
    normalize_count_length_edges,
    parse_numeric_calculated_upper_range_constraint_text,
    parse_numeric_enum_constraint_text,
    parse_numeric_range_constraint_text,
)

logger = logging.getLogger(__name__)

LEAF_TYPES = {"field", "selector", "type", "length", "checksum"}
SCALAR_VALUE_NODE_TYPES = set(LEAF_TYPES)


_LITTLE_ENDIAN = {"little", "little-endian", "little_endian", "le"}


def decode_typed_value(raw: int, size_bits: int, node: Dict[str, Any]) -> int:
    """Decode the raw big-endian bit read under the node's declared byte order and signedness.

    Byte order applies to whole-byte fields wider than one byte; signed types (int8/16/32/64, "signed ...")
    are read in two's complement.  Other declared types keep the unsigned value.
    """
    value = int(raw)
    order = str(node.get("byte_order") or "").strip().lower()
    if order in _LITTLE_ENDIAN and size_bits > 8 and size_bits % 8 == 0:
        value = int.from_bytes(value.to_bytes(size_bits // 8, "big"), "little")
    dtype = str(node.get("data_type") or "").strip().lower()
    if size_bits > 0 and (dtype.startswith("int") or dtype.startswith("signed") or dtype.startswith("sint")):
        if value >= 1 << (size_bits - 1):
            value -= 1 << size_bits
    return value

@dataclass
class ConstraintViolationError(Exception):
    """Raised when a node's value violates its constraints."""
    node_id: int
    constraint_text: str
    value: Any
    kind: str = "expr"

    def __post_init__(self) -> None:  # pragma: no cover - simple dataclass init
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"Constraint violation at node {self.node_id}: {self.constraint_text} (value={self.value})"

class IncompleteStreamError(Exception):
    """Raised when reading/seeking beyond the end of the stream."""

    def __init__(
        self,
        message: str,
        *,
        bit_pos: Optional[int] = None,
        total_bits: Optional[int] = None,
        bits_needed: Optional[int] = None,
        bits_remaining: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.bit_pos = bit_pos
        self.total_bits = total_bits
        self.bits_needed = bits_needed
        self.bits_remaining = bits_remaining

class UnresolvedReferenceError(Exception):
    """Raised when an expression references a node that is not yet parsed."""
    pass

class NodeParseError(Exception):
    """Represents a node-specific parse failure (e.g., size_bits leads to OOB read)."""

    def __init__(self, node_id: int, message: str, *, kind: str = "structure", cause: Optional[Exception] = None):
        super().__init__(message)
        self.node_id = node_id
        self.kind = kind
        self.cause = cause


class RoutingError(Exception):
    """Raised when no variant/branch can be selected during parsing."""

    def __init__(
        self,
        node_id: int,
        message: str,
        variant_errors: Optional[List[Any]] = None,
        *,
        selector_id: Optional[int] = None,
        candidate_variants: Optional[List[int]] = None,
    ):
        super().__init__(message)
        self.node_id = node_id
        self.variant_errors = variant_errors or []
        self.selector_id = selector_id
        self.candidate_variants = candidate_variants or []


@dataclass
class NodeSizeInfo:
    has_declared_size: bool
    declared_size_bits: Optional[int]


@dataclass
class LengthMismatchDetail:
    src: int
    dst: int
    expected_bits: int
    actual_bits: int
    content_bits: Optional[int] = None
    wire_bits: Optional[int] = None
    mismatch_kind: str = "wire_mismatch"  # wire_mismatch | coverage_gap | overflow | truncation
    formula: Optional[str] = None
    message_type: Optional[str] = None


class LengthMismatchError(Exception):
    """Represents one or more length_of violations with structured detail."""

    def __init__(self, mismatches: List[LengthMismatchDetail]):
        self.mismatches = mismatches
        super().__init__(str(self))

    @property
    def node_id(self) -> Optional[int]:
        if not self.mismatches:
            return None
        return self.mismatches[0].dst

    def __str__(self) -> str:
        if not self.mismatches:
            return "Length mismatch"
        if len(self.mismatches) == 1:
            m = self.mismatches[0]
            return f"Length mismatch at edge {m.src}->{m.dst}: expected {m.expected_bits} bits, actual {m.actual_bits} bits"
        parts = [
            f"edge {m.src}->{m.dst}: expected {m.expected_bits} bits, actual {m.actual_bits} bits"
            for m in self.mismatches
        ]
        return "Length mismatches: " + "; ".join(parts)


def _coerce_node_id(raw: Any) -> Union[int, Any]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw

@dataclass
class RuntimeContext:
    """Stores the runtime state of a parsing session."""
    values: Dict[Union[str, int], Any] = field(default_factory=dict)  # node_id -> parsed value
    bit_starts: Dict[Union[str, int], int] = field(default_factory=dict) # node_id -> absolute bit start
    sizes: Dict[Union[str, int], int] = field(default_factory=dict)      # node_id -> declared size in bits
    actual_sizes: Dict[Union[str, int], int] = field(default_factory=dict)  # node_id -> content bits consumed (pre-padding/padding excluded)
    wire_sizes: Dict[Union[str, int], int] = field(default_factory=dict)  # node_id -> bits covered on wire (after padding/seek)
    # All node occurrences emitted during parsing. The node_id-keyed maps above
    # intentionally keep the latest value for expression evaluation, but TLV and
    # option sequences need occurrence-level spans for validation.
    node_spans: List[Dict[str, Any]] = field(default_factory=list)
    # Leaf spans emitted during parsing (supports repeated templates such as tlv_seq).
    # Each entry is an occurrence, even if node_id repeats.
    field_spans: List[Dict[str, Any]] = field(default_factory=list)
    length_gaps: List["LengthMismatchDetail"] = field(default_factory=list)  # non-fatal coverage gaps for length_of
    parsing_path: List[Union[str, int]] = field(default_factory=list)    # list of successfully parsed node_ids (legacy)
    visited_nodes: Set[Union[str, int]] = field(default_factory=set)     # nodes actually entered during this parse
    parent_on_path: Dict[Union[str, int], Optional[Union[str, int]]] = field(default_factory=dict)  # child -> parent on this path
    visit_log: List[Tuple[Union[str, int], Optional[Union[str, int]]]] = field(default_factory=list)  # ordered DFS visits
    active_variants: Dict[Union[str, int], List[Union[str, int]]] = field(default_factory=dict)  # selector -> variants activated on this packet
    # Diagnostics: speculative parsing failures for non-chosen variants.
    variant_trial_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_value(self, node_id: Union[str, int]) -> Any:
        return self.values.get(node_id)

class BitStream:
    """Reads bits from a binary buffer."""
    def __init__(self, data: bytes):
        self.data = data
        self.total_bits = len(data) * 8
        self.cursor = 0

    def seek(self, bit_pos: int):
        if bit_pos < 0 or bit_pos > self.total_bits:
            raise IncompleteStreamError(
                f"Stream ended while seeking: invalid position {bit_pos}, total_bits={self.total_bits}",
                bit_pos=bit_pos,
                total_bits=self.total_bits,
            )
        self.cursor = bit_pos

    def read_bits(self, count: int) -> int:
        if count == 0:
            return 0
        if self.cursor + count > self.total_bits:
            raise IncompleteStreamError(
                f"Not enough bits: need {count}, have {self.total_bits - self.cursor}",
                bits_needed=count,
                bits_remaining=self.total_bits - self.cursor,
                total_bits=self.total_bits,
            )
        
        value = 0
        # Simple slow implementation for now; optimize if needed
        for _ in range(count):
            byte_idx = self.cursor // 8
            bit_idx = 7 - (self.cursor % 8)
            bit = (self.data[byte_idx] >> bit_idx) & 1
            value = (value << 1) | bit
            self.cursor += 1
        return value

    def read_bytes(self, count: int) -> bytes:
        if self.cursor % 8 != 0:
            # If not byte-aligned, read bit by bit (slow but correct)
            # Or just align? For now, assume byte alignment for byte reads usually
            pass 
        
        # Optimized byte read if aligned
        if self.cursor % 8 == 0:
            byte_start = self.cursor // 8
            if byte_start + count > len(self.data):
                raise IncompleteStreamError(
                    f"Not enough bytes: need {count}, have {len(self.data) - byte_start}",
                    bits_needed=count * 8,
                    bits_remaining=(len(self.data) - byte_start) * 8,
                    total_bits=self.total_bits,
                )
            self.cursor += count * 8
            return self.data[byte_start : byte_start + count]
        
        # Fallback
        val = self.read_bits(count * 8)
        return val.to_bytes(count, 'big')

class ExpressionEvaluator:
    """Evaluates string expressions using runtime context."""

    _ALLOWED_FUNCS = {"min": min, "max": max, "abs": abs, "ceil": math.ceil}

    def __init__(
        self,
        context: RuntimeContext,
        node_types: Optional[Dict[Union[str, int], str]] = None,
    ):
        self.context = context
        self.node_types = node_types or {}

    @staticmethod
    def _coerce_node_id(raw: str) -> Union[str, int]:
        return int(raw) if raw.lstrip('-').isdigit() else raw

    @staticmethod
    def _convert_c_style_ternary(expr: str) -> str:
        if "?" not in expr or ":" not in expr:
            return expr

        depth = 0
        q_idx = -1
        for idx, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "?" and depth == 0:
                q_idx = idx
                break
        if q_idx < 0:
            return expr

        depth = 0
        colon_idx = -1
        for idx in range(q_idx + 1, len(expr)):
            ch = expr[idx]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == ":" and depth == 0:
                colon_idx = idx
                break
        if colon_idx < 0:
            return expr

        cond = expr[:q_idx].strip()
        true_expr = expr[q_idx + 1 : colon_idx].strip()
        false_expr = expr[colon_idx + 1 :].strip()
        if not cond or not true_expr or not false_expr:
            return expr
        return f"({true_expr} if ({cond}) else {false_expr})"

    def evaluate(
        self,
        expr: Union[str, int, float],
        current_node_id: Optional[Union[str, int]] = None,
        current_value: Optional[Any] = None,
    ) -> Any:
        if isinstance(expr, (int, float)):
            return expr
        if not isinstance(expr, str):
            return expr

        # Keep a copy for logging before any normalization
        raw_expr = expr

        # Normalize common Unicode variants so DSL can use wider characters safely
        expr = (
            expr.replace("：", ":")
            .replace("≤", "<=")
            .replace("≥", ">=")
            .replace("≠", "!=")
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
            .replace("\u200b", "")
            .replace("\u200c", "")
            .replace("\u200d", "")
            .replace("\ufeff", "")
            .replace("\u2060", "")
        ).strip()
        # Collapse spaced not-equal (e.g., "! =") into a valid "!=" before unary "!" handling.
        expr = re.sub(r"!\s*=", "!=", expr)
        if not expr:
            return 0
        if expr.lower() == "true":
            return True
        if expr.lower() == "false":
            return False

        # Support DSL-style "formula:" constraints and single "=" equality.
        lower_expr = expr.lower()
        if lower_expr.startswith("formula:"):
            expr_body = desugar_implication(expr.split(":", 1)[1].strip())
            # Treat a lone "=" as equality (avoid touching existing comparisons).
            expr_body = re.sub(r"(?<![<>=!])=(?!=)", "==", expr_body, count=1)
            expr = expr_body

        def unresolved(message: str) -> UnresolvedReferenceError:
            prefix = f"while evaluating node {current_node_id}: " if current_node_id is not None else ""
            return UnresolvedReferenceError(prefix + message)

        # Replace val(ID) with actual value from context
        def replace_val(match):
            nid = self._coerce_node_id(match.group(1))
            node_type = str(self.node_types.get(nid) or self.node_types.get(str(nid)) or "").lower()
            if node_type and node_type not in SCALAR_VALUE_NODE_TYPES:
                raise unresolved(
                    f"reference to non-scalar structural node val({nid}) of type {node_type}"
                )
            val = self.context.get_value(nid)
            if val is None:
                raise unresolved(f"reference to unparsed node val({nid})")
            return repr(val)

        parsed_expr = re.sub(r'val\(([\w-]+)\)', replace_val, expr)

        # Replace node_id.bit_start / size_bits references
        def replace_attr(match):
            nid = self._coerce_node_id(match.group(1))
            attr = match.group(2)
            if attr == 'bit_start':
                val = self.context.bit_starts.get(nid)
            elif attr == 'size_bits':
                val = self.context.sizes.get(nid)
            else:
                val = None
            if val is None:
                raise unresolved(f"reference to unknown attribute {nid}.{attr}")
            return repr(val)

        parsed_expr = re.sub(r'([\w-]+)\.(bit_start|size_bits)', replace_attr, parsed_expr)

        # Handle "root.bit_start" etc if root has ID 0 or "root"
        parsed_expr = parsed_expr.replace("root.bit_start", "0")

        # "value" keyword denotes the current node's parsed value
        if current_value is not None:
            parsed_expr = re.sub(r'\bvalue\b', repr(current_value), parsed_expr)

        # Normalize logical operators for Python's eval.
        # NOTE: Preserve "!=" while supporting unary "!" as "not".
        parsed_expr = parsed_expr.replace("&&", " and ").replace("||", " or ")
        parsed_expr = re.sub(r"!(?!=)", " not ", parsed_expr)
        parsed_expr = re.sub(r"\btrue\b", "True", parsed_expr, flags=re.IGNORECASE)
        parsed_expr = re.sub(r"\bfalse\b", "False", parsed_expr, flags=re.IGNORECASE)
        parsed_expr = self._convert_c_style_ternary(parsed_expr)

        try:
            parsed_ast = ast.parse(parsed_expr, mode="eval")
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name) or node.func.id not in self._ALLOWED_FUNCS:
                        raise ValueError(
                            "Invalid expression: unexpected function call. "
                            "This usually indicates a missing boolean operator between clauses (use `and`/`or`)."
                        )
            return eval(parsed_expr, {"__builtins__": None}, self._ALLOWED_FUNCS)
        except UnresolvedReferenceError:
            raise
        except Exception as e:
            # Enhanced error logging: capture detailed context for debugging
            logger.error(
                "Failed to evaluate expression. Details:\n"
                "  Original expression: %s\n"
                "  Parsed expression: %s\n"
                "  Node ID: %s\n"
                "  Exception type: %s\n"
                "  Exception message: %s\n"
                "  Available values in context: %s\n"
                "  Available bit_starts: %s\n"
                "  Available sizes: %s",
                raw_expr,
                parsed_expr,
                current_node_id,
                type(e).__name__,
                str(e),
                dict(self.context.values),
                dict(self.context.bit_starts),
                dict(self.context.sizes),
            )
            raise ValueError(
                f"Failed to evaluate expression '{expr}' (parsed: '{parsed_expr}') for node {current_node_id}: {e}"
            )

class DynamicTreeInterpreter:
    def __init__(self, protocol_tree: Dict[str, Any], target_message_type: Optional[str] = None):
        self.tree = protocol_tree
        normalize_count_length_edges(self.tree)
        self.nodes: Dict[Union[str, int], Dict[str, Any]] = {}
        self.node_types: Dict[Union[str, int], str] = {}
        for n in self.tree.get("nodes", []):
            raw_id = n.get("node_id")
            if raw_id is None:
                continue
            node_type = str(n.get("node_type") or "").lower()
            self.nodes[raw_id] = n
            self.node_types[raw_id] = node_type
            coerced = _coerce_node_id(raw_id)
            self.nodes[coerced] = n
            self.node_types[coerced] = node_type
            self.node_types[str(coerced)] = node_type

        root_raw = self.tree.get('root_node_id')
        try:
            self.root_id = _coerce_node_id(root_raw)
        except Exception:
            self.root_id = root_raw
        self.target_message_type = (target_message_type or "").lower() or None

        # Interpreter diagnostics can be extremely noisy when running over many packets.
        # By default we suppress per-packet "node overflow" warnings (they are promoted into
        # actionable issues by the traffic semantic validator). Enable to debug with:
        #   export STEP2_TRAFFIC_LOG_NODE_OVERFLOW=1
        self._log_node_overflow = os.getenv("STEP2_TRAFFIC_LOG_NODE_OVERFLOW", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self._node_overflow_warned: Set[Tuple[int, int]] = set()
        
        # Index edges for fast lookup
        self.edges = self.tree.get('edges', [])
        self.condition_edges = {} # src -> list of edges
        self.repeat_count_edges_by_dst: Dict[Union[str, int], List[Dict[str, Any]]] = {}
        self.length_edges_by_dst: Dict[Union[str, int], List[Dict[str, Any]]] = {}
        for e in self.edges:
            if e.get('rel') == 'condition_on':
                src = _coerce_node_id(e.get('src'))
                if src not in self.condition_edges:
                    self.condition_edges[src] = []
                self.condition_edges[src].append(e)
            elif e.get('rel') == 'repeat_count':
                dst = _coerce_node_id(e.get('dst'))
                self.repeat_count_edges_by_dst.setdefault(dst, []).append(e)
            elif e.get('rel') == 'length_of':
                dst = _coerce_node_id(e.get('dst'))
                self.length_edges_by_dst.setdefault(dst, []).append(e)

    def parse(self, data: bytes) -> Tuple[bool, Optional[RuntimeContext], Optional[Any]]:
        """
        Parses the data.
        Returns: (success, context, error_message)
        """
        stream = BitStream(data)
        context = RuntimeContext()
        evaluator = ExpressionEvaluator(context, node_types=self.node_types)
        
        try:
            self._parse_node(self.root_id, stream, context, evaluator, parent_id=None)

            # Check for unconsumed data
            remaining = stream.total_bits - stream.cursor
            if remaining > 0:
                return False, context, (
                    f"Incomplete parse: {remaining} bits remain unconsumed at end of stream. "
                    "This usually indicates missing payload fields/variants or incorrect size_bits."
                )

            mismatches = self._validate_length_edges(context, evaluator)
            if mismatches:
                return False, context, LengthMismatchError(mismatches)

            return True, context, None
        except RoutingError as e:
            # Routing failures are semantic parse failures (e.g., overlapping/mis-specified variants),
            # not interpreter crashes. Return the structured error so validators/MCTS can act on it.
            logger.debug("Routing failed at node %s: %s", getattr(e, "node_id", "?"), e)
            return False, context, e
        except NodeParseError as e:
            if os.getenv("STEP2_TRAFFIC_LOG_PARSE_WARNINGS", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }:
                logger.warning("Parsing failed at node %s: %s", e.node_id, e)
            else:
                logger.debug("Parsing failed at node %s: %s", e.node_id, e)
            return False, context, e
        except ConstraintViolationError as e:
            node_id = getattr(e, "node_id", "?")
            if os.getenv("STEP2_TRAFFIC_LOG_PARSE_WARNINGS", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }:
                logger.warning("Constraint violation at node %s: %s", node_id, e)
            else:
                logger.debug("Constraint violation at node %s: %s", node_id, e)
            return False, context, e
        except LengthMismatchError as e:
            if os.getenv("STEP2_TRAFFIC_LOG_PARSE_WARNINGS", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }:
                logger.warning("Length mismatch: %s", e)
            else:
                logger.debug("Length mismatch: %s", e)
            return False, context, e
        except Exception as e:
            if os.getenv("STEP2_TRAFFIC_LOG_PARSE_WARNINGS", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }:
                logger.warning("Parsing failed: %s", e)
            else:
                logger.debug("Parsing failed: %s", e)
            return False, context, str(e)

    def _parse_node(
        self,
        node_id: Union[str, int],
        stream: BitStream,
        context: RuntimeContext,
        evaluator: ExpressionEvaluator,
        parent_id: Optional[Union[str, int]] = None,
        *,
        force_current_bit_start: bool = False,
    ):
        node_id = _coerce_node_id(node_id)
        parent_id = _coerce_node_id(parent_id) if parent_id is not None else None
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        # Record traversal of this node on the current parse path
        context.visited_nodes.add(node_id)
        context.parent_on_path[node_id] = parent_id
        context.visit_log.append((node_id, parent_id))
        logger.debug("Enter node %s (parent=%s)", node_id, parent_id)

        node_type = str(node.get('node_type', 'field')).lower()
        is_leaf = node_type in LEAF_TYPES

        # 1. Determine Position (bit_start)
        # In a dynamic parser, we might trust the stream cursor OR enforce bit_start.
        # Enforcing bit_start is safer to validate the tree's logic.
        
        # Evaluate bit_start expression
        bit_start_expr = node.get('bit_start')
        if force_current_bit_start:
            # `tlv_seq` children act as a repeated item template and should be parsed from the
            # *current* cursor on each iteration. Many generated trees still anchor the item root
            # at `tlv_seq.bit_start`, which would rewind the cursor and trigger the stuck guard.
            context.bit_starts[node_id] = stream.cursor
        elif bit_start_expr is not None:
            try:
                calc_start = int(evaluator.evaluate(bit_start_expr, node_id))
                if calc_start != stream.cursor:
                    try:
                        stream.seek(calc_start)
                    except IncompleteStreamError as exc:
                        raise NodeParseError(
                            node_id,
                            (
                                "bit_start expression leads to seek past end of packet "
                                f"(node={node_id}, bit_pos={calc_start}, total_bits={stream.total_bits})"
                            ),
                            kind="structure",
                            cause=exc,
                        ) from exc

                context.bit_starts[node_id] = calc_start
                logger.debug("Node %s bit_start=%s (cursor now %s)", node_id, calc_start, stream.cursor)
            except NodeParseError:
                raise
            except Exception as exc:
                raise NodeParseError(
                    node_id,
                    f"Error calculating bit_start for node {node_id}: {exc}",
                    kind="structure",
                    cause=exc,
                ) from exc
        else:
            # If no bit_start, assume contiguous?
            context.bit_starts[node_id] = stream.cursor

        # Mark the start of this node after positioning
        node_start_cursor = stream.cursor
        consumed_before_padding = 0

        # 2. Determine Size
        size_expr = node.get('size_bits')
        size_bits = 0
        size_info = self._compute_declared_size_bits(node_id, parent_id, stream, context, evaluator)
        if size_info.has_declared_size and size_info.declared_size_bits is not None:
            if size_info.declared_size_bits < 0:
                raise NodeParseError(
                    node_id,
                    f"invalid size_bits expression: evaluated to negative {size_info.declared_size_bits} for node {node_id}",
                    kind="structure",
                )
            size_bits = size_info.declared_size_bits
            context.sizes[node_id] = size_bits
            logger.debug("Node %s declared size_bits=%s", node_id, size_bits)
        else:
            context.sizes[node_id] = size_bits

        # 3. Read Data (if Field)
        node_type = node.get('node_type', 'field')
        val_int = None
        if node_type in ['field', 'selector', 'type', 'length', 'checksum']: # Leaf nodes
            if self._is_encoded_domain_name_node(node) and not size_info.has_declared_size:
                raw_value, encoded_size_bits = self._read_dns_encoded_name(node_id, stream)
                context.values[node_id] = raw_value
                context.sizes[node_id] = encoded_size_bits
                size_bits = encoded_size_bits
                val_int = raw_value
                logger.debug("Node %s encoded domain-name size_bits=%s", node_id, encoded_size_bits)
            elif size_bits > 0:
                remaining = stream.total_bits - stream.cursor
                if size_bits > remaining:
                    length_error = self._length_truncation_error(
                        node_id=node_id,
                        node_start_cursor=node_start_cursor,
                        available_bits=remaining,
                        context=context,
                        evaluator=evaluator,
                    )
                    if length_error is not None:
                        raise length_error
                    raise NodeParseError(
                        node_id,
                        f"size_bits expression leads to read past end of packet (node={node_id}, bits_needed={size_bits}, bits_remaining={remaining})",
                        kind="structure",
                        cause=IncompleteStreamError(
                            f"Not enough bits for node {node_id}: need {size_bits}, have {remaining}"
                        ),
                    )
                val_int = decode_typed_value(stream.read_bits(size_bits), size_bits, node)
                context.values[node_id] = val_int
                logger.debug("Node %s value=%s (size_bits=%s)", node_id, val_int, size_bits)
            else:
                # Zero size field?
                context.values[node_id] = 0
                val_int = 0
        
        # 3.5 Check Constraints
        constraints = node.get('constraints', [])
        if constraints and val_int is not None:
            for constraint in constraints:
                if not constraint:
                    continue
                if isinstance(constraint, dict):
                    constraint_type = str(
                        constraint.get("type") or constraint.get("kind") or ""
                    ).strip().lower()
                    try:
                        if constraint_type == "enum":
                            raw_values = (
                                constraint.get("values")
                                or constraint.get("valid_values")
                                or constraint.get("allowed")
                                or constraint.get("enum")
                                or []
                            )
                            if not isinstance(raw_values, (list, tuple, set)):
                                raw_values = [raw_values]
                            allowed = []
                            for item in raw_values:
                                try:
                                    allowed.append(int(item, 0) if isinstance(item, str) else int(item))
                                except Exception:
                                    continue
                            if allowed and val_int not in allowed:
                                if self._can_relax_priority_controlled_spare_enum(
                                    node_id=node_id,
                                    node=node,
                                    value=val_int,
                                    allowed=allowed,
                                    context=context,
                                ):
                                    continue
                                raise ConstraintViolationError(
                                    node_id=node_id,
                                    constraint_text=f"enum {allowed}",
                                    value=val_int,
                                    kind="enum",
                                )
                            continue
                        if constraint_type == "range":
                            lo_raw = constraint.get("min", constraint.get("min_value"))
                            hi_raw = constraint.get("max", constraint.get("max_value"))
                            lo = int(lo_raw, 0) if isinstance(lo_raw, str) else lo_raw
                            hi = int(hi_raw, 0) if isinstance(hi_raw, str) else hi_raw
                            if lo is None and hi is None:
                                continue
                            target_kind = str(constraint.get("target") or "value").strip().lower()
                            target_val = val_int
                            if target_kind == "size_bits":
                                target_val = context.sizes.get(node_id)
                            elif target_kind == "bit_start":
                                target_val = context.bit_starts.get(node_id)
                            if target_val is None:
                                continue
                            if lo is not None and int(target_val) < int(lo):
                                raise ConstraintViolationError(
                                    node_id=node_id,
                                    constraint_text=f"range {lo}..{hi} ({target_kind})",
                                    value=target_val,
                                    kind="range",
                                )
                            if hi is not None and int(target_val) > int(hi):
                                raise ConstraintViolationError(
                                    node_id=node_id,
                                    constraint_text=f"range {lo}..{hi} ({target_kind})",
                                    value=target_val,
                                    kind="range",
                                )
                            continue
                    except ConstraintViolationError:
                        raise
                    except Exception as e:
                        raise ValueError(f"Error evaluating constraint '{constraint}' for node {node_id}: {e}")
                    # Dicts with only prose metadata are not executable constraints.
                    continue
                if not isinstance(constraint, str):
                    continue
                constraint_str = constraint.strip() if isinstance(constraint, str) else constraint
                try:
                    if isinstance(constraint_str, str):
                        normalized_cs = (
                            constraint_str.replace("：", ":")
                            .replace("≤", "<=")
                            .replace("≥", ">=")
                            .replace("\u00a0", " ")
                        )
                        lower = normalized_cs.lower()
                        if lower.startswith((
                            "description:",
                            "desc:",
                            "note:",
                            "notes:",
                            "source:",
                            "meaning:",
                            "conditional:",
                            "condition:",
                        )):
                            continue
                        if is_constraint_ir_text(normalized_cs):
                            continue
                        # Support Step2-style enum/range prefixes
                        if lower.startswith("enum:"):
                            parsed_allowed = parse_numeric_enum_constraint_text(normalized_cs)
                            allowed = list(parsed_allowed) if parsed_allowed is not None else None
                            if allowed is None:
                                # Descriptive strings such as "enum: Structured option
                                # data ... sections 2-9" are metadata, not executable
                                # enumerations.  Do not mine incidental numbers from prose.
                                continue
                            if not allowed:
                                raise ValueError(f"Malformed enum constraint for node {node_id}: {constraint}")
                            if val_int not in allowed:
                                if self._can_relax_priority_controlled_spare_enum(
                                    node_id=node_id,
                                    node=node,
                                    value=val_int,
                                    allowed=allowed,
                                    context=context,
                                ):
                                    continue
                                raise ConstraintViolationError(
                                    node_id=node_id,
                                    constraint_text=f"enum {allowed}",
                                    value=val_int,
                                    kind="enum",
                                )
                            continue  # handled enum; no fallback eval
                        if lower.startswith("range:"):
                            bounds = parse_numeric_range_constraint_text(normalized_cs)
                            if bounds is None:
                                lower_bound = parse_numeric_calculated_upper_range_constraint_text(normalized_cs)
                                if lower_bound is None:
                                    raise ValueError(f"Malformed range constraint for node {node_id}: {constraint}")
                                if val_int < lower_bound:
                                    raise ConstraintViolationError(
                                        node_id=node_id,
                                        constraint_text=f"range >= {lower_bound} (value)",
                                        value=val_int,
                                        kind="range",
                                    )
                                continue
                            lo, hi = bounds
                            # Support range constraints over node attributes (e.g., "range: 8 ≤ size_bits ≤ 128").
                            # By default range constraints apply to the node's parsed numeric value.
                            target_kind = "value"
                            target_val = val_int
                            if re.search(r"\bsize_bits\b", lower):
                                target_kind = "size_bits"
                                target_val = context.sizes.get(node_id)
                            elif re.search(r"\bbit_start\b", lower):
                                target_kind = "bit_start"
                                target_val = context.bit_starts.get(node_id)

                            if target_val is None:
                                raise ValueError(
                                    f"Malformed range constraint for node {node_id}: {constraint} (missing {target_kind})"
                                )

                            if not (lo <= int(target_val) <= hi):
                                raise ConstraintViolationError(
                                    node_id=node_id,
                                    constraint_text=f"range {lo}..{hi} ({target_kind})",
                                    value=target_val,
                                    kind="range",
                                )
                            continue  # handled range; no fallback eval
                        # Fallback to expression evaluation
                    result = evaluator.evaluate(constraint_str, node_id, current_value=val_int)
                except ConstraintViolationError:
                    raise
                except Exception as e:
                    raise ValueError(f"Error evaluating constraint '{constraint}' for node {node_id}: {e}")
                if not result:
                    raise ConstraintViolationError(
                        node_id=node_id,
                        constraint_text=str(constraint),
                        value=val_int,
                        kind="expr",
                    )
                logger.debug("Node %s constraint ok: %s (value=%s)", node_id, constraint, val_int)

        # Record successful parse of this node
        context.parsing_path.append(node_id)

        # 4. Process Children
        children_ids = self._filter_redundant_alias_children(node_id, node.get('children_ids', []))
        node_type_lower = str(node_type).lower() if node_type is not None else ""

        if node_type_lower == "tlv_seq":
            active_children_for_seq = [
                _coerce_node_id(child_id)
                for child_id in children_ids
                if self._is_node_active(_coerce_node_id(child_id), context, evaluator)
            ]
            if len(active_children_for_seq) == 1:
                repeated_child_id = active_children_for_seq[0]
                repeat_count = self._repeat_count_for_node(repeated_child_id, context, evaluator)
                if repeat_count is not None:
                    self._parse_repeated_child(
                        repeated_child_id,
                        stream,
                        context,
                        evaluator,
                        parent_id=node_id,
                        repeat_count=repeat_count,
                    )
                    actual_consumed_bits = stream.cursor - node_start_cursor
                    context.sizes[node_id] = int(actual_consumed_bits)
                    context.actual_sizes[node_id] = int(actual_consumed_bits)
                    context.wire_sizes[node_id] = int(actual_consumed_bits)
                    self._record_node_occurrence(
                        context,
                        node_id=node_id,
                        parent_id=parent_id,
                        bit_start=node_start_cursor,
                        wire_size_bits=actual_consumed_bits,
                        content_size_bits=actual_consumed_bits,
                        value=None,
                    )
                    return

            stop_formula = node.get("stop_condition") or node.get("repeat_until")
            try:
                max_items = int(node.get("max_items") or os.getenv("TLV_SEQ_MAX_ITEMS", "1024"))
            except Exception:
                max_items = 1024
            max_items = max(1, max_items)

            # Optional hard limit on the sequence bounds if declared.
            limit_end: Optional[int] = None
            if size_info.has_declared_size and size_info.declared_size_bits is not None:
                try:
                    declared_bits = int(size_info.declared_size_bits)
                    if declared_bits >= 0:
                        limit_end = node_start_cursor + declared_bits
                except Exception:
                    limit_end = None

            iterations = 0
            while True:
                if iterations >= max_items:
                    raise NodeParseError(
                        int(node_id),
                        f"tlv_seq exceeded max_items={max_items}",
                        kind="structure",
                    )

                if limit_end is not None and stream.cursor >= limit_end:
                    break

                remaining_bits = (limit_end - stream.cursor) if limit_end is not None else (stream.total_bits - stream.cursor)
                if remaining_bits < 8:
                    break

                before = stream.cursor
                for child_id in children_ids:
                    child_id = _coerce_node_id(child_id)
                    if not self._is_node_active(child_id, context, evaluator):
                        continue
                    self._parse_node(
                        child_id,
                        stream,
                        context,
                        evaluator,
                        parent_id=node_id,
                        force_current_bit_start=True,
                    )

                if stream.cursor <= before:
                    raise NodeParseError(
                        int(node_id),
                        "tlv_seq item consumed 0 bits (stuck)",
                        kind="structure",
                    )

                iterations += 1

                if stop_formula:
                    try:
                        if bool(evaluator.evaluate(stop_formula, node_id)):
                            break
                    except UnresolvedReferenceError:
                        # If the stop condition references a field that wasn't parsed, treat it as false.
                        pass
                    except Exception as exc:
                        raise NodeParseError(
                            int(node_id),
                            f"tlv_seq stop_condition failed to evaluate: {exc}",
                            kind="semantics",
                            cause=exc,
                        ) from exc

            actual_consumed_bits = stream.cursor - node_start_cursor
            context.sizes[node_id] = int(actual_consumed_bits)
            context.actual_sizes[node_id] = int(actual_consumed_bits)
            context.wire_sizes[node_id] = int(actual_consumed_bits)
            self._record_node_occurrence(
                context,
                node_id=node_id,
                parent_id=parent_id,
                bit_start=node_start_cursor,
                wire_size_bits=actual_consumed_bits,
                content_size_bits=actual_consumed_bits,
                value=None,
            )
            return

        # Precompute an execution sequence with variant groups for speculative parsing
        def _selector_for_child(cid: Union[str, int]) -> Optional[Union[str, int]]:
            return self._find_selector_for_variant(cid)

        def _is_variant_member(cid: Union[str, int]) -> Tuple[bool, Optional[Union[str, int]]]:
            cid = _coerce_node_id(cid)
            child = self.nodes.get(cid) or {}
            node_type = str(child.get("node_type") or "").lower()
            has_children = bool(child.get("children_ids"))
            selector_id = _selector_for_child(cid)

            if node_type == "variant":
                return True, selector_id

            if selector_id is not None and has_children:
                return True, selector_id

            return False, selector_id

        active_sequence: List[Tuple[str, List[Union[str, int]]]] = []
        current_group_selector: Optional[Union[str, int]] = None
        current_group_members: List[Union[str, int]] = []

        for child_id in children_ids:
            child_id = _coerce_node_id(child_id)
            child_node = self.nodes.get(child_id)
            if not child_node:
                continue

            is_variant_like, sel_id = _is_variant_member(child_id)

            if not is_variant_like:
                if current_group_members:
                    active_sequence.append(("group", current_group_members))
                    current_group_members = []
                    current_group_selector = None
                active_sequence.append(("single", [child_id]))
                continue

            if current_group_selector is None:
                current_group_selector = sel_id
                current_group_members = [child_id]
            else:
                if sel_id == current_group_selector:
                    current_group_members.append(child_id)
                else:
                    active_sequence.append(("group", current_group_members))
                    current_group_members = [child_id]
                    current_group_selector = sel_id

        if current_group_members:
            active_sequence.append(("group", current_group_members))

        for item_type, item_data in active_sequence:
            if item_type == 'single':
                child_id = item_data[0]
                if not self._is_node_active(child_id, context, evaluator):
                    continue
                repeat_count = self._repeat_count_for_node(child_id, context, evaluator)
                if repeat_count is None:
                    self._parse_node(child_id, stream, context, evaluator, parent_id=node_id)
                else:
                    self._parse_repeated_child(
                        child_id,
                        stream,
                        context,
                        evaluator,
                        parent_id=node_id,
                        repeat_count=repeat_count,
                    )
            elif item_type == 'group':
                candidates = [
                    cid for cid in item_data if self._is_node_active(cid, context, evaluator)
                ]
                if not candidates:
                    continue

                # Fast path if only one candidate in the group
                if len(candidates) == 1:
                    variant_id = candidates[0]
                    selector_id = self._find_selector_for_variant(variant_id)
                    if selector_id is not None:
                        variants_list = context.active_variants.setdefault(selector_id, [])
                        if variant_id not in variants_list:
                            variants_list.append(variant_id)
                    self._parse_node(variant_id, stream, context, evaluator, parent_id=node_id)
                    continue

                start_cursor = stream.cursor
                base_context = copy.deepcopy(context)
                best_candidate = None
                best_score: Tuple[int, int] = (-1, -1)  # (content_bits, wire_bits)
                last_error: Optional[Exception] = None
                variant_errors: List[Any] = []
                selector_id_for_group: Optional[int] = None
                try:
                    selector_id_for_group = self._find_selector_for_variant(candidates[0]) if candidates else None
                    if selector_id_for_group is not None:
                        selector_id_for_group = int(selector_id_for_group)
                except Exception:
                    selector_id_for_group = None

                for var_id in candidates:
                    # Restore snapshot
                    stream.seek(start_cursor)
                    context.values = copy.deepcopy(base_context.values)
                    context.bit_starts = copy.deepcopy(base_context.bit_starts)
                    context.sizes = copy.deepcopy(base_context.sizes)
                    context.actual_sizes = copy.deepcopy(base_context.actual_sizes)
                    context.wire_sizes = copy.deepcopy(base_context.wire_sizes)
                    context.length_gaps = list(base_context.length_gaps)
                    context.parsing_path = list(base_context.parsing_path)
                    context.visited_nodes = set(base_context.visited_nodes)
                    context.parent_on_path = dict(base_context.parent_on_path)
                    context.visit_log = list(base_context.visit_log)
                    context.active_variants = copy.deepcopy(base_context.active_variants)
                    context.node_spans = list(getattr(base_context, "node_spans", []))
                    context.field_spans = list(getattr(base_context, "field_spans", []))
                    context.variant_trial_errors = copy.deepcopy(getattr(base_context, "variant_trial_errors", []))

                    try:
                        self._parse_node(var_id, stream, context, evaluator, parent_id=node_id)
                        # Prefer variants that actually parse more content (excluding seek/pad),
                        # so empty catch-all variants with only length_of-driven seek do not dominate.
                        content_bits = context.actual_sizes.get(var_id)
                        if content_bits is None:
                            content_bits = context.actual_sizes.get(str(var_id), 0)
                        try:
                            content_bits_int = int(content_bits) if content_bits is not None else 0
                        except Exception:
                            content_bits_int = 0
                        wire_bits_int = stream.cursor - start_cursor
                        score = (content_bits_int, wire_bits_int)
                        if score > best_score:
                            best_candidate = var_id
                            best_score = score
                    except (
                        ConstraintViolationError,
                        IncompleteStreamError,
                        LengthMismatchError,
                        ValueError,
                        RoutingError,
                    ) as e:
                        last_error = e
                        variant_errors.append((var_id, e))
                        continue
                    except NodeParseError as e:
                        last_error = e
                        variant_errors.append((var_id, e))
                        continue

                # Revert to base state before committing the winner
                stream.seek(start_cursor)
                context.values = copy.deepcopy(base_context.values)
                context.bit_starts = copy.deepcopy(base_context.bit_starts)
                context.sizes = copy.deepcopy(base_context.sizes)
                context.actual_sizes = copy.deepcopy(base_context.actual_sizes)
                context.wire_sizes = copy.deepcopy(base_context.wire_sizes)
                context.length_gaps = list(base_context.length_gaps)
                context.parsing_path = list(base_context.parsing_path)
                context.visited_nodes = set(base_context.visited_nodes)
                context.parent_on_path = dict(base_context.parent_on_path)
                context.visit_log = list(base_context.visit_log)
                context.active_variants = copy.deepcopy(base_context.active_variants)
                context.node_spans = list(getattr(base_context, "node_spans", []))
                context.field_spans = list(getattr(base_context, "field_spans", []))
                context.variant_trial_errors = copy.deepcopy(getattr(base_context, "variant_trial_errors", []))

                if best_candidate is None:
                    raise RoutingError(
                        node_id,
                        f"Routing failed at node {node_id}: None of the {len(candidates)} variants could be parsed. "
                        "This indicates a selector/branching logic error rather than a leaf constraint failure.",
                        variant_errors=variant_errors,
                        selector_id=selector_id_for_group,
                        candidate_variants=[_coerce_node_id(cid) for cid in candidates],
                    )

                # Record failing candidate attempts so the validator can surface
                # broken variants even when the overall parse succeeds via fallback.
                if variant_errors:
                    max_records = max(1, int(os.getenv("STEP2_TRAFFIC_MAX_VARIANT_TRIAL_ERRORS", "8")))

                    def _safe_int(value: Any) -> Optional[int]:
                        try:
                            return int(value)
                        except Exception:
                            return None

                    def _eval_int_expr(expr: Any, current_id: int) -> Optional[int]:
                        if expr is None:
                            return None
                        try:
                            return int(evaluator.evaluate(expr, current_id))
                        except Exception:
                            return None

                    for var_id, exc in variant_errors[:max_records]:
                        var_node = self.nodes.get(_coerce_node_id(var_id)) or {}
                        bit_start_eval = _eval_int_expr(var_node.get("bit_start"), int(var_id))
                        if bit_start_eval is None:
                            bit_start_eval = _safe_int(start_cursor)
                        size_bits_eval = self._eval_size_expr(var_node.get("size_bits"), evaluator, int(var_id))
                        declared_end = (
                            int(bit_start_eval) + int(size_bits_eval)
                            if bit_start_eval is not None and size_bits_eval is not None
                            else None
                        )

                        record: Dict[str, Any] = {
                            "group_parent_id": int(node_id),
                            "selector_id": int(selector_id_for_group) if selector_id_for_group is not None else None,
                            "candidate_variants": [int(_coerce_node_id(cid)) for cid in candidates],
                            "variant_id": int(_coerce_node_id(var_id)),
                            "variant_name": str(var_node.get("name") or ""),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "bit_start_eval": bit_start_eval,
                            "size_bits_eval": size_bits_eval,
                            "declared_end_bit": declared_end,
                            "packet_total_bits": int(getattr(stream, "total_bits", 0) or 0),
                        }
                        if isinstance(exc, NodeParseError):
                            record["node_id"] = int(getattr(exc, "node_id", -1))
                            record["node_error_kind"] = str(getattr(exc, "kind", "") or "")
                            cause = getattr(exc, "cause", None)
                            if cause is not None:
                                record["cause_type"] = type(cause).__name__
                                record["cause_message"] = str(cause)
                                if isinstance(cause, IncompleteStreamError):
                                    record["cause_bit_pos"] = getattr(cause, "bit_pos", None)
                                    record["cause_total_bits"] = getattr(cause, "total_bits", None)
                                    record["cause_bits_needed"] = getattr(cause, "bits_needed", None)
                                    record["cause_bits_remaining"] = getattr(cause, "bits_remaining", None)
                        if isinstance(exc, IncompleteStreamError):
                            record["error_bit_pos"] = getattr(exc, "bit_pos", None)
                            record["error_total_bits"] = getattr(exc, "total_bits", None)
                            record["error_bits_needed"] = getattr(exc, "bits_needed", None)
                            record["error_bits_remaining"] = getattr(exc, "bits_remaining", None)
                        if isinstance(exc, ConstraintViolationError):
                            record["constraint_node_id"] = int(getattr(exc, "node_id", -1))
                            record["constraint_text"] = str(getattr(exc, "constraint_text", "") or "")
                            record["constraint_value"] = getattr(exc, "value", None)
                            record["constraint_kind"] = str(getattr(exc, "kind", "") or "")

                        try:
                            context.variant_trial_errors.append(record)
                        except Exception:
                            # Best-effort: never fail parsing due to diagnostics.
                            pass

                selector_id = self._find_selector_for_variant(best_candidate)
                if selector_id is not None:
                    variants_list = context.active_variants.setdefault(selector_id, [])
                    if best_candidate not in variants_list:
                        variants_list.append(best_candidate)

                logger.debug(
                    "Speculative Parsing: Winner is %s (content_bits=%s, wire_bits=%s)",
                    best_candidate,
                    best_score[0],
                    best_score[1],
                )
                self._parse_node(best_candidate, stream, context, evaluator, parent_id=node_id)

        # 5. Deferred size re-evaluation and padding adjustment
        final_size_bits = size_bits
        actual_consumed_bits = stream.cursor - node_start_cursor
        consumed_before_padding = actual_consumed_bits
        node_type_lower = str(node_type).lower() if node_type is not None else ""
        if node_type_lower in ["protocol", "header", "payload", "variant", "container"]:
            # If `size_bits` was specified but could not be evaluated at node entry (commonly because it
            # depends on a value parsed within this node, e.g., IPv4 IHL), try again now that children
            # have populated `context.values`. This makes sibling positioning via `<node>.size_bits`
            # reflect runtime sizes and avoids defaulting to 0.
            if not (size_info.has_declared_size and size_info.declared_size_bits is not None):
                reevaluated: Optional[int] = self._eval_size_expr(size_expr, evaluator, node_id)
                if reevaluated is None:
                    for edge in self.edges:
                        if edge.get("rel") != "length_of":
                            continue
                        dst = _coerce_node_id(edge.get("dst"))
                        if dst != node_id:
                            continue
                        formula = edge.get("formula")
                        if not formula:
                            continue
                        val = self._eval_size_expr(formula, evaluator, edge.get("src"))
                        if val is None:
                            continue
                        reevaluated = val
                        break

                if reevaluated is not None and int(reevaluated) >= 0:
                    try:
                        declared = int(reevaluated)
                    except Exception:
                        declared = -1
                    if declared >= 0:
                        size_info = NodeSizeInfo(True, declared)
                        context.sizes[node_id] = declared
                        final_size_bits = declared

            if size_info.has_declared_size and size_info.declared_size_bits is not None:
                declared_bits = size_info.declared_size_bits
                target_end = node_start_cursor + declared_bits
                declared_size_is_structural_hint = (
                    node_type_lower in {"header", "container"}
                    and bool(children_ids or [])
                    and not self._has_incoming_length_of(node_id)
                )
                if target_end < node_start_cursor:
                    logger.warning(
                        "Node %s declared_size_bits=%s is negative; ignoring declared size",
                        node_id,
                        declared_bits,
                    )
                else:
                    if target_end > stream.total_bits:
                        if declared_size_is_structural_hint:
                            context.sizes[node_id] = int(actual_consumed_bits)
                            final_size_bits = int(actual_consumed_bits)
                        else:
                            length_error = self._length_truncation_error(
                                node_id=node_id,
                                node_start_cursor=node_start_cursor,
                                available_bits=max(0, stream.total_bits - node_start_cursor),
                                context=context,
                                evaluator=evaluator,
                            )
                            if length_error is not None:
                                raise length_error
                            cause = IncompleteStreamError(
                                f"Stream ended while seeking: invalid position {target_end}, total_bits={stream.total_bits}",
                                bit_pos=target_end,
                                total_bits=stream.total_bits,
                            )
                            raise NodeParseError(
                                node_id,
                                (
                                    "size_bits expression leads to read past end of packet "
                                    f"(node={node_id}, declared_end={target_end}, total_bits={stream.total_bits})"
                                ),
                                kind="structure",
                                cause=cause,
                            ) from cause
                    if actual_consumed_bits > declared_bits:
                        if self._log_node_overflow:
                            try:
                                key = (int(node_id), int(target_end))
                            except Exception:
                                key = (str(node_id), int(target_end))  # type: ignore[assignment]
                            if key not in self._node_overflow_warned:
                                self._node_overflow_warned.add(key)
                                logger.warning(
                                    "Node %s children consumed beyond declared size: consumed_bits=%s, declared_end_bit=%s",
                                    node_id,
                                    actual_consumed_bits,
                                    target_end,
                                )
                    else:
                        remaining_pad = declared_bits - actual_consumed_bits
                        if remaining_pad > 0:
                            should_pad_to_declared = not declared_size_is_structural_hint
                            if should_pad_to_declared:
                                logger.debug(
                                    "Node %s padding/skip %s bits to align to declared size %s",
                                    node_id,
                                    remaining_pad,
                                    declared_bits,
                                )
                                stream.seek(target_end)
                                actual_consumed_bits = stream.cursor - node_start_cursor
                            else:
                                context.sizes[node_id] = int(actual_consumed_bits)
                                final_size_bits = int(actual_consumed_bits)

        content_bits = consumed_before_padding if consumed_before_padding >= 0 else actual_consumed_bits
        # Treat structural leaf-like payload/variant/container nodes as consuming their declared span.
        # These nodes often represent uninterpreted byte blobs where we don't need to decode a scalar
        # value, but they should still contribute to content coverage and length_of accounting.
        if node_type_lower in {"payload", "variant", "container"} and not (children_ids or []) and actual_consumed_bits > 0:
            content_bits = actual_consumed_bits
        context.actual_sizes[node_id] = content_bits
        context.wire_sizes[node_id] = actual_consumed_bits
        self._record_node_occurrence(
            context,
            node_id=node_id,
            parent_id=parent_id,
            bit_start=node_start_cursor,
            wire_size_bits=actual_consumed_bits,
            content_size_bits=content_bits,
            value=val_int,
        )

        # Record a leaf occurrence span (supports repeated templates such as tlv_seq).
        try:
            is_leaf_for_span = (node_type_lower in LEAF_TYPES) or not (children_ids or [])
            if is_leaf_for_span and actual_consumed_bits > 0:
                bit_start_val = context.bit_starts.get(node_id, node_start_cursor)
                context.field_spans.append(
                    {
                        "node_id": node_id,
                        "bit_start": int(bit_start_val),
                        "wire_size_bits": int(actual_consumed_bits),
                        "content_size_bits": int(content_bits),
                    }
                )
        except Exception:
            # Best-effort: never fail parsing due to diagnostics.
            pass

        # For container-like nodes with no usable declared size, expose the actual parsed span via
        # `context.sizes`. Many trees position siblings using `<prev>.size_bits`; if the original
        # size expression references inactive or not-yet-parsed children, keeping the initial zero
        # would rewind later siblings even though this node has consumed a concrete wire span.
        try:
            if not (size_info.has_declared_size and size_info.declared_size_bits is not None):
                if node_type_lower in {"protocol", "header", "payload", "variant", "container"}:
                    if actual_consumed_bits >= 0:
                        context.sizes[node_id] = int(actual_consumed_bits)
        except Exception:
            pass

    def _has_incoming_length_of(self, node_id: Union[str, int]) -> bool:
        node_id = _coerce_node_id(node_id)
        return bool(self.length_edges_by_dst.get(node_id))

    def _length_truncation_error(
        self,
        *,
        node_id: Union[str, int],
        node_start_cursor: int,
        available_bits: int,
        context: RuntimeContext,
        evaluator: ExpressionEvaluator,
    ) -> Optional[LengthMismatchError]:
        """Attribute a direct short read to an already-resolved ``length_of`` edge.

        This conversion is deliberately narrow.  The source must have completed
        parsing before this occurrence of the destination, its formula must be
        resolvable, and the formula must demand more bits than remain from the
        destination start.  Constraint, routing, and other child failures retain
        their existing priority because they are raised before this helper is
        reached and are never rewritten here.
        """

        dst = _coerce_node_id(node_id)
        incoming = self.length_edges_by_dst.get(dst, ())
        if not incoming:
            return None

        available = max(0, int(available_bits))
        details: List[LengthMismatchDetail] = []
        for edge in incoming:
            edge_msg_type = str(edge.get("message_type") or "bidirectional").lower()
            if self.target_message_type and edge_msg_type not in {"bidirectional", self.target_message_type}:
                continue

            src = _coerce_node_id(edge.get("src"))
            source_was_parsed = any(
                _coerce_node_id(occ.get("node_id")) == src
                and int(occ.get("bit_start", -1)) >= 0
                and int(occ.get("bit_start", -1))
                + int(occ.get("wire_size_bits", occ.get("size_bits", 0)) or 0)
                <= int(node_start_cursor)
                for occ in (getattr(context, "node_spans", None) or ())
            )
            if not source_was_parsed:
                continue

            formula = edge.get("formula")
            try:
                expected = context.get_value(src) if formula is None else evaluator.evaluate(formula, src)
                if isinstance(expected, bool):
                    expected = int(expected)
                expected_bits = int(expected)
            except Exception:
                continue
            if expected_bits < 0 or expected_bits <= available:
                continue

            details.append(
                LengthMismatchDetail(
                    src=int(src),
                    dst=int(dst),
                    expected_bits=expected_bits,
                    actual_bits=available,
                    content_bits=available,
                    wire_bits=available,
                    mismatch_kind="truncation",
                    formula=str(formula) if formula is not None else None,
                    message_type=edge.get("message_type"),
                )
            )

        return LengthMismatchError(details) if details else None

    def _can_relax_priority_controlled_spare_enum(
        self,
        *,
        node_id: Union[str, int],
        node: Dict[str, Any],
        value: Any,
        allowed: List[int],
        context: RuntimeContext,
    ) -> bool:
        """Avoid treating priority bits as spare bits in over-coarse full-byte fields.

        Some specs encode an optional priority nibble in an octet otherwise described
        as spare/reserved. If the tree lacks the finer split, a full-byte `enum: 0`
        over that octet rejects valid packets. This relaxation is intentionally narrow:
        it applies only to 8-bit spare/reserved leaves, only when the low nibble is
        still zero, and only after a parsed priority flag is active.
        """
        if set(int(v) for v in allowed) != {0}:
            return False
        try:
            val_int = int(value)
        except Exception:
            return False
        if val_int == 0 or (val_int & 0x0F) != 0:
            return False
        try:
            size_bits = int(node.get("size_bits"))
        except Exception:
            return False
        if size_bits != 8:
            return False

        tokens = set(self._name_tokens_from_text(node.get("name")))
        if not tokens.intersection({"spare", "reserved"}):
            return False

        for parsed_id, parsed_value in context.values.items():
            try:
                if int(parsed_value) != 1:
                    continue
            except Exception:
                continue
            parsed_node = self.nodes.get(_coerce_node_id(parsed_id)) or {}
            parsed_tokens = set(self._name_tokens_from_text(parsed_node.get("name")))
            if "priority" in parsed_tokens and "flag" in parsed_tokens:
                logger.debug(
                    "Relaxing over-coarse spare enum on node %s because priority flag node %s is active",
                    node_id,
                    parsed_id,
                )
                return True
        return False

    def _record_node_occurrence(
        self,
        context: RuntimeContext,
        *,
        node_id: Union[str, int],
        parent_id: Optional[Union[str, int]],
        bit_start: int,
        wire_size_bits: int,
        content_size_bits: int,
        value: Optional[Any],
    ) -> None:
        """Record one parsed node occurrence for repeated template validation."""
        try:
            record: Dict[str, Any] = {
                "node_id": _coerce_node_id(node_id),
                "parent_id": _coerce_node_id(parent_id) if parent_id is not None else None,
                "bit_start": int(bit_start),
                "wire_size_bits": int(wire_size_bits),
                "content_size_bits": int(content_size_bits),
            }
            if value is not None:
                record["value"] = value
            context.node_spans.append(record)
        except Exception:
            # Diagnostics only; parsing must not depend on this record.
            pass

    def _is_encoded_domain_name_node(self, node: Dict[str, Any]) -> bool:
        """Detect DNS-style encoded names when the IR marks the leaf explicitly."""
        data_type = str(node.get("data_type") or node.get("semantic_type") or "").strip().lower()
        return data_type in {"domain_name", "dns_name", "compressed_domain_name"}

    def _read_dns_encoded_name(self, node_id: Union[str, int], stream: BitStream) -> Tuple[bytes, int]:
        """Consume one DNS encoded name, including compression pointers.

        The parser records the wire bytes it consumed. It does not follow compression
        pointers because traffic validation only needs the consumed span and local
        structural validity.
        """
        start = stream.cursor
        if start % 8 != 0:
            raise NodeParseError(
                int(node_id),
                "DNS encoded name must start on a byte boundary",
                kind="structure",
            )

        chunks: List[int] = []
        labels_seen = 0
        max_labels = int(os.getenv("STEP2_DNS_NAME_MAX_LABELS", "128"))
        while True:
            if labels_seen >= max_labels:
                raise NodeParseError(
                    int(node_id),
                    f"DNS encoded name exceeded max labels={max_labels}",
                    kind="structure",
                )
            if stream.cursor + 8 > stream.total_bits:
                raise NodeParseError(
                    int(node_id),
                    "DNS encoded name ended before a length octet",
                    kind="structure",
                    cause=IncompleteStreamError(
                        "Not enough bits for DNS label length",
                        bits_needed=8,
                        bits_remaining=stream.total_bits - stream.cursor,
                        total_bits=stream.total_bits,
                    ),
                )
            length_octet = stream.read_bits(8)
            chunks.append(length_octet)
            labels_seen += 1

            tag = length_octet & 0xC0
            if tag == 0xC0:
                if stream.cursor + 8 > stream.total_bits:
                    raise NodeParseError(
                        int(node_id),
                        "DNS compression pointer ended early",
                        kind="structure",
                        cause=IncompleteStreamError(
                            "Not enough bits for DNS compression pointer",
                            bits_needed=8,
                            bits_remaining=stream.total_bits - stream.cursor,
                            total_bits=stream.total_bits,
                        ),
                    )
                chunks.append(stream.read_bits(8))
                break
            if tag != 0:
                raise NodeParseError(
                    int(node_id),
                    f"invalid DNS label tag 0x{tag:02x}",
                    kind="structure",
                )
            label_len = length_octet & 0x3F
            if label_len == 0:
                break
            if stream.cursor + (label_len * 8) > stream.total_bits:
                raise NodeParseError(
                    int(node_id),
                    f"DNS label length {label_len} exceeds packet bounds",
                    kind="structure",
                    cause=IncompleteStreamError(
                        "Not enough bits for DNS label payload",
                        bits_needed=label_len * 8,
                        bits_remaining=stream.total_bits - stream.cursor,
                        total_bits=stream.total_bits,
                    ),
                )
            label_bytes = stream.read_bytes(label_len)
            chunks.extend(label_bytes)

        return bytes(chunks), int(stream.cursor - start)

    def _repeat_count_for_node(
        self,
        node_id: Union[str, int],
        context: RuntimeContext,
        evaluator: ExpressionEvaluator,
    ) -> Optional[int]:
        edges = self.repeat_count_edges_by_dst.get(_coerce_node_id(node_id)) or []
        if not edges:
            return None
        applicable: List[Dict[str, Any]] = []
        for edge in edges:
            edge_msg_type = str(edge.get("message_type") or "bidirectional").lower()
            if self.target_message_type and edge_msg_type not in {"bidirectional", self.target_message_type}:
                continue
            applicable.append(edge)
        if not applicable:
            return None

        counts: List[int] = []
        for edge in applicable:
            formula = edge.get("formula")
            if formula is None:
                src = _coerce_node_id(edge.get("src"))
                raw = context.get_value(src)
            else:
                raw = evaluator.evaluate(formula, edge.get("src"))
            if isinstance(raw, bool):
                raw = int(raw)
            try:
                counts.append(max(0, int(raw)))
            except Exception as exc:
                raise NodeParseError(
                    int(_coerce_node_id(node_id)),
                    f"repeat_count formula failed to evaluate (edge {edge.get('src')}->{edge.get('dst')}): {exc}",
                    kind="semantics",
                    cause=exc,
                ) from exc
        if not counts:
            return None
        return max(counts)

    def _parse_repeated_child(
        self,
        child_id: Union[str, int],
        stream: BitStream,
        context: RuntimeContext,
        evaluator: ExpressionEvaluator,
        *,
        parent_id: Union[str, int],
        repeat_count: int,
    ) -> None:
        child_id = _coerce_node_id(child_id)
        if repeat_count <= 0:
            self._mark_zero_width_node(child_id, stream, context, parent_id=parent_id)
            return

        max_items = max(1, int(os.getenv("STEP2_REPEAT_COUNT_MAX_ITEMS", "4096")))
        if repeat_count > max_items:
            raise NodeParseError(
                int(child_id),
                f"repeat_count={repeat_count} exceeds max_items={max_items}",
                kind="structure",
            )

        for _ in range(repeat_count):
            before = stream.cursor
            self._parse_node(
                child_id,
                stream,
                context,
                evaluator,
                parent_id=parent_id,
                force_current_bit_start=True,
            )
            if stream.cursor <= before:
                raise NodeParseError(
                    int(child_id),
                    "repeat_count item consumed 0 bits (stuck)",
                    kind="structure",
                )

    def _mark_zero_width_node(
        self,
        node_id: Union[str, int],
        stream: BitStream,
        context: RuntimeContext,
        *,
        parent_id: Optional[Union[str, int]],
    ) -> None:
        node_id = _coerce_node_id(node_id)
        parent_id = _coerce_node_id(parent_id) if parent_id is not None else None
        context.bit_starts[node_id] = int(stream.cursor)
        context.sizes[node_id] = 0
        context.actual_sizes[node_id] = 0
        context.wire_sizes[node_id] = 0
        self._record_node_occurrence(
            context,
            node_id=node_id,
            parent_id=parent_id,
            bit_start=int(stream.cursor),
            wire_size_bits=0,
            content_size_bits=0,
            value=None,
        )

    def _is_node_active(self, node_id: int, context: RuntimeContext, evaluator: ExpressionEvaluator) -> bool:
        # Check if there are incoming 'condition_on' edges to this node
        # In the edges list, dst == node_id
        node_id = _coerce_node_id(node_id)

        conditions = [e for e in self.edges if _coerce_node_id(e.get('dst')) == node_id and e.get('rel') == 'condition_on']
        
        if not conditions:
            return True # No condition = always active (unless it's a variant without explicit edge? assume active)
        
        # Filter conditions by message_type first
        filtered = []
        for cond in conditions:
            edge_msg_type = str(cond.get("message_type") or "bidirectional").lower()
            if self.target_message_type:
                if edge_msg_type not in {"bidirectional", self.target_message_type}:
                    continue
            filtered.append(cond)

        # If we're targeting a specific message type and no conditions apply, mark inactive
        if self.target_message_type and not filtered:
            return False

        # Evaluate applicable conditions
        from collections import defaultdict
        conditions_by_src = defaultdict(list)
        for cond in filtered:
            conditions_by_src[_coerce_node_id(cond.get('src'))].append(cond)

        for src_id, group in conditions_by_src.items():
            # OR logic: match ANY edge in the group
            group_results = []
            for c in group:
                try:
                    formula = c.get("formula")
                    group_results.append(evaluator.evaluate(formula, node_id))
                except UnresolvedReferenceError:
                    # condition_on gates branch applicability.  In repeated or
                    # nested selector structures, a branch may reference a
                    # selector that is not on the current parse path; that makes
                    # the branch inactive, not malformed.  Keep malformed
                    # formulas as hard errors below.
                    target_node = self.nodes.get(node_id) or {}
                    target_type = str(target_node.get("node_type") or "").lower()
                    is_plain_leaf = target_type in LEAF_TYPES and not (target_node.get("children_ids") or [])
                    size_expr = str(target_node.get("size_bits") or "")
                    has_self_guarded_size = "?" in size_expr
                    group_results.append(True if (is_plain_leaf and has_self_guarded_size) else False)
                except Exception as exc:
                    strict = os.getenv("STEP2_TRAFFIC_STRICT_CONDITION_EVAL", "1").strip().lower() not in {
                        "0",
                        "false",
                        "no",
                        "off",
                    }
                    if strict:
                        raise NodeParseError(
                            int(node_id),
                            (
                                "condition_on formula failed to evaluate "
                                f"(edge {c.get('src')}->{c.get('dst')}, formula={c.get('formula')})"
                            ),
                            kind="semantics",
                            cause=exc,
                        ) from exc
                    group_results.append(False)

            if not any(group_results):
                return False  # This selector group failed

        return True

    @staticmethod
    def _name_tokens_from_text(text: Any) -> Tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9]+", str(text or "").replace("_", " ").lower()))

    def _node_name_tokens(self, node_id: Union[str, int]) -> Tuple[str, ...]:
        node = self.nodes.get(_coerce_node_id(node_id)) or {}
        return self._name_tokens_from_text(node.get("name"))

    @staticmethod
    def _is_leaf_like_node(node: Dict[str, Any]) -> bool:
        node_type = str(node.get("node_type") or "").lower()
        return node_type in LEAF_TYPES or not (node.get("children_ids") or [])

    def _leaf_descendant_token_sequences(
        self,
        node_id: Union[str, int],
        seen: Optional[Set[Union[str, int]]] = None,
    ) -> List[Tuple[str, ...]]:
        node_id = _coerce_node_id(node_id)
        if seen is None:
            seen = set()
        if node_id in seen:
            return []
        seen.add(node_id)

        node = self.nodes.get(node_id) or {}
        if not node:
            return []
        if self._is_leaf_like_node(node):
            tokens = self._name_tokens_from_text(node.get("name"))
            return [tokens] if tokens else []

        out: List[Tuple[str, ...]] = []
        for child_id in node.get("children_ids") or []:
            out.extend(self._leaf_descendant_token_sequences(child_id, seen))
        return out

    @staticmethod
    def _tokens_end_with(tokens: Tuple[str, ...], suffix: Tuple[str, ...]) -> bool:
        return bool(suffix) and len(tokens) >= len(suffix) and tokens[-len(suffix) :] == suffix

    def _matching_prior_leaf_suffix(
        self,
        tokens: Tuple[str, ...],
        prior_child_ids: List[Union[str, int]],
    ) -> Optional[Tuple[str, ...]]:
        for prior_id in prior_child_ids:
            for prior_tokens in self._leaf_descendant_token_sequences(prior_id):
                if self._tokens_end_with(tokens, prior_tokens):
                    return prior_tokens
        return None

    @staticmethod
    def _expr_references_any_node(expr: Any, node_ids: Set[Union[str, int]]) -> bool:
        if expr is None or not node_ids:
            return False
        text = str(expr)
        for node_id in node_ids:
            escaped = re.escape(str(node_id))
            if re.search(rf"(?<![\w-]){escaped}\s*\.\s*(?:bit_start|size_bits)", text):
                return True
            if re.search(rf"\bval\(\s*{escaped}\s*\)", text):
                return True
        return False

    def _filter_redundant_alias_children(
        self,
        parent_id: Union[str, int],
        children_ids: List[Union[str, int]],
    ) -> List[Union[str, int]]:
        """Drop generated prefixed alias leaves when canonical fields already exist.

        LLM trees sometimes keep both a canonical field (e.g., `Version`) and a later
        display-style alias (e.g., `PFCP Common Header Version`) under the same
        container. The alias is redundant and can carry stale bit_start chains that
        break parsing. This filter is structural: it only removes leaf aliases whose
        name is a suffix match for an earlier sibling/descendant leaf.
        """
        parent_tokens = self._node_name_tokens(parent_id)
        parent_markers = {tok for tok in parent_tokens if tok not in {"root", "protocol", "packet", "message"}}
        kept: List[Union[str, int]] = []
        prior: List[Union[str, int]] = []
        skipped: Set[Union[str, int]] = set()

        for raw_child_id in children_ids or []:
            child_id = _coerce_node_id(raw_child_id)
            child = self.nodes.get(child_id) or {}
            if not child:
                continue

            if self._is_leaf_like_node(child):
                tokens = self._name_tokens_from_text(child.get("name"))
                matched_suffix = self._matching_prior_leaf_suffix(tokens, prior)
                has_parent_marker = bool(parent_markers.intersection(tokens))
                has_prefixed_duplicate_name = bool(
                    matched_suffix
                    and len(tokens) > len(matched_suffix)
                    and has_parent_marker
                )
                chained_to_skipped_alias = bool(
                    self._expr_references_any_node(child.get("bit_start"), skipped)
                    or self._expr_references_any_node(child.get("size_bits"), skipped)
                )
                if has_prefixed_duplicate_name or chained_to_skipped_alias:
                    skipped.add(child_id)
                    logger.debug(
                        "Skipping redundant alias child %s under parent %s (matched_suffix=%s)",
                        child_id,
                        parent_id,
                        matched_suffix,
                    )
                    continue

            kept.append(child_id)
            prior.append(child_id)

        return kept

    def _compute_declared_size_bits(
        self,
        node_id: Union[str, int],
        parent_id: Optional[Union[str, int]],
        stream: BitStream,
        context: RuntimeContext,
        evaluator: ExpressionEvaluator,
    ) -> NodeSizeInfo:
        node_id = _coerce_node_id(node_id)
        parent_id = _coerce_node_id(parent_id) if parent_id is not None else None
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found while computing size_bits")

        size_expr = node.get("size_bits")

        # Root: fallback to total_bits when unspecified or explicitly variable/unknown
        if parent_id is None or node_id == _coerce_node_id(self.root_id):
            # Prefer explicit numeric/expressions first
            evaluated = self._eval_size_expr(size_expr, evaluator, node_id)
            if evaluated is not None and evaluated >= 0:
                return NodeSizeInfo(True, evaluated)
            if size_expr is None or (isinstance(size_expr, str) and size_expr.strip().lower() in {"variable", "unknown", "dynamic", ""}):
                return NodeSizeInfo(True, stream.total_bits)

        # Non-root path
        evaluated = self._eval_size_expr(size_expr, evaluator, node_id)
        if evaluated is not None and evaluated >= 0:
            return NodeSizeInfo(True, evaluated)

        # length_of edges
        for edge in self.edges:
            if edge.get("rel") != "length_of":
                continue
            dst = _coerce_node_id(edge.get("dst"))
            if dst != node_id:
                continue
            formula = edge.get("formula")
            if not formula:
                continue
            val = self._eval_size_expr(formula, evaluator, edge.get("src"))
            if val is not None and val >= 0:
                return NodeSizeInfo(True, val)

        # Childless structural nodes under a declared-size parent are usually uninterpreted payload
        # alternatives. Consume the parent's remaining span instead of treating them as zero-width.
        if parent_id is not None and not (node.get("children_ids") or []):
            parent_start = context.bit_starts.get(parent_id)
            parent_size = context.sizes.get(parent_id)
            if parent_start is not None and parent_size is not None:
                try:
                    if int(parent_size) <= 0:
                        return NodeSizeInfo(False, None)
                    parent_end = int(parent_start) + int(parent_size)
                    remaining = parent_end - int(stream.cursor)
                    if remaining > 0:
                        return NodeSizeInfo(True, remaining)
                except Exception:
                    pass

        return NodeSizeInfo(False, None)

    def _eval_size_expr(
        self,
        expr: Any,
        evaluator: ExpressionEvaluator,
        current_node_id: Optional[Union[str, int]] = None,
    ) -> Optional[int]:
        if expr is None:
            return None
        if isinstance(expr, (int, float)) and not isinstance(expr, bool):
            try:
                return int(expr)
            except Exception:
                return None
        if isinstance(expr, str):
            normalized = expr.strip()
            if not normalized:
                return None
            if normalized.lower() in {"variable", "unknown", "dynamic"}:
                return None
            if normalized.isdigit() or (normalized.startswith("-") and normalized[1:].isdigit()):
                try:
                    return int(normalized)
                except Exception:
                    return None
            try:
                raw_val = evaluator.evaluate(normalized, current_node_id)
                if isinstance(raw_val, bool):
                    raw_val = int(raw_val)
                return int(raw_val)
            except UnresolvedReferenceError:
                return None
            except Exception:
                logger.debug("Failed to evaluate size expr '%s' for node %s", expr, current_node_id, exc_info=True)
                return None
        return None

    def _find_selector_for_variant(self, variant_id: Union[str, int]) -> Optional[Union[str, int]]:
        """Return selector node_id that controls this variant via condition_on edge, if any."""
        for e in self.edges:
            if e.get('rel') == 'condition_on' and _coerce_node_id(e.get('dst')) == _coerce_node_id(variant_id):
                return _coerce_node_id(e.get('src'))
        return None

    def _validate_length_edges(self, context: RuntimeContext, evaluator: ExpressionEvaluator) -> List[LengthMismatchDetail]:
        """Verify length_of edges match actual parsed size.

        Returns only fatal mismatches (wire/overflow). Coverage gaps are recorded
        on the context for higher-level handling but do not fail the parse.
        """
        mismatches: List[LengthMismatchDetail] = []
        context.length_gaps = []

        def _lookup(map_obj: Dict[Any, Any], key: Any) -> Optional[Any]:
            if key in map_obj:
                return map_obj.get(key)
            skey = str(key)
            if skey in map_obj:
                return map_obj.get(skey)
            return None

        def _estimate_from_intervals(dst_id: Union[str, int]) -> Optional[int]:
            starts = getattr(context, "bit_starts", {}) or {}
            declared_sizes = getattr(context, "sizes", {}) or {}
            actual_sizes = getattr(context, "actual_sizes", {}) or {}
            start_bit = _lookup(starts, dst_id)
            if start_bit is None:
                return None
            max_end = 0
            try:
                for nid, s in starts.items():
                    if s is None:
                        continue
                    size_val = _lookup(actual_sizes, nid)
                    if size_val is None:
                        size_val = _lookup(declared_sizes, nid)
                    if size_val is None:
                        continue
                    candidate = int(s) + int(size_val)
                    if candidate > max_end:
                        max_end = candidate
            except Exception:
                return None
            if max_end and max_end >= int(start_bit):
                return max_end - int(start_bit)
            return None

        def _node_occurrences(node_id: Union[str, int]) -> List[Dict[str, Any]]:
            node_id_norm = _coerce_node_id(node_id)
            out: List[Dict[str, Any]] = []
            for occ in getattr(context, "node_spans", []) or []:
                if _coerce_node_id(occ.get("node_id")) == node_id_norm:
                    out.append(occ)
            return sorted(out, key=lambda item: int(item.get("bit_start", 0) or 0))

        def _to_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except Exception:
                return None

        def _evaluate_expected_for_occurrence(
            formula: Any,
            current_node_id: Union[str, int],
            dst_start: int,
        ) -> int:
            # Reconstruct the latest parsed value/span state before this dst
            # occurrence. This avoids validating repeated TLV/option templates
            # with values overwritten by a later occurrence of the same node id.
            local_context = copy.deepcopy(context)
            local_context.values = copy.deepcopy(getattr(context, "values", {}) or {})
            local_context.bit_starts = copy.deepcopy(getattr(context, "bit_starts", {}) or {})
            local_context.sizes = copy.deepcopy(getattr(context, "sizes", {}) or {})
            local_context.actual_sizes = copy.deepcopy(getattr(context, "actual_sizes", {}) or {})
            local_context.wire_sizes = copy.deepcopy(getattr(context, "wire_sizes", {}) or {})

            for occ in sorted(getattr(context, "node_spans", []) or [], key=lambda item: int(item.get("bit_start", 0) or 0)):
                try:
                    if int(occ.get("bit_start", 0)) > int(dst_start):
                        continue
                    occ_id = _coerce_node_id(occ.get("node_id"))
                    if "value" in occ:
                        local_context.values[occ_id] = occ.get("value")
                    if "bit_start" in occ:
                        local_context.bit_starts[occ_id] = int(occ.get("bit_start"))
                    if "wire_size_bits" in occ:
                        local_context.sizes[occ_id] = int(occ.get("wire_size_bits"))
                        local_context.wire_sizes[occ_id] = int(occ.get("wire_size_bits"))
                    if "content_size_bits" in occ:
                        local_context.actual_sizes[occ_id] = int(occ.get("content_size_bits"))
                except Exception:
                    continue

            expected = ExpressionEvaluator(local_context, node_types=self.node_types).evaluate(
                formula,
                current_node_id=current_node_id,
            )
            if isinstance(expected, bool):
                expected = int(expected)
            return int(expected)

        def _append_length_check(
            *,
            src: Union[str, int],
            dst: Union[str, int],
            expected_bits: int,
            content_bits: Optional[int],
            wire_bits: Optional[int],
            formula: Any,
            message_type: Optional[str],
        ) -> None:
            if expected_bits < 0:
                return
            if wire_bits is None:
                wire_bits = content_bits
            if content_bits is None:
                content_bits = wire_bits

            if content_bits is not None and int(content_bits) > int(expected_bits):
                mismatches.append(
                    LengthMismatchDetail(
                        src=int(src),
                        dst=int(dst),
                        expected_bits=int(expected_bits),
                        actual_bits=int(content_bits),
                        content_bits=int(content_bits),
                        wire_bits=int(wire_bits) if wire_bits is not None else int(content_bits),
                        mismatch_kind="overflow",
                        formula=str(formula),
                        message_type=message_type,
                    )
                )
                return

            if wire_bits is not None and int(wire_bits) != int(expected_bits):
                mismatches.append(
                    LengthMismatchDetail(
                        src=int(src),
                        dst=int(dst),
                        expected_bits=int(expected_bits),
                        actual_bits=int(wire_bits),
                        content_bits=int(content_bits) if content_bits is not None else None,
                        wire_bits=int(wire_bits),
                        mismatch_kind="wire_mismatch",
                        formula=str(formula),
                        message_type=message_type,
                    )
                )
                return

            if (
                wire_bits is not None
                and int(wire_bits) == int(expected_bits)
                and content_bits is not None
                and int(content_bits) < int(expected_bits)
            ):
                context.length_gaps.append(
                    LengthMismatchDetail(
                        src=int(src),
                        dst=int(dst),
                        expected_bits=int(expected_bits),
                        actual_bits=int(content_bits),
                        content_bits=int(content_bits),
                        wire_bits=int(wire_bits),
                        mismatch_kind="coverage_gap",
                        formula=str(formula),
                        message_type=message_type,
                    )
                )

        for edge in self.edges:
            if edge.get("rel") != "length_of":
                continue
            src = _coerce_node_id(edge.get("src"))
            dst = _coerce_node_id(edge.get("dst"))
            formula = edge.get("formula")
            if formula is None:
                continue

            src_occurrences = _node_occurrences(src)
            dst_occurrences = _node_occurrences(dst)
            if src_occurrences and dst_occurrences and (len(src_occurrences) > 1 or len(dst_occurrences) > 1):
                for dst_occ in dst_occurrences:
                    try:
                        dst_start = int(dst_occ.get("bit_start", 0))
                        expected_bits = _evaluate_expected_for_occurrence(formula, src, dst_start)
                    except Exception:
                        continue
                    _append_length_check(
                        src=src,
                        dst=dst,
                        expected_bits=expected_bits,
                        content_bits=_to_int(dst_occ.get("content_size_bits")),
                        wire_bits=_to_int(dst_occ.get("wire_size_bits")),
                        formula=formula,
                        message_type=edge.get("message_type"),
                    )
                continue

            content_bits = _lookup(getattr(context, "actual_sizes", {}) or {}, dst)
            wire_bits = _lookup(getattr(context, "wire_sizes", {}) or {}, dst)
            declared_bits = _lookup(getattr(context, "sizes", {}) or {}, dst)

            try:
                expected = evaluator.evaluate(formula, current_node_id=src)
                if isinstance(expected, bool):
                    expected = int(expected)
                expected_bits = int(expected)
            except Exception:
                continue
            # Guard: some trees may contain inconsistent length_of formulas (or formulas
            # that are only meaningful under a different extraction context). If the
            # expected size is negative, treat the constraint as non-applicable rather
            # than failing the parse.
            if expected_bits < 0:
                continue

            if wire_bits is None:
                wire_bits = declared_bits if declared_bits is not None else _estimate_from_intervals(dst)
            if content_bits is None:
                content_bits = wire_bits if wire_bits is not None else declared_bits
            _append_length_check(
                src=src,
                dst=dst,
                expected_bits=expected_bits,
                content_bits=_to_int(content_bits),
                wire_bits=_to_int(wire_bits),
                formula=formula,
                message_type=edge.get("message_type"),
            )

        return mismatches
