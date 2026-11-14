"""Review specification coverage in a PKG.

The model compares each specification fragment with the current graph and
reports missing, incomplete, or incorrect layout, decoding, and validity
knowledge. Each finding includes a supporting quotation. Repairs use the
Stage B patch interface and are accepted after preservation, static,
traffic, and source-coverage checks. Review repeats until a complete pass
finds no new items or the configured budget is exhausted."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

KINDS = ("layout", "decoding", "validity")
STATUSES = ("missing", "incomplete", "incorrect")
VERDICTS = ("represented", "partial", "absent")


class BudgetExhausted(RuntimeError):
    """The review spent its call or cost budget."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except Exception:
        return default


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def tree_payload(tree: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(tree, dict) and isinstance(tree.get("protocol_tree"), dict):
        return tree["protocol_tree"]
    return tree


# ---------------------------------------------------------------------------
# Specification fragments and PKG rendering


def load_sections(path: Path) -> List[Any]:
    """Sections as Step 2 reads them: a list, or the ``sections`` array of an object."""

    from pargen.protocol_knowledge.repair.step2_fix import _coerce_sections

    return list(_coerce_sections(json.loads(Path(path).read_text(encoding="utf-8"))))


def section_text(section: Any) -> str:
    if not isinstance(section, dict):
        return str(section or "")
    title = " ".join(str(section.get(key) or "") for key in ("title",)).strip()
    content = str(section.get("content") or section.get("summary") or "")
    return f"{title}\n{content}".strip() if title and not content.lstrip().startswith(title) else content


def build_fragments(sections: Sequence[Any], max_chars: int = 8000) -> List[Dict[str, Any]]:
    """Cut the specification into fragments of at most ``max_chars``.

    Short consecutive sections share a fragment; a long section is split at
    blank lines (never inside a paragraph or diagram).  Every piece keeps an
    ``[S<index>]`` header so findings name their section.
    """

    pieces: List[Tuple[int, str]] = []
    for index, section in enumerate(sections):
        text = section_text(section)
        if len(text) <= max_chars:
            pieces.append((index, text))
            continue
        chunk = ""
        for paragraph in re.split(r"\n\s*\n", text):
            candidate = f"{chunk}\n\n{paragraph}" if chunk else paragraph
            if chunk and len(candidate) > max_chars:
                pieces.append((index, chunk))
                chunk = paragraph
            else:
                chunk = candidate
        if chunk.strip():
            pieces.append((index, chunk))

    groups: List[List[Tuple[int, str]]] = []
    current: List[Tuple[int, str]] = []
    size = 0
    for index, text in pieces:
        if current and size + len(text) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append((index, text))
        size += len(text)
    if current:
        groups.append(current)

    return [
        {
            "id": f"F{number}",
            "sections": sorted({index for index, _ in group}),
            "text": "\n\n".join(f"[S{index}]\n{text}" for index, text in group),
        }
        for number, group in enumerate(groups, start=1)
    ]


def _short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_pkg(tree: Dict[str, Any]) -> str:
    payload = tree_payload(tree)
    lines = ["Nodes: id | name | type | parent | message_type | bit_start | size_bits | data_type byte_order | "
             "constraints | source sections | description"]
    for node in payload.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        constraints = [_short(item, 320) for item in (node.get("constraints") or [])]
        lines.append(
            " | ".join(
                [
                    str(node.get("node_id")),
                    str(node.get("name") or ""),
                    str(node.get("node_type") or ""),
                    f"parent={node.get('parent_id')}",
                    str(node.get("message_type") or ""),
                    f"bit_start={node.get('bit_start')}",
                    f"size_bits={node.get('size_bits')}",
                    f"{node.get('data_type') or ''} {node.get('byte_order') or ''}".strip(),
                    f"constraints={constraints}",
                    f"src={node.get('source_section_ids') or []}",
                    _short(node.get("description") or "", 120),
                ]
            )
        )
    lines.append("Edges: src -> dst | rel | formula | message_type")
    for edge in payload.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        lines.append(
            f"{edge.get('src')} -> {edge.get('dst')} | {edge.get('rel')} | "
            f"{_short(edge.get('formula') or '', 240)} | {edge.get('message_type') or ''}"
        )
    return "\n".join(lines)


def _nodes_citing(tree: Dict[str, Any], section_ids: Sequence[int]) -> List[Any]:
    wanted = {int(value) for value in section_ids}
    cited = []
    for node in tree_payload(tree).get("nodes") or []:
        refs = node.get("source_section_ids") if isinstance(node, dict) else None
        if not isinstance(refs, (list, tuple)):
            continue
        values = set()
        for ref in refs:
            try:
                values.add(int(ref))
            except Exception:
                continue
        if values & wanted:
            cited.append(node.get("node_id"))
    return cited


# ---------------------------------------------------------------------------
# LLM transport with a single-file cache and a cost budget


class ReviewLLM:
    """Stage-B DeepSeek transport (``EnhancedPureAIAgent``) plus a JSONL call log.

    The log doubles as a replay cache keyed by the request, so an interrupted
    review resumes without paying twice.  One file per run (inode budget).
    """

    def __init__(
        self,
        *,
        log_path: Path,
        provider: str = "deepseek",
        model: Optional[str] = None,
        max_calls: int = 40,
        budget_usd: float = 1.0,
        price_in_per_mtok: float = 0.30,
        price_out_per_mtok: float = 1.20,
        transport: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.max_calls = max_calls
        self.budget_usd = budget_usd
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.calls = 0
        self.cached_calls = 0
        self.spent_usd = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self._agent = None
        self._transport = transport
        self.model = model
        if transport is None:
            from pargen.protocol_knowledge.repair.agent import EnhancedPureAIAgent

            self._agent = EnhancedPureAIAgent(provider, cache_dir=self.log_path.parent / "agent")
            self.model = model or self._agent.default_model
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if record.get("ok") and record.get("key"):
                    self._cache[record["key"]] = record

    def _clean(self, raw: str) -> str:
        if self._agent is not None:
            return self._agent._clean_raw_response(raw)
        return (raw or "").strip()

    def _cost(self, usage: Dict[str, Any]) -> float:
        if isinstance(usage.get("cost"), (int, float)):
            return float(usage["cost"])
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        return prompt * self.price_in / 1e6 + completion * self.price_out / 1e6

    def call(self, stage: str, messages: List[Dict[str, str]], *, max_tokens: int, effort: Optional[str]) -> Any:
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": effort} if effort else None,
        }
        key = _sha(request)
        cached = self._cache.get(key)
        if cached is not None:
            self.cached_calls += 1
            return cached.get("parsed")
        if self.calls >= self.max_calls:
            raise BudgetExhausted(f"call budget {self.max_calls} spent")
        if self.spent_usd >= self.budget_usd:
            raise BudgetExhausted(f"cost budget ${self.budget_usd:.2f} spent (${self.spent_usd:.3f})")
        payload = {key_: value for key_, value in request.items() if value is not None}
        payload.update(stream=False, usage={"include": True})
        self.calls += 1
        started = time.time()
        record: Dict[str, Any] = {"key": key, "stage": stage, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ok": False}
        try:
            if self._transport is not None:
                result = self._transport(payload)
            else:
                result = self._agent._call_api_with_retry(payload)  # type: ignore[union-attr]
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", seconds=round(time.time() - started, 1))
            self._append(record)
            logger.warning("Source review %s call failed: %s", stage, exc)
            return None
        usage = result.get("usage") or {}
        cost = self._cost(usage)
        self.spent_usd += cost
        self.tokens_in += int(usage.get("prompt_tokens") or 0)
        self.tokens_out += int(usage.get("completion_tokens") or 0)
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw = message.get("content") if isinstance(message, dict) else str(message)
        clean = self._clean(raw or "")
        parsed: Any = None
        try:
            parsed = json.loads(clean) if clean else None
        except Exception:
            parsed = None
        record.update(
            ok=parsed is not None,
            finish_reason=choice.get("finish_reason"),
            upstream_provider=result.get("provider"),
            usage=usage,
            cost_usd=round(cost, 6),
            seconds=round(time.time() - started, 1),
            parsed=parsed,
            raw=None if parsed is not None else (raw or "")[:2000],
        )
        self._append(record)
        if parsed is not None:
            self._cache[key] = record
        return parsed

    def _append(self, record: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stats(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "spent_usd": round(self.spent_usd, 4),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "max_calls": self.max_calls,
            "budget_usd": self.budget_usd,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Reviewer prompts


DETECT_SYSTEM = """You review a protocol knowledge graph (PKG) against the protocol specification it was built from. A parser will be generated from the PKG, so knowledge the specification states but the PKG lacks will be missing from the parser.

You receive one specification fragment and the current PKG. Report the parser-relevant knowledge that the fragment states and that the PKG does not represent, represents only in part, or represents incorrectly.

Parser-relevant knowledge:
- layout: a message element (header, field, option, record, variant body, padding) and its position, order, size or length rule, containment, repetition, or presence condition;
- decoding: how an element's bits are read: data type, width, bit or byte order, units or scaling, or a code value that selects how later bytes are parsed;
- validity: rules that make a single message invalid when broken: fixed, reserved or allowed values, value ranges, length or count relations, and conditions linking one element's value or presence to another's. Only rules the fragment states as mandatory for the message itself (MUST, SHALL, REQUIRED, MUST NOT, or a definition of what the element contains).

Do not report:
- recommendations (SHOULD, MAY, RECOMMENDED, "normally", "usually") as validity rules;
- requirements on an implementation rather than on a message: what an endpoint must be able to receive, store or process, buffer sizes, timers, retransmission, state machines, routing or processing decisions, how a sender chooses values;
- labels or meanings of code values that change neither how later bytes are parsed nor which values are allowed;
- explanations, examples, history, or pure naming differences;
- anything the PKG already encodes in an equivalent form. Names, units and grouping may differ; a node, a size expression, a data type, a node constraint (enum:, range:, formula:, constraint_ir:), an edge (length_of, condition_on, repeat_count) or a variant condition all count as representing the knowledge. A node description alone does not.

Use only the fragment and the PKG, never your own memory of the protocol. Each item must quote, verbatim, the span of the fragment that states it."""


VERIFY_SYSTEM = """You check whether given pieces of protocol knowledge are represented in a protocol knowledge graph (PKG). An item is represented when the PKG encodes it in a form a parser generator can use: nodes with the right position, size and data type, node constraints (enum:, range:, formula:, constraint_ir:), edges (length_of, condition_on, repeat_count) or variant conditions. Names, units and grouping may differ. A node description, or a node of zero size that carries no content, does not represent anything. An encoding that contradicts the item does not represent it."""


def _detect_messages(protocol: str, fragment: Dict[str, Any], tree: Dict[str, Any], max_items: int) -> List[Dict[str, str]]:
    payload = tree_payload(tree)
    cited = _nodes_citing(tree, fragment["sections"])
    user = (
        f"Protocol: {protocol}\n"
        f"Specification fragment {fragment['id']} (sections {', '.join('S%d' % s for s in fragment['sections'])}):\n"
        f"<<<\n{fragment['text']}\n>>>\n\n"
        f"Current PKG ({len(payload.get('nodes') or [])} nodes, {len(payload.get('edges') or [])} edges).\n"
        f"{render_pkg(tree)}\n\n"
        f"PKG nodes whose provenance cites these sections: {cited or 'none'}\n\n"
        "Answer with JSON only:\n"
        '{"items": [{"section": <section index, the number in [S#]>, "kind": "layout" | "decoding" | "validity", '
        '"status": "missing" | "incomplete" | "incorrect", "statement": "<one precise sentence stating the knowledge '
        'with the specification\'s element names and values>", "quote": "<verbatim span copied from the fragment>", '
        '"pkg_nodes": [<ids of related PKG nodes, if any>], "why": "<what the PKG lacks or gets wrong>"}]}\n'
        f"List at most {max_items} items, the most consequential for parsing and validation first. "
        'Return {"items": []} when the PKG already covers the fragment.'
    )
    return [{"role": "system", "content": DETECT_SYSTEM}, {"role": "user", "content": user}]


def _verify_messages(findings: Sequence["Finding"], tree: Dict[str, Any]) -> List[Dict[str, str]]:
    items = "\n".join(
        f'- {item.id} [{item.kind}] {item.statement} (specification: "{_short(item.quote, 400)}")' for item in findings
    )
    user = (
        f"Items:\n{items}\n\nPKG:\n{render_pkg(tree)}\n\n"
        "Answer with JSON only:\n"
        '{"results": [{"id": "<item id>", "verdict": "represented" | "partial" | "absent", '
        '"evidence": [<node ids>], "reason": "<one sentence>"}]}'
    )
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]


@dataclass
class Finding:
    id: str
    section: int
    kind: str
    status: str
    statement: str
    quote: str
    pkg_nodes: List[Any] = field(default_factory=list)
    why: str = ""
    fragment: str = ""
    pass_index: int = 0


def _quote_in_text(quote: str, text: str) -> bool:
    quote_norm = _norm(quote).strip(" .\"'")
    if len(quote_norm) < 8:
        return False
    text_norm = _norm(text)
    if quote_norm in text_norm:
        return True
    # An elided quote ("A ... B") counts when every elided part is verbatim.
    parts = [part.strip(" .\"'") for part in re.split(r"\.\.\.|…", quote_norm)]
    parts = [part for part in parts if part]
    return len(parts) > 1 and all(len(part) >= 8 and part in text_norm for part in parts)


def parse_findings(
    parsed: Any,
    fragment: Dict[str, Any],
    sections: Sequence[Any],
    *,
    pass_index: int,
    max_items: int,
) -> Tuple[List[Finding], List[Dict[str, Any]]]:
    """Validate the reviewer's items; returns (findings, rejected items with reasons)."""

    items = parsed.get("items") if isinstance(parsed, dict) else None
    findings: List[Finding] = []
    rejected: List[Dict[str, Any]] = []
    for item in (items or [])[: max_items]:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or "").strip()
        quote = str(item.get("quote") or "").strip()
        try:
            section = int(str(item.get("section")).lstrip("Ss"))
        except Exception:
            section = fragment["sections"][0] if len(fragment["sections"]) == 1 else -1
        if section not in fragment["sections"]:
            # Fall back to the fragment section whose text holds the quote.
            section = next(
                (index for index in fragment["sections"] if _quote_in_text(quote, section_text(sections[index]))),
                -1,
            )
        reason = None
        if not statement:
            reason = "no statement"
        elif section < 0 or not _quote_in_text(quote, section_text(sections[section])):
            reason = "quote not found verbatim in the fragment"
        if reason:
            rejected.append({"item": item, "reason": reason, "fragment": fragment["id"]})
            continue
        kind = str(item.get("kind") or "").strip().lower()
        status = str(item.get("status") or "").strip().lower()
        findings.append(
            Finding(
                id="sr_" + _sha([section, _norm(quote), _norm(statement)])[:12],
                section=section,
                kind=kind if kind in KINDS else "validity",
                status=status if status in STATUSES else "missing",
                statement=statement,
                quote=quote,
                pkg_nodes=list(item.get("pkg_nodes") or [])[:8] if isinstance(item.get("pkg_nodes"), list) else [],
                why=str(item.get("why") or "").strip(),
                fragment=fragment["id"],
                pass_index=pass_index,
            )
        )
    return findings, rejected


def parse_verdicts(parsed: Any, findings: Sequence[Finding]) -> Dict[str, Dict[str, Any]]:
    wanted = {item.id for item in findings}
    verdicts: Dict[str, Dict[str, Any]] = {}
    for row in (parsed.get("results") if isinstance(parsed, dict) else None) or []:
        if not isinstance(row, dict) or str(row.get("id")) not in wanted:
            continue
        verdict = str(row.get("verdict") or "").strip().lower()
        verdicts[str(row["id"])] = {
            "verdict": verdict if verdict in VERDICTS else "absent",
            "evidence": row.get("evidence") or [],
            "reason": str(row.get("reason") or ""),
        }
    for item in findings:
        verdicts.setdefault(item.id, {"verdict": "absent", "evidence": [], "reason": "no verdict returned"})
    return verdicts


# ---------------------------------------------------------------------------
# Repair and acceptance gates


REPAIR_FEEDBACK = """SOURCE-COVERAGE REVIEW. A reviewer compared the specification sections below with the PKG and found knowledge the PKG does not represent (the PENDING ISSUES with ids sr_...). Write ONE patch that adds as many of these items as can be added safely:
- attach data types, sizes and constraints (enum:, range:, formula:, constraint_ir:) to the existing nodes the items concern;
- add a node only for an element the PKG lacks: give it parent_id, node_type, message_type, bit_start and size_bits consistent with its siblings, add it to the parent's children_ids, use node_id values starting at {next_id}, and set source_section_ids to the section_index values of the sections that state it;
- express relations between elements with edges (length_of, condition_on, repeat_count) or constraint_ir;
- do not remove, rename or re-parent existing nodes, and keep every existing constraint;
- leave out an item that cannot be represented without restructuring, and say why in validation_notes.
The observed traffic packets must still parse, and the patch must not introduce validation errors."""


def _repair_messages(
    tree: Dict[str, Any],
    sections: Sequence[Any],
    findings: Sequence[Finding],
    *,
    retry_feedback: Optional[str],
) -> List[Dict[str, str]]:
    from pargen.protocol_knowledge.repair.message import build_patch_refinement_message

    payload = tree_payload(tree)
    next_id = 1 + max([int(node.get("node_id")) for node in payload.get("nodes") or [] if str(node.get("node_id", "")).lstrip("-").isdigit()] or [0])
    wanted = sorted({item.section for item in findings})
    prompt_sections = []
    for index in wanted:
        section = copy.deepcopy(sections[index]) if isinstance(sections[index], dict) else {"content": str(sections[index])}
        section["section_index"] = index
        section["is_focused"] = True
        prompt_sections.append(section)
    issues = [
        {
            "id": item.id,
            "type": "COVERAGE",
            "severity": "ERROR",
            "description": (
                f"[{item.kind}/{item.status}] {item.statement} Specification (section_index {item.section}): "
                f"\"{_short(item.quote, 600)}\". PKG gap: {item.why}"
            ),
            "target": ", ".join(str(node) for node in item.pkg_nodes) or None,
        }
        for item in findings
    ]
    feedback = REPAIR_FEEDBACK.format(next_id=next_id)
    if retry_feedback:
        feedback += "\n\nThe previous patch for these items was rejected: " + retry_feedback
    return build_patch_refinement_message(
        payload,
        prompt_sections,
        0,
        len(prompt_sections),
        feedback,
        None,
        {"pending_issues": issues, "validator_errors": [], "validator_extras": []},
        mode="fix",
    )


def _as_patch(parsed: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None
    if isinstance(parsed.get("patch"), dict):
        parsed = parsed["patch"]
    keys = ("new_nodes", "node_updates", "new_edges", "edge_updates", "nodes_to_remove", "edge_removes")
    if not any(parsed.get(key) for key in keys):
        return None
    return parsed


def _constraint_set(node: Dict[str, Any]) -> set:
    return {_norm(item if isinstance(item, str) else json.dumps(item, sort_keys=True)) for item in node.get("constraints") or []}


def additive_gate(before: Dict[str, Any], patch: Dict[str, Any], after: Dict[str, Any]) -> Optional[str]:
    if patch.get("nodes_to_remove") or patch.get("edge_removes"):
        return "patch removes nodes or edges"
    before_nodes = {str(node.get("node_id")): node for node in tree_payload(before).get("nodes") or []}
    after_nodes = {str(node.get("node_id")): node for node in tree_payload(after).get("nodes") or []}
    missing = sorted(set(before_nodes) - set(after_nodes))
    if missing:
        return f"nodes lost: {missing[:8]}"
    for node_id, node in before_nodes.items():
        if not _constraint_set(node) <= _constraint_set(after_nodes[node_id]):
            return f"node {node_id} lost a constraint"
        if node.get("name") != after_nodes[node_id].get("name") or str(node.get("parent_id")) != str(after_nodes[node_id].get("parent_id")):
            return f"node {node_id} renamed or re-parented"
    if len(tree_payload(after).get("edges") or []) < len(tree_payload(before).get("edges") or []):
        return "edges lost"
    return None


def _error_ids(tree: Dict[str, Any]) -> Tuple[set, set]:
    from pargen.protocol_knowledge.repair.refinement import run_full_validation
    from pargen.protocol_knowledge.validation.syntax_validator import Severity, is_hard_error

    report = run_full_validation(tree_payload(tree))
    errors, hard = set(), set()
    for issue_id, issue in (getattr(report, "issues", {}) or {}).items():
        if getattr(issue, "severity", None) == Severity.ERROR:
            errors.add(str(issue_id))
            if is_hard_error(issue):
                hard.add(str(issue_id))
    return errors, hard


def anchor_records(tree: Dict[str, Any], traffic_path: Optional[Path], max_packets: int) -> List[Dict[str, Any]]:
    if traffic_path is None or not Path(traffic_path).exists():
        return []
    from pargen.protocol_knowledge.construction.step2_format_graph_builder import _build_traffic_anchor_records

    return _build_traffic_anchor_records(tree_payload(tree), traffic_path, max_packets=max_packets)


def anchor_gate(before_records: Sequence[Dict[str, Any]], after_records: Sequence[Dict[str, Any]]) -> Optional[str]:
    after = {int(record.get("packet_index", idx)): record for idx, record in enumerate(after_records)}
    for idx, record in enumerate(before_records):
        if not record.get("success"):
            continue
        packet = int(record.get("packet_index", idx))
        if not (after.get(packet) or {}).get("success"):
            return f"anchor packet {packet} no longer parses: {str((after.get(packet) or {}).get('error'))[:300]}"
    return None


def _record_provenance(before: Dict[str, Any], after: Dict[str, Any], sections_of_group: Sequence[int]) -> None:
    """New nodes and nodes that gained constraints cite the reviewed sections (ρ)."""

    before_nodes = {str(node.get("node_id")): node for node in tree_payload(before).get("nodes") or []}
    for node in tree_payload(after).get("nodes") or []:
        node_id = str(node.get("node_id"))
        old = before_nodes.get(node_id)
        gained = old is None or not _constraint_set(node) <= _constraint_set(old)
        if not gained:
            continue
        refs = node.get("source_section_ids")
        refs = list(refs) if isinstance(refs, list) else []
        for index in sections_of_group:
            if index not in refs:
                refs.append(index)
        node["source_section_ids"] = refs


# ---------------------------------------------------------------------------
# The review loop


def run_source_review(
    tree: Dict[str, Any],
    sections: Sequence[Any],
    *,
    llm: ReviewLLM,
    protocol: str,
    traffic_path: Optional[Path] = None,
    max_passes: Optional[int] = None,
    fragment_chars: Optional[int] = None,
    group_size: Optional[int] = None,
    attempts_per_group: Optional[int] = None,
    max_items_per_fragment: Optional[int] = None,
    detect_max_tokens: Optional[int] = None,
    repair_max_tokens: Optional[int] = None,
    detect_effort: Optional[str] = None,
    verify_effort: Optional[str] = None,
    repair_effort: Optional[str] = None,
    anchor_packets: int = 64,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Review ``tree`` against ``sections``; returns (reviewed tree, summary)."""

    from pargen.protocol_knowledge.repair.refinement import apply_patch

    max_passes = max_passes or _env_int("STEP2_SOURCE_REVIEW_MAX_PASSES", 3)
    fragment_chars = fragment_chars or _env_int("STEP2_SOURCE_REVIEW_FRAGMENT_CHARS", 8000)
    group_size = group_size or _env_int("STEP2_SOURCE_REVIEW_GROUP_SIZE", 6)
    attempts_per_group = attempts_per_group or _env_int("STEP2_SOURCE_REVIEW_ATTEMPTS", 2)
    max_items = max_items_per_fragment or _env_int("STEP2_SOURCE_REVIEW_MAX_ITEMS", 12)
    detect_max_tokens = detect_max_tokens or _env_int("STEP2_SOURCE_REVIEW_MAX_TOKENS", 32000)
    repair_max_tokens = repair_max_tokens or _env_int("STEP2_SOURCE_REVIEW_REPAIR_MAX_TOKENS", 48000)
    detect_effort = detect_effort or os.getenv("STEP2_SOURCE_REVIEW_DETECT_EFFORT", "medium")
    verify_effort = verify_effort or os.getenv("STEP2_SOURCE_REVIEW_VERIFY_EFFORT", "low")
    repair_effort = repair_effort or os.getenv("STEP2_SOURCE_REVIEW_REPAIR_EFFORT", "medium")

    wrapper = copy.deepcopy(tree)
    current = copy.deepcopy(tree_payload(tree))
    fragments = build_fragments(sections, fragment_chars)
    summary: Dict[str, Any] = {
        "detector": "llm_judge",
        "fragments": [{"id": frag["id"], "sections": frag["sections"], "chars": len(frag["text"])} for frag in fragments],
        "passes": [],
        "findings": {},
        "rejected_findings": [],
        "groups": [],
        "resolved": [],
        "stop_reason": None,
        "settings": {
            "max_passes": max_passes,
            "fragment_chars": fragment_chars,
            "group_size": group_size,
            "attempts_per_group": attempts_per_group,
            "max_items_per_fragment": max_items,
            "detect_max_tokens": detect_max_tokens,
            "repair_max_tokens": repair_max_tokens,
            "effort": {"detect": detect_effort, "verify": verify_effort, "repair": repair_effort},
            "anchor_packets": anchor_packets,
        },
    }
    # A finding is keyed by its section and quote.  It is closed once resolved,
    # or after it was targeted in ``attempts_per_group`` groups without success.
    closed: set = set()
    tries: Dict[Tuple[int, str], int] = {}
    base_records = anchor_records(current, traffic_path, anchor_packets)
    current_errors = _error_ids(current)
    summary["anchors"] = {
        "traffic_path": str(traffic_path) if traffic_path else None,
        "parsed_before": sum(1 for record in base_records if record.get("success")),
        "total": len(base_records),
    }

    try:
        for pass_index in range(1, max_passes + 1):
            pass_record: Dict[str, Any] = {"pass": pass_index, "new_findings": 0, "accepted_groups": 0}
            summary["passes"].append(pass_record)
            new_findings: List[Finding] = []
            seen: set = set()
            for fragment in fragments:
                parsed = llm.call(
                    "detect",
                    _detect_messages(protocol, fragment, current, max_items),
                    max_tokens=detect_max_tokens,
                    effort=detect_effort,
                )
                if parsed is None:
                    pass_record.setdefault("failed_fragments", []).append(fragment["id"])
                    continue
                findings, rejected = parse_findings(parsed, fragment, sections, pass_index=pass_index, max_items=max_items)
                summary["rejected_findings"].extend(rejected)
                for item in findings:
                    key = (item.section, _norm(item.quote))
                    if key in closed or key in seen:
                        continue
                    seen.add(key)
                    new_findings.append(item)
                    summary["findings"][item.id] = asdict(item)
            pass_record["new_findings"] = len(new_findings)
            if not new_findings:
                summary["stop_reason"] = "complete pass found no new knowledge"
                break

            by_section: Dict[int, List[Finding]] = {}
            for item in new_findings:
                by_section.setdefault(item.section, []).append(item)
            groups: List[List[Finding]] = []
            for section in sorted(by_section):
                items = by_section[section]
                groups.extend(items[i : i + group_size] for i in range(0, len(items), group_size))

            for group in groups:
                group_record: Dict[str, Any] = {
                    "pass": pass_index,
                    "findings": [item.id for item in group],
                    "sections": sorted({item.section for item in group}),
                    "attempts": [],
                    "accepted": False,
                }
                summary["groups"].append(group_record)
                retry_feedback: Optional[str] = None
                for attempt in range(1, attempts_per_group + 1):
                    attempt_record: Dict[str, Any] = {"attempt": attempt}
                    group_record["attempts"].append(attempt_record)
                    parsed = llm.call(
                        "repair",
                        _repair_messages(current, sections, group, retry_feedback=retry_feedback),
                        max_tokens=repair_max_tokens,
                        effort=repair_effort,
                    )
                    patch = _as_patch(parsed)
                    if patch is None:
                        attempt_record["rejected"] = "no usable patch"
                        retry_feedback = "it contained no patch actions."
                        continue
                    attempt_record["patch_metadata"] = patch.get("patch_metadata")
                    attempt_record["actions"] = {
                        key: len(patch.get(key) or [])
                        for key in ("new_nodes", "node_updates", "new_edges", "edge_updates", "nodes_to_remove", "edge_removes")
                    }
                    try:
                        candidate = apply_patch(current, patch)
                    except Exception as exc:
                        attempt_record["rejected"] = f"apply failed: {type(exc).__name__}: {exc}"[:400]
                        retry_feedback = attempt_record["rejected"]
                        continue
                    _record_provenance(current, candidate, group_record["sections"])
                    reason = additive_gate(current, patch, candidate)
                    if reason is None:
                        before_errors, before_hard = current_errors
                        after_errors, after_hard = _error_ids(candidate)
                        introduced = sorted(after_errors - before_errors)
                        attempt_record["static"] = {
                            "errors_before": len(before_errors),
                            "errors_after": len(after_errors),
                            "introduced": introduced[:12],
                        }
                        if introduced or len(after_hard) > len(before_hard):
                            reason = f"static validation introduced errors: {introduced[:6]}"
                    if reason is None:
                        after_records = anchor_records(candidate, traffic_path, anchor_packets)
                        reason = anchor_gate(base_records, after_records)
                        attempt_record["anchors_parsed_after"] = sum(1 for record in after_records if record.get("success"))
                    if reason is None:
                        verdicts = parse_verdicts(
                            llm.call(
                                "verify",
                                _verify_messages(group, candidate),
                                max_tokens=detect_max_tokens,
                                effort=verify_effort,
                            ),
                            group,
                        )
                        attempt_record["verdicts"] = verdicts
                        resolved = [item.id for item in group if verdicts[item.id]["verdict"] == "represented"]
                        attempt_record["resolved"] = resolved
                        if not resolved:
                            reason = "judge found none of the targeted items represented after the patch"
                    if reason is not None:
                        attempt_record["rejected"] = reason
                        retry_feedback = reason
                        continue
                    current = candidate
                    current_errors = _error_ids(current)
                    base_records = anchor_records(current, traffic_path, anchor_packets) or base_records
                    group_record["accepted"] = True
                    pass_record["accepted_groups"] += 1
                    summary["resolved"].extend(attempt_record["resolved"])
                    break
                resolved_now = set(group_record["attempts"][-1].get("resolved") or []) if group_record["accepted"] else set()
                for item in group:
                    key = (item.section, _norm(item.quote))
                    tries[key] = tries.get(key, 0) + 1
                    if item.id in resolved_now or tries[key] >= attempts_per_group:
                        closed.add(key)
            if not pass_record["accepted_groups"]:
                summary["stop_reason"] = "no patch accepted in this pass"
                break
        else:
            summary["stop_reason"] = f"pass budget {max_passes} spent"
    except BudgetExhausted as exc:
        summary["stop_reason"] = f"budget: {exc}"

    summary["llm"] = llm.stats()
    summary["counts"] = {
        "findings": len(summary["findings"]),
        "resolved": len(set(summary["resolved"])),
        "groups": len(summary["groups"]),
        "accepted_groups": sum(1 for group in summary["groups"] if group["accepted"]),
        "nodes_before": len(tree_payload(tree).get("nodes") or []),
        "nodes_after": len(current.get("nodes") or []),
        "constraints_before": sum(len(node.get("constraints") or []) for node in tree_payload(tree).get("nodes") or []),
        "constraints_after": sum(len(node.get("constraints") or []) for node in current.get("nodes") or []),
        "edges_before": len(tree_payload(tree).get("edges") or []),
        "edges_after": len(current.get("edges") or []),
    }
    if isinstance(wrapper, dict) and isinstance(wrapper.get("protocol_tree"), dict):
        wrapper["protocol_tree"] = current
        return wrapper, summary
    return current, summary
