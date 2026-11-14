from __future__ import annotations

"""Prompt helpers for the new codegen-oriented Stage C."""

import json
import textwrap
from typing import Dict, List, Optional

from .backend_contract import backend_contract_prompt
from .constraint_contract import build_contract, format_contract_for_prompt
from .host_ir_schema import ArtifactSpec, HostIR, ParseAPIOperation, SymbolSpec
from .lowering_plan import build_stage_c_lowering_plan
from .parse_ir import Edge, ParseIR


def _output_keys(parse_ir: ParseIR) -> Dict[str, str]:
    from .pkg_field_conformance import output_keys

    return output_keys((node.id, node.name) for node in parse_ir.nodes.values())


def _format_nodes(parse_ir: ParseIR) -> str:
    lines: List[str] = []
    keys = _output_keys(parse_ir)
    for node_id in sorted(parse_ir.nodes.keys()):
        node = parse_ir.nodes[node_id]
        children = ",".join(str(cid) for cid in node.children_ids) if node.children_ids else "-"
        constraints = "; ".join(node.constraints) if node.constraints else "-"
        desc = node.description or "-"
        if len(desc) > 80:
            desc = desc[:77] + "..."
        bo = node.byte_order or "-"
        stop = getattr(node, "stop_condition", None) or "-"
        max_items = getattr(node, "max_items", None)
        max_items_str = str(max_items) if max_items is not None else "-"
        lines.append(
            f"{node.id}\t{name_clean(node.name)}\tkey={keys.get(str(node.id), '')}\t{node.kind}\t{node.data_type}\t"
            f"start={node.bit_start}\tsize_bits={node.size_bits}\tparent={node.parent_id}\t"
            f"children=[{children}]\tconstraints=[{constraints}]\t"
            f"message_type={node.message_type}\tstop_condition={stop}\tmax_items={max_items_str}\t"
            f"desc={desc}\tbyte_order={bo}"
        )
    return "\n".join(lines) if lines else "(no nodes)"


def _format_edges(edges: List[Edge]) -> str:
    lines: List[str] = []
    for edge in sorted(edges, key=lambda e: (e.src, e.dst, e.rel)):
        lines.append(
            f"{edge.src} -> {edge.dst}\trel={edge.rel}\tformula={edge.formula}\tmessage_type={edge.message_type}"
        )
    return "\n".join(lines) if lines else "(no edges)"


def _format_artifacts(artifacts: List[ArtifactSpec]) -> str:
    lines: List[str] = []
    for art in artifacts:
        lines.append(
            f"- id={art.id}, filename_template={art.filename_template}, language={art.language}, role={art.role}, kind={art.kind}, build_src={art.build_src}"
        )
    return "\n".join(lines) if lines else "(no artifacts specified)"


def _format_symbols(symbols: List[SymbolSpec]) -> str:
    lines: List[str] = []
    for sym in symbols:
        used = ",".join(sym.used_in) if sym.used_in else "-"
        lines.append(
            f"- {sym.name} (visibility={sym.visibility}, defined_in={sym.defined_in}, used_in=[{used}])"
        )
    return "\n".join(lines) if lines else "(no symbols specified)"


def _format_parse_api(ops: List[ParseAPIOperation]) -> str:
    lines: List[str] = []
    for op in ops:
        desc = op.description.replace("\n", " ") if op.description else ""
        lines.append(f"- {op.name} ({op.kind}): {op.signature} :: {desc}")
    return "\n".join(lines) if lines else "(no parse_api ops specified)"


def _parse_ir_json(parse_ir: ParseIR) -> str:
    nodes_data: List[Dict[str, object]] = []
    keys = _output_keys(parse_ir)
    for node in parse_ir.nodes.values():
        payload: Dict[str, object] = {
            "id": node.id,
            "name": name_clean(node.name),
            "output_key": keys.get(str(node.id), ""),
            "kind": node.kind,
            "data_type": node.data_type,
            "bit_start": node.bit_start,
            "size_bits": node.size_bits,
            "message_type": node.message_type,
            "parent_id": node.parent_id,
            "children_ids": node.children_ids,
            "constraints": node.constraints,
            "description": node.description,
            "byte_order": node.byte_order,
        }
        if getattr(node, "stop_condition", None):
            payload["stop_condition"] = getattr(node, "stop_condition")
        if getattr(node, "max_items", None) is not None:
            payload["max_items"] = getattr(node, "max_items")
        nodes_data.append(payload)

    edges_data: List[Dict[str, object]] = []
    for edge in parse_ir.edges:
        edges_data.append(
            {
                "src": edge.src,
                "dst": edge.dst,
                "rel": edge.rel,
                "formula": edge.formula,
                "message_type": edge.message_type,
            }
        )

    payload = {
        "protocol_name": parse_ir.protocol_name,
        "root_id": parse_ir.root_id,
        "nodes": nodes_data,
        "edges": edges_data,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def name_clean(name: str) -> str:
    return name.replace("\t", " ").strip()


def _lowering_plan_prompt_json(parse_ir: ParseIR, codegen_profile: Optional[Dict[str, object]]) -> str:
    plan = build_stage_c_lowering_plan(parse_ir, codegen_profile=codegen_profile)
    payload: Dict[str, object] = {
        "selector_dispatches": [
            {
                "selector_id": item.selector_id,
                "selector_name": item.selector_name,
                "selector_value": item.selector_value,
                "variant_id": item.variant_id,
                "variant_name": item.variant_name,
                "fields": [
                    {
                        "node_id": field.node_id,
                        "name": field.name,
                        "path": field.path,
                        "data_type": field.data_type,
                        "bit_start": field.bit_start,
                        "size_bits": field.size_bits,
                        "base_bit_offset": field.base_bit_offset,
                        "byte_offset": field.byte_offset,
                        "bit_offset": field.bit_offset,
                        "value_terms": [
                            {
                                "node_id": term.node_id,
                                "node_name": term.node_name,
                                "scale_bits": term.scale_bits,
                            }
                            for term in field.value_terms
                        ],
                        "flag_node_id": field.flag_node_id,
                        "flag_name": field.flag_name,
                        "flag_base_bit_offset": field.flag_base_bit_offset,
                        "flag_byte_offset": field.flag_byte_offset,
                        "flag_bit_offset": field.flag_bit_offset,
                        "flag_mask": field.flag_mask,
                        "flag_contract_id": field.flag_contract_id,
                    }
                    for field in item.fields
                ],
            }
            for item in plan.selector_dispatches
        ],
        "length_checks": [
            {
                "src": item.src,
                "src_name": item.src_name,
                "dst": item.dst,
                "dst_name": item.dst_name,
                "rel": item.rel,
                "formula": item.formula,
                "contract_id": item.contract_id,
            }
            for item in plan.length_checks
        ],
        "constraint_diagnostics": [
            {
                "id": item.id,
                "kind": item.kind,
                "obligation_type": item.obligation_type,
                "node_id": item.node_id,
                "node_name": item.node_name,
                "src": item.src,
                "src_name": item.src_name,
                "dst": item.dst,
                "dst_name": item.dst_name,
                "rel": item.rel,
                "formula": item.formula,
            }
            for item in plan.constraint_diagnostics
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _structured_codegen_output_contract() -> str:
    return (
        "Return only the independently complete JSON unit requested by the caller, using its JSON schema. "
        "The caller separately assembles the final files and summary; do not emit file or summary fences.\n"
        "For the final summary unit, artifacts_used contains artifact ids and public_symbols contains names. "
        "field_to_code_mapping entries contain format_path, artifact (id), and operation "
        "(container_op|field_op|selector_op|helper). constraint_to_code_mapping entries contain "
        "constraint_id (without BS_CONSTRAINT prefix), artifact, operation "
        "(enum_check|range_check|length_check|selector_branch_check|presence_check|helper), "
        "branch_guard (ungated or the exact selector/message direction guard) and failure_signal "
        "(BS_CONSTRAINT <id>). "
        "Use actual source evidence, not intended behavior.\n"
        "For compact Wireshark-C source, field_to_code_mapping may be empty when hf_* handles registered "
        "in hf_register_info and reachable proto_tree_add_* calls provide concrete field evidence. "
        "Per-field marker comments are needed only when handle, display label and add call cannot identify a field."
    )


def build_codegen_prompt(
    parse_ir: ParseIR,
    host_ir: HostIR,
    target_protocol_name: Optional[str] = None,
    api_doc_text: Optional[str] = None,
    codegen_profile: Optional[Dict[str, object]] = None,
    structured: bool = False,
) -> str:
    """Assemble a constrained prompt that treats host_ir + format_tree + api_doc as specs."""

    nodes_table = _format_nodes(parse_ir)
    edges_table = _format_edges(parse_ir.edges)
    artifacts_text = _format_artifacts(host_ir.artifacts)
    symbols_text = _format_symbols(host_ir.symbols)
    parse_api_text = _format_parse_api(host_ir.parse_api)
    extras_json = json.dumps(host_ir.extras or {}, ensure_ascii=False, indent=2)
    parse_ir_json = _parse_ir_json(parse_ir)
    lowering_plan_json = _lowering_plan_prompt_json(parse_ir, codegen_profile)
    target_proto = target_protocol_name or parse_ir.protocol_name or "unknown_protocol"
    api_doc_snippet = (api_doc_text or "(no api_doc found)").strip()
    if len(api_doc_snippet) > 6000:
        api_doc_snippet = api_doc_snippet[:6000] + "\n... (truncated)"

    profile_json = "(none)"
    if isinstance(codegen_profile, dict) and codegen_profile:
        profile_json = json.dumps({k: v for k, v in codegen_profile.items() if k != '_source_layout_context'},
                                  ensure_ascii=False, indent=2)

    output_schema = textwrap.dedent(
        """
First output a summary block using EXACTLY this fence and schema:
```summary
{
  "artifacts_used": ["artifact_id", ...],
  "public_symbols": ["symbol_name", ...],
  "field_to_code_mapping": [
    { "format_path": "path.from.root", "artifact": "artifact_id", "operation": "container_op|field_op|selector_op|helper" },
    { "format_path": "RootLayer.Header.LeafField", "artifact": "dissector_source", "operation": "field_op" },
    { "format_path": "RootLayer.Header", "artifact": "dissector_source", "operation": "container_op" }
  ],
  "constraint_to_code_mapping": [
    {
      "constraint_id": "BS_CONSTRAINT id without prefix",
      "artifact": "artifact_id",
      "operation": "enum_check|range_check|length_check|selector_branch_check|presence_check|helper",
      "branch_guard": "ungated|selector <id> == <value>|message_type request|message_type response",
      "failure_signal": "BS_CONSTRAINT <id>"
    }
  ]
}
```
Then output EACH artifact's content in its own fence:
```file:rendered_filename
<complete file content>
```
No prose, no extra text.

For large Wireshark-C artifacts, keep the summary and source compact:
`field_to_code_mapping` may be empty when the generated C source contains
concrete field evidence through `hf_*` field handles registered in
`hf_register_info` and used by reachable `proto_tree_add_*` calls. Do not add
per-field marker comments unless a field cannot be identified from the C field
handle, display label, and add call.
        """
    ).strip()

    if structured:
        output_schema = _structured_codegen_output_contract()

    constraints = textwrap.dedent(
        """
Hard constraints (host-agnostic):
- Artifacts: you may ONLY emit artifacts declared in host_ir.artifacts; do not invent or drop artifacts. Use artifact ids in summary; render filenames using the provided filename_template.
- Build role: kind=="source" && build_src==true files must be included in build scripts; kind=="header" must NOT be compiled directly.
- Symbols: visibility=="public" must be declared in interface artifacts (headers/modules) and defined in their designated source artifact; visibility=="internal" must stay private.
- Output keys: every Parse-IR node has an `output_key` derived from its PKG identity. Emit each field (and each visible aggregate item) under exactly that key: the Wireshark field abbreviation is `<protocol filter>.<output_key>`, the Scapy field name is `<output_key>`, and the Spicy unit field is `<output_key>`. Display labels may use the node name. Validation maps parser output back to PKG nodes through these keys.
- Format tree fidelity: every Parse-IR leaf field must be parsed at its specified span; keep parent/child hierarchy intact; do not reuse the same field handle for multiple regions; do not wrap a field node as a container. For kind=="tlv_seq" nodes, parse the template leaf fields once per item occurrence (repeated parsing).
- Visible aggregate spans: when a non-leaf node has an on-wire span (for example kind in ["payload","container","tlv_seq"] with bit_start and size_bits), emit a visible host tree item spanning the whole node before emitting descendants. For byte-oriented hosts this may use a bytes/none ProtoField or host-native subtree item named after the Parse-IR node. This preserves aggregate fields such as an options area while still allowing child fields under that item.
- Operation mapping: for Parse-IR nodes with kind=="field", you MUST ONLY use operations of kind "field_op" in field_to_code_mapping; never use "container_op" or "selector_op" for these leaf fields. Container-like nodes (kind in ["protocol","header","payload","container","variant","selector","tlv_seq"]) may use "container_op" or "selector_op" as appropriate. A visible aggregate host item for a container/tlv_seq still maps as "container_op", not "field_op".
- Compact Wireshark-C field evidence: if the generated C source would make `field_to_code_mapping` very large, leave `field_to_code_mapping` empty and rely on concrete C evidence: `hf_*` declarations, `hf_register_info` labels/abbreviations, and reachable `proto_tree_add_*` calls for each Parse-IR leaf field. Add `/* BS_FIELD <format_path> */` comments only for fields that cannot be identified from those C symbols and labels.
- Wireshark-C output budget: keep C complete and compact. Avoid duplicate `proto_tree_add_*` calls for the same field, avoid long explanatory comments, share short helpers for repeated bounds checks, and always finish registration/handoff functions instead of spending tokens on prose or redundant mapping entries.
- Selector/length edges: respect rel=condition_on and rel=length_of exactly as given in Parse-IR; do not reorder fields or ignore offsets/lengths.
- Stage C lowering plan: every item in `stage_c_lowering_plan.selector_dispatches` must become an explicit bounded selector branch in code. Parse and validate that branch's listed fields inside the selected branch, not in a generic catch-all branch. Every item in `stage_c_lowering_plan.length_checks` must become a branch-scoped length/consumption check.
- Branch-scoped validation: when a contract item is guarded by a selector or message direction, run its check inside the selected branch and record that guard in `constraint_to_code_mapping.branch_guard`. A global helper may compute values, but it must not turn a selected-branch missing-field violation into a silent skip.
- Selector fallback: do not use a Generic/Other branch to mask a missing branch that is present in `stage_c_lowering_plan.selector_dispatches`. Only accept an unknown/fallback selector value if the Parse-IR has an explicit fallback/unknown variant or the codegen profile explicitly requests permissive fallback behavior.
- TLV-seq semantics: when kind=="tlv_seq", treat its children as the TLV item template and parse it repeatedly from the current cursor, stopping when the node description/constraints indicate an end-of-list condition (e.g., stop_condition) or when the enclosing buffer ends.
- API docs: honor constraints/field meanings described in api_doc when mapping to format_tree nodes; do not ignore documented on-wire fields unless clearly out-of-band.
- Constraints & validation: Parse-IR node.constraints and normative API-doc rules (MUST/SHALL) are NOT optional. Implement them as runtime validation checks. If a check fails, emit a host-idiomatic diagnostic signal (e.g., Wireshark Expert Info / Wireshark Lua expert info / Zeek+Spicy parse failure) and stop parsing the packet ONLY when continuing would be unsafe.
- Constraint mapping: for every item listed in `<pargen_constraint_contract>`, add one `constraint_to_code_mapping` entry and make the generated source contain a reachable runtime check whose diagnostic includes exactly `BS_CONSTRAINT <id>`. A tag literal in unreachable code, comments, a dead helper, or an unrelated branch does not satisfy the contract.
- Wireshark-C Expert Info: when generating Wireshark-C, never attach constraint diagnostics to `NULL`; attach Expert Info to the current field item, selected variant tree, or packet/protocol tree so `tshark -V` exposes the `BS_CONSTRAINT <id>` evidence.
- Bytes/payload semantics: if a node represents raw bytes (kind=="payload" or data_type indicates bytes/blob), NEVER implement length bounds as numeric "value" range constraints on the bytes themselves. Use size_bits bounds, length fields, and rel=length_of sizing formulas instead.
- Codegen profile: if <codegen_profile> is provided, treat it as authoritative additional constraints (e.g., direction rules and generic parsing heuristics like count-prefixed payload handling). Do NOT ignore it.
- Implicit fields: you MAY introduce extra on-wire fields ONLY when required by <codegen_profile> (e.g., splitting a count/length byte from a count-prefixed blob/array). Do NOT remove or skip any Parse-IR leaf fields; keep offsets/lengths consistent.
        """
    ).strip()

    backend_rules = backend_contract_prompt(host_ir.host_name, context="parse_ir")
    if backend_rules:
        constraints += "\n\n" + backend_rules

    # Contract: require codegen to implement runtime constraint validation (no deterministic injection).
    contract_items = build_contract(parse_ir)
    constraints += "\n\n" + textwrap.dedent(
        f"""
<pargen_constraint_contract>
{format_contract_for_prompt(host_name=host_ir.host_name, items=contract_items)}
</pargen_constraint_contract>
        """
    ).strip()

    if isinstance(codegen_profile, dict) and codegen_profile.get('_fixed_layout_guard'):
        from .layout_guard import layout_prompt
        constraints += layout_prompt({'nodes': [dict(node_id=n.id, name=n.name, node_type=n.kind,
            children_ids=n.children_ids, bit_start=n.bit_start, size_bits=n.size_bits,
            byte_order=n.byte_order) for n in parse_ir.nodes.values()]})
        if codegen_profile.get('_source_layout_context'):
            constraints += ('\n<source_layout_clarification>\n'
                'The following relationships were derived by the model from the specification. '
                'Read the exact original source spans supplied. '
                'Preserve the declared PKG nodes and required outputs; honor source-supported aggregate, '
                'alternative branch and alias relationships without consuming the same wire bytes twice. '
                'Do not delete fields, assume a branch is active, or invent an independent additional '
                'wire region from catalog sibling placement.\n' +
                json.dumps(codegen_profile['_source_layout_context'], ensure_ascii=False) +
                '\n</source_layout_clarification>\n')

    return textwrap.dedent(
        f"""
You are a code generation agent. Produce complete plugin source files for host "{host_ir.host_name}".
- Plugin kind: {host_ir.plugin_kind}
- Target protocol name for this run: {target_proto}
- Parse-IR protocol name: {parse_ir.protocol_name}
- Inputs are authoritative specs: format_tree (Parse-IR), host_ir, api_doc.

<format_tree as table>
{nodes_table}

<format_tree edges>
{edges_table}

<format_tree JSON>
{parse_ir_json}

<stage_c_lowering_plan>
{lowering_plan_json}

<host_ir.artifacts>
{artifacts_text}

<host_ir.symbols>
{symbols_text}

<host_ir.parse_api>
{parse_api_text}

<host_ir.summary>
{host_ir.summary}

<host_ir.rules>
{host_ir.rules}

<host_ir.templates>
{host_ir.templates}

<host_ir.extras>
{extras_json}

<codegen_profile>
{profile_json}

<api_doc>
{api_doc_snippet}

{constraints}

Output format (strict):
{output_schema}
"""
    ).strip()


def build_doc_summary_codegen_prompt(
    *,
    protocol_name: str,
    document_summary: str,
    host_ir: HostIR,
    target_protocol_name: Optional[str] = None,
    api_doc_text: Optional[str] = None,
    codegen_profile: Optional[Dict[str, object]] = None,
    constraints_contract: Optional[str] = None,
    structured: bool = False,
) -> str:
    """
    Doc-only Stage C prompt: generate a host plugin using ONLY high-level protocol summaries (no Parse-IR).

    This is intended for ablations where the model must infer parsing structure from text.
    """

    artifacts_text = _format_artifacts(host_ir.artifacts)
    symbols_text = _format_symbols(host_ir.symbols)
    parse_api_text = _format_parse_api(host_ir.parse_api)
    extras_json = json.dumps(host_ir.extras or {}, ensure_ascii=False, indent=2)
    target_proto = target_protocol_name or protocol_name or "unknown_protocol"

    doc_snippet = (document_summary or "(no document summary found)").strip()
    if len(doc_snippet) > 12000:
        doc_snippet = doc_snippet[:12000] + "\n... (truncated)"

    api_doc_snippet = (api_doc_text or "").strip()
    if not api_doc_snippet:
        api_doc_snippet = "(no api_doc provided)"
    if len(api_doc_snippet) > 6000:
        api_doc_snippet = api_doc_snippet[:6000] + "\n... (truncated)"

    profile_json = "(none)"
    if isinstance(codegen_profile, dict) and codegen_profile:
        profile_json = json.dumps(codegen_profile, ensure_ascii=False, indent=2)

    output_schema = textwrap.dedent(
        """
First output a summary block using EXACTLY this fence and schema:
```summary
{
  "artifacts_used": ["artifact_id", ...],
  "public_symbols": ["symbol_name", ...],
  "field_to_code_mapping": [
    { "format_path": "Your.Logical.Path", "artifact": "artifact_id", "operation": "container_op|field_op|selector_op|helper" }
  ]
}
```
Then output EACH artifact's content in its own fence:
```file:rendered_filename
<complete file content>
```
No prose, no extra text.
        """
    ).strip()

    if structured:
        output_schema = _structured_codegen_output_contract()

    constraints = textwrap.dedent(
        """
Hard constraints (host-agnostic):
- Artifacts: you may ONLY emit artifacts declared in host_ir.artifacts; do not invent or drop artifacts. Use artifact ids in summary; render filenames using the provided filename_template.
- Build role: kind=="source" && build_src==true files must be included in build scripts; kind=="header" must NOT be compiled directly.
- Symbols: visibility=="public" must be declared in interface artifacts (headers/modules) and defined in their designated source artifact; visibility=="internal" must stay private.
- Protocol fidelity: treat <document_summary> as the ONLY authoritative protocol-format spec (no format_tree will be provided). Infer field structure, offsets, lengths, and variants conservatively.
- Robustness: always bounds-check before reading; handle truncated frames safely (mark malformed or stop parsing when unsafe).
- Registration: register a dissector for the Target protocol name so that `tshark -Y <filter_name>` works (filter_name == target protocol).
- Codegen profile: if <codegen_profile> is provided, treat it as authoritative extra constraints (ports/direction rules/payload handling). Do NOT ignore it.
        """
    ).strip()

    backend_rules = backend_contract_prompt(host_ir.host_name, context="doc_summary")
    if backend_rules:
        constraints += "\n\n" + backend_rules

    contract_text = (constraints_contract or "").strip()
    if contract_text:
        constraints += "\n\n" + textwrap.dedent(
            f"""
Constraint alerting contract (exp_constraint):
- If the contract below is provided, implement the runtime validation checks it describes.
- When a listed constraint is VIOLATED, emit the tag EXACTLY as a substring: `BS_CONSTRAINT <id>`.
- Do NOT emit tags on success.

<constraints_contract>
{contract_text}
            """
        ).strip()

    return textwrap.dedent(
        f"""
You are a code generation agent. Produce complete plugin source files for host "{host_ir.host_name}".
- Plugin kind: {host_ir.plugin_kind}
- Target protocol name for this run: {target_proto}
- Input spec: document_summary only (no Parse-IR)

<document_summary>
{doc_snippet}

<host_ir.artifacts>
{artifacts_text}

<host_ir.symbols>
{symbols_text}

<host_ir.parse_api>
{parse_api_text}

<host_ir.summary>
{host_ir.summary}

<host_ir.rules>
{host_ir.rules}

<host_ir.templates>
{host_ir.templates}

<host_ir.extras>
{extras_json}

<codegen_profile>
{profile_json}

<api_doc>
{api_doc_snippet}

{constraints}

Output format (strict):
{output_schema}
"""
    ).strip()
