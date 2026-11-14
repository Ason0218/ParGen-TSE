from __future__ import annotations

"""LLM code agent that renders host plugins from Parse-IR + Host-IR."""

import json
import re
import textwrap
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from .code_repair_agent import CodeRepairAgent, RepairDiagnostics
from .fence_utils import split_fenced_files
from .host_ir_schema import HostIR
from .llm_client import LLMClient
from .parse_ir import Node, ParseIR
from .profile_utils import profile_max_attempts, profile_violations
from .prompt_builder import build_codegen_prompt, build_doc_summary_codegen_prompt, name_clean
from .source_sanity import generated_source_violations


class FieldCoverageError(ValueError):
    """Validation failure that keeps structured missing leaf paths for scoped repair."""

    def __init__(self, message: str, *, missing_paths: List[str]):
        super().__init__(message)
        self.missing_paths = list(missing_paths)


_RE_TABLE_LOOP = re.compile(r"\bfor\b[^\n]*?\bin\s+i?pairs\s*\(\s*([A-Za-z_][\w.]*)\s*\)\s*do\b")


@lru_cache(maxsize=32)
def _table_driven_protofield_names(source_text: str) -> frozenset:
    """Lower-cased string literals of tables whose loop body creates ProtoFields.

    `local defs = {{2, "Transaction_Identifier", "u16"}, ...}` followed by
    `for _, d in ipairs(defs) do ... ProtoField.uint16(abbr, name) ... end` declares every
    field named in `defs`, though no field name sits next to a ProtoField call.
    """
    names: set = set()
    for loop in _RE_TABLE_LOOP.finditer(source_text):
        if "protofield" not in source_text[loop.end() : loop.end() + 800].lower():
            continue
        table = loop.group(1)
        decl = re.search(r"(?<![\w.])" + re.escape(table) + r"\s*=\s*\{", source_text)
        if not decl:
            continue
        depth, end = 0, None
        for i in range(decl.end() - 1, len(source_text)):
            ch = source_text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        body = source_text[decl.end() : end] if end is not None else ""
        for a, b in re.findall(r'"([^"\n]*)"|\'([^\'\n]*)\'', body):
            literal = (a or b).strip().lower()
            if literal:
                names.update({literal, literal.replace(" ", "_"), literal.replace("_", " ")})
    return frozenset(names)


class CodegenAgent:
    def _complete_codegen(self, prompt, host_ir, target_protocol):
        if callable(getattr(type(self.llm), 'complete_codegen_structured', None)):
            artifacts = {self._render_artifact_filename(a, target_protocol=target_protocol or 'protocol',
                         host_name=self._host_name(host_ir)): a.id for a in host_ir.artifacts}
            if len(artifacts) != len(host_ir.artifacts):
                raise ValueError('Host-IR renders duplicate artifact filenames')
            return self.llm.complete_codegen_structured(prompt, artifacts)
        return self.llm.complete(prompt)

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.last_summary: Optional[Dict[str, object]] = None
        self.last_files: Dict[str, str] = {}

    def _allow_partial_realization_for_diagnostics(self, codegen_profile: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(codegen_profile, dict):
            return False
        return bool(codegen_profile.get("allow_partial_realization_for_diagnostics", False))


    def _host_name(self, host_ir: HostIR) -> str:
        return str(getattr(host_ir, "host_name", "") or "").strip().lower().replace("-", "_")


    def _normalize_host_artifact_files(
        self,
        files: Dict[str, str],
        host_ir: HostIR,
        *,
        target_protocol: Optional[str],
    ) -> Dict[str, str]:
        if not files:
            return files
        host_name = self._host_name(host_ir)
        expected_names = [
            Path(
                self._render_artifact_filename(
                    artifact,
                    target_protocol=target_protocol or "protocol",
                    host_name=host_name,
                )
            ).name
            for artifact in host_ir.artifacts
        ]
        if any(Path(str(name)).name in expected_names for name in files):
            return files

        if host_name == "scapy":
            py_artifacts = [
                artifact
                for artifact in host_ir.artifacts
                if str(getattr(artifact, "language", "") or "").strip().lower() in {"python", "py"}
                or str(getattr(artifact, "filename_template", "") or "").strip().endswith(".py")
            ]
            artifact = py_artifacts[0] if py_artifacts else (host_ir.artifacts[0] if host_ir.artifacts else None)
            if artifact is None:
                return files
            fallback_name = Path(
                self._render_artifact_filename(
                    artifact,
                    target_protocol=target_protocol or "protocol",
                    host_name=host_name,
                )
            ).name
            for name, content in files.items():
                source = str(content or "")
                if (
                    Path(str(name)).suffix in {"", ".txt", ".py"}
                    and "from scapy" in source
                    and "Packet" in source
                    and ("fields_desc" in source or "bind_layers" in source)
                ):
                    patched = dict(files)
                    patched[fallback_name] = source
                    return patched

        return files


    def _render_artifact_filename(self, artifact: ArtifactSpec, *, target_protocol: str, host_name: str) -> str:
        cleaned_proto = name_clean(target_protocol or "protocol").strip("_").lower() or "protocol"
        host_clean = name_clean(host_name or "host").strip("_").lower() or "host"
        template = str(getattr(artifact, "filename_template", "") or "").strip()
        if not template:
            ext = ".lua" if host_clean == "wireshark_lua" else ".txt"
            return f"packet-{cleaned_proto}{ext}"
        rendered = template
        replacements = {
            "{protocol}": cleaned_proto,
            "${protocol}": cleaned_proto,
            "{{protocol}}": cleaned_proto,
            "{protocol_name}": cleaned_proto,
            "${protocol_name}": cleaned_proto,
            "{{protocol_name}}": cleaned_proto,
            "{host}": host_clean,
            "${host}": host_clean,
            "{{host}}": host_clean,
        }
        for old, new in replacements.items():
            rendered = rendered.replace(old, new)
        return rendered

    def _normalize_summary_artifact_refs(
        self,
        summary: Dict[str, object],
        files: Dict[str, str],
        host_ir: HostIR,
        *,
        target_protocol: str,
    ) -> Dict[str, object]:
        """Map filename-like artifact aliases in model summaries back to Host-IR ids."""
        if not isinstance(summary, dict) or not summary:
            return summary

        def alias(value: str) -> str:
            return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").lower()

        aliases: Dict[str, str] = {}
        ambiguous: set[str] = set()
        for artifact in host_ir.artifacts:
            artifact_id = str(getattr(artifact, "id", "") or "").strip()
            if not artifact_id:
                continue
            filename = self._render_artifact_filename(
                artifact,
                target_protocol=target_protocol,
                host_name=host_ir.host_name,
            )
            candidates = {
                artifact_id,
                filename,
                Path(filename).name,
                Path(filename).stem,
            }
            for file_name in files:
                if Path(file_name).name == Path(filename).name:
                    candidates.update({file_name, Path(file_name).name, Path(file_name).stem})
            for candidate in candidates:
                key = alias(candidate)
                if not key:
                    continue
                existing = aliases.get(key)
                if existing is not None and existing != artifact_id:
                    ambiguous.add(key)
                    continue
                aliases[key] = artifact_id

        if not aliases:
            return summary

        def normalize_artifact(value: object) -> object:
            raw = str(value or "").strip()
            key = alias(raw)
            if key and key not in ambiguous and key in aliases:
                return aliases[key]
            return value

        normalized = dict(summary)
        artifacts_used = normalized.get("artifacts_used")
        if isinstance(artifacts_used, list):
            normalized_used = [normalize_artifact(item) for item in artifacts_used]
            used_ids = {str(item) for item in normalized_used}
            for artifact in host_ir.artifacts:
                artifact_id = str(getattr(artifact, "id", "") or "").strip()
                if not artifact_id or artifact_id in used_ids:
                    continue
                filename = self._render_artifact_filename(
                    artifact,
                    target_protocol=target_protocol,
                    host_name=host_ir.host_name,
                )
                expected_name = Path(filename).name
                if any(Path(str(file_name)).name == expected_name for file_name in files):
                    normalized_used.append(artifact_id)
                    used_ids.add(artifact_id)
            normalized["artifacts_used"] = normalized_used

        for mapping_key in ("field_to_code_mapping", "constraint_to_code_mapping"):
            mapping = normalized.get(mapping_key)
            if not isinstance(mapping, list):
                continue
            new_mapping: list[object] = []
            for item in mapping:
                if isinstance(item, dict) and "artifact" in item:
                    patched = dict(item)
                    patched["artifact"] = normalize_artifact(patched.get("artifact"))
                    new_mapping.append(patched)
                else:
                    new_mapping.append(item)
            normalized[mapping_key] = new_mapping

        return normalized


    def _resolve_bit_start(self, parse_ir: ParseIR, node: Node, *, _seen: Optional[set[int]] = None) -> Optional[int]:
        seen = set(_seen or set())
        if int(node.id) in seen:
            return None
        seen.add(int(node.id))

        raw = node.bit_start
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return int(raw)
        if isinstance(raw, str):
            text = raw.strip()
            if re.fullmatch(r"-?\d+", text):
                return int(text)
            match = re.fullmatch(r"(\d+)\.bit_start(?:\s*\+\s*(-?\d+))?", text)
            if match:
                ref = parse_ir.nodes.get(int(match.group(1)))
                if ref is None:
                    return None
                base = self._resolve_bit_start(parse_ir, ref, _seen=seen)
                if base is None:
                    return None
                return base + int(match.group(2) or 0)
        return None


    @staticmethod
    def _condition_flag_name_relevant(src_node: Node, dst_node: Node) -> bool:
        src_name = str(src_node.name or "")
        if "flag" not in src_name.lower():
            return True

        generic = {
            "common",
            "field",
            "flag",
            "header",
            "ie",
            "information",
            "element",
            "item",
            "length",
            "message",
            "present",
            "type",
            "value",
            "variant",
            "variants",
        }

        def tokens(name: str) -> set[str]:
            text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name or ""))
            raw = {tok.lower() for tok in re.split(r"[^A-Za-z0-9]+", text) if tok}
            return {tok for tok in raw if tok not in generic}

        src_tokens = tokens(src_name)
        dst_tokens = tokens(str(dst_node.name or ""))
        if src_tokens & dst_tokens:
            return True

        aliases = {
            "v4": {"ipv4"},
            "v6": {"ipv6"},
            "ipv4": {"v4"},
            "ipv6": {"v6"},
        }
        for token in src_tokens:
            if aliases.get(token, set()) & dst_tokens:
                return True
        for token in dst_tokens:
            if aliases.get(token, set()) & src_tokens:
                return True
        return False


    def generate_plugin(
        self,
        parse_ir: ParseIR,
        host_ir: HostIR,
        target_protocol: Optional[str] = None,
        api_doc: Optional[str] = None,
        codegen_profile: Optional[Dict[str, Any]] = None,
        validate: bool = False,
    ) -> Dict[str, str]:
        """
        Returns a mapping of filename -> file content.
        If validate=True, perform consistency checks between LLM summary and Parse-IR/Host-IR.
        """

        prompt = build_codegen_prompt(
            parse_ir,
            host_ir,
            target_protocol_name=target_protocol,
            api_doc_text=api_doc,
            codegen_profile=codegen_profile,
            structured=callable(getattr(type(self.llm), 'complete_codegen_structured', None)),
        )

        max_attempts = 1
        if isinstance(codegen_profile, dict) and codegen_profile:
            max_attempts = profile_max_attempts(codegen_profile, default=2, key="max_codegen_attempts")
            if validate:
                max_attempts = max(
                    max_attempts,
                    profile_max_attempts(codegen_profile, default=max_attempts, key="max_realization_attempts"),
                )

        last_violations: List[str] = []
        summary: Dict[str, object] = {}
        files: Dict[str, str] = {}
        partial_realization_emitted = False
        for attempt in range(1, max_attempts + 1):
            attempt_prompt = prompt
            if last_violations:
                attempt_prompt = (
                    attempt_prompt
                    + "\n\nCODEGEN VIOLATIONS (fix and regenerate ALL artifacts; output format must remain identical):\n- "
                    + "\n- ".join(last_violations)
                )
            raw = self._complete_codegen(attempt_prompt, host_ir, target_protocol or parse_ir.protocol_name)
            summary, files = self._parse_response(raw)
            files = self._normalize_host_artifact_files(
                files,
                host_ir,
                target_protocol=target_protocol or parse_ir.protocol_name,
            )
            summary = self._normalize_summary_artifact_refs(
                summary,
                files,
                host_ir,
                target_protocol=target_protocol or parse_ir.protocol_name,
            )
            self.last_summary = summary
            self.last_files = dict(files)
            last_violations = self._codegen_violations(files, codegen_profile)
            if validate and not last_violations:
                last_violations, missing_leaf_paths = self._validation_result(summary, files, parse_ir, host_ir)
                if missing_leaf_paths:
                    repaired_files = self._repair_missing_leaf_chunks(
                        summary=summary,
                        files=files,
                        parse_ir=parse_ir,
                        host_ir=host_ir,
                        codegen_profile=codegen_profile,
                        api_doc=api_doc,
                        missing_paths=missing_leaf_paths,
                    )
                    if repaired_files is not None:
                        files = repaired_files
                        self.last_summary = summary
                        last_violations = self._codegen_violations(files, codegen_profile)
                        if not last_violations:
                            last_violations, _ = self._validation_result(summary, files, parse_ir, host_ir)
            if not last_violations:
                break

        if last_violations:
            # Forbidden substrings are considered hard failures (they usually represent
            # safety/ABI constraints). Required substrings are often "mechanical" hints and
            # can be repaired in later loops, so we surface them as a soft warning.
            source_violations = [v for v in last_violations if v.startswith("source_sanity:")]
            if source_violations:
                raise ValueError("Codegen output is incomplete or syntactically malformed: " + "; ".join(source_violations))
            has_forbidden = any(v.startswith("forbidden_substring=") for v in last_violations)
            if has_forbidden:
                raise ValueError("Codegen output violates codegen_profile constraints: " + "; ".join(last_violations))
            validation_violations = [v for v in last_violations if v.startswith("validation:")]
            if validation_violations:
                if self._allow_partial_realization_for_diagnostics(codegen_profile):
                    summary = dict(summary) if isinstance(summary, dict) else {}
                    summary.setdefault("profile_violations", list(last_violations))
                    summary.setdefault("realization_violation_mode", "diagnostic_partial")
                    self.last_summary = summary
                    last_violations = []
                    partial_realization_emitted = True
                else:
                    raise ValueError("Codegen output violates Parse-IR realization coverage: " + "; ".join(validation_violations))
            summary = dict(summary) if isinstance(summary, dict) else {}
            summary.setdefault("profile_violations", list(last_violations))
            summary.setdefault("profile_violation_mode", "soft_required_only")
            self.last_summary = summary

        if validate and not partial_realization_emitted:
            try:
                self._validate(summary, files, parse_ir, host_ir)
            except FieldCoverageError as exc:
                if not self._allow_partial_realization_for_diagnostics(codegen_profile):
                    raise
                summary = dict(summary) if isinstance(summary, dict) else {}
                summary.setdefault("profile_violations", [f"validation: {exc}"])
                summary.setdefault("realization_violation_mode", "diagnostic_partial")
                self.last_summary = summary
        return files

    def generate_plugin_from_doc_summary(
        self,
        *,
        protocol_name: str,
        document_summary: str,
        host_ir: HostIR,
        target_protocol: Optional[str] = None,
        api_doc: Optional[str] = None,
        codegen_profile: Optional[Dict[str, Any]] = None,
        constraints_contract: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Doc-only codegen path: render host plugin code from document summaries (no Parse-IR).

        Intended for ablations where the model must infer parsing structure from text.
        """

        prompt = build_doc_summary_codegen_prompt(
            protocol_name=protocol_name,
            document_summary=document_summary,
            host_ir=host_ir,
            target_protocol_name=target_protocol,
            api_doc_text=api_doc,
            codegen_profile=codegen_profile,
            constraints_contract=constraints_contract,
            structured=callable(getattr(type(self.llm), 'complete_codegen_structured', None)),
        )

        max_attempts = 1
        if isinstance(codegen_profile, dict) and codegen_profile:
            max_attempts = profile_max_attempts(codegen_profile, default=2, key="max_codegen_attempts")

        last_violations: List[str] = []
        summary: Dict[str, object] = {}
        files: Dict[str, str] = {}
        for attempt in range(1, max_attempts + 1):
            attempt_prompt = prompt
            if last_violations:
                attempt_prompt = (
                    attempt_prompt
                    + "\n\nCODEGEN VIOLATIONS (fix and regenerate ALL artifacts; output format must remain identical):\n- "
                    + "\n- ".join(last_violations)
                )
            raw = self._complete_codegen(attempt_prompt, host_ir, target_protocol or protocol_name)
            summary, files = self._parse_response(raw)
            files = self._normalize_host_artifact_files(
                files,
                host_ir,
                target_protocol=target_protocol or parse_ir.protocol_name,
            )
            self.last_summary = summary
            last_violations = self._codegen_violations(files, codegen_profile)
            if not last_violations:
                break

        if last_violations:
            source_violations = [v for v in last_violations if v.startswith("source_sanity:")]
            if source_violations:
                raise ValueError("Codegen output is incomplete or syntactically malformed: " + "; ".join(source_violations))
            has_forbidden = any(v.startswith("forbidden_substring=") for v in last_violations)
            if has_forbidden:
                raise ValueError("Codegen output violates codegen_profile constraints: " + "; ".join(last_violations))
            summary = dict(summary) if isinstance(summary, dict) else {}
            summary.setdefault("profile_violations", list(last_violations))
            summary.setdefault("profile_violation_mode", "soft_required_only")
            self.last_summary = summary

        return files

    def _codegen_violations(
        self, files: Dict[str, str], codegen_profile: Optional[Dict[str, Any]]
    ) -> List[str]:
        violations = list(profile_violations(files, codegen_profile))
        violations.extend(generated_source_violations(files))
        return violations

    def _validation_violations(
        self,
        summary: Dict[str, object],
        files: Dict[str, str],
        parse_ir: ParseIR,
        host_ir: HostIR,
    ) -> List[str]:
        violations, _missing_paths = self._validation_result(summary, files, parse_ir, host_ir)
        return violations

    def _validation_result(
        self,
        summary: Dict[str, object],
        files: Dict[str, str],
        parse_ir: ParseIR,
        host_ir: HostIR,
    ) -> Tuple[List[str], List[str]]:
        try:
            self._validate(summary, files, parse_ir, host_ir)
        except FieldCoverageError as exc:
            return [f"validation: {exc}"], list(exc.missing_paths)
        except ValueError as exc:
            return [f"validation: {exc}"], []
        return [], []

    def _missing_leaf_group_key(self, path: str) -> str:
        marker = ".IE_Value_Body."
        if marker in path:
            tail = path.split(marker, 1)[1]
            return "IE_Value_Body." + tail.split(".", 1)[0]
        marker = ".Message_Body."
        if marker in path:
            tail = path.split(marker, 1)[1]
            return "Message_Body." + tail.split(".", 1)[0]
        marker = ".Common_Header."
        if marker in path:
            return "Common_Header"
        return path.rsplit(".", 1)[0] if "." in path else path

    def _missing_leaf_groups(self, missing_paths: List[str]) -> List[Tuple[str, List[str]]]:
        groups: DefaultDict[str, List[str]] = defaultdict(list)
        for path in missing_paths:
            groups[self._missing_leaf_group_key(path)].append(path)
        return sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))

    def _format_missing_leaf_paths(self, missing_paths: List[str]) -> str:
        if not missing_paths:
            return "[]"

        ranked = self._missing_leaf_groups(missing_paths)
        group_parts: List[str] = []
        for group, paths in ranked[:12]:
            leaves = [p.rsplit(".", 1)[-1] for p in paths[:8]]
            suffix = ", ..." if len(paths) > len(leaves) else ""
            group_parts.append(f"{group} ({len(paths)}): {', '.join(leaves)}{suffix}")
        omitted = len(ranked) - 12
        if omitted > 0:
            group_parts.append(f"... {omitted} more groups")
        sample = missing_paths[:20]
        return (
            f"total={len(missing_paths)}; grouped_missing=["
            + " | ".join(group_parts)
            + f"]; sample_paths={sample}"
        )

    def _repair_missing_leaf_chunks(
        self,
        *,
        summary: Dict[str, object],
        files: Dict[str, str],
        parse_ir: ParseIR,
        host_ir: HostIR,
        codegen_profile: Optional[Dict[str, Any]],
        api_doc: Optional[str],
        missing_paths: List[str],
    ) -> Optional[Dict[str, str]]:
        """Repair Parse-IR realization gaps by targeting one missing leaf group at a time."""

        if not isinstance(codegen_profile, dict) or not codegen_profile:
            return None
        try:
            max_repairs = int(codegen_profile.get("max_realization_chunk_repairs", 0) or 0)
        except Exception:
            max_repairs = 0
        max_repairs = max(0, min(12, max_repairs))
        if max_repairs <= 0 or not files or not missing_paths:
            return None

        try:
            groups_per_repair = int(codegen_profile.get("max_realization_chunk_groups_per_repair", 1) or 1)
        except Exception:
            groups_per_repair = 1
        groups_per_repair = max(1, min(4, groups_per_repair))
        try:
            repair_attempts = int(codegen_profile.get("max_realization_chunk_repair_attempts", 1) or 1)
        except Exception:
            repair_attempts = 1
        repair_attempts = max(1, min(3, repair_attempts))

        repair_agent = CodeRepairAgent(self.llm)
        current_files = dict(files)
        changed_any = False
        blocked_groups: set[str] = set()

        for _round in range(max_repairs):
            _violations, current_missing = self._validation_result(summary, current_files, parse_ir, host_ir)
            if not current_missing:
                return current_files if changed_any else None

            groups = [(group, paths) for group, paths in self._missing_leaf_groups(current_missing) if group not in blocked_groups]
            if not groups:
                break
            selected = groups[:groups_per_repair]
            selected_group_names = [group for group, _paths in selected]
            selected_paths = [path for _group, paths in selected for path in paths]
            errors = [
                "Parse-IR realization is missing concrete decoded leaf fields.",
                "Repair only the listed chunk groups; do not rewrite unrelated parser logic.",
            ]
            for group, paths in selected:
                leaves = ", ".join(path.rsplit(".", 1)[-1] for path in paths[:12])
                suffix = ", ..." if len(paths) > 12 else ""
                errors.append(f"{group}: {len(paths)} missing leaf path(s): {leaves}{suffix}")
            errors.extend(f"- {path}" for path in selected_paths[:40])

            diagnostics = RepairDiagnostics(
                tool_name="parse_ir_realization_chunk",
                protocol_name=parse_ir.protocol_name,
                errors=tuple(errors),
                host_docs=api_doc or "",
                host_docs_max_chars=12000,
            )
            extra_rules = textwrap.dedent(
                """
                Chunked realization repair rules:
                - Patch concrete decode logic only for the missing groups listed in Tool errors.
                - Prefer adding concrete field declarations and add/add_packet_field calls for the exact leaf names.
                - For selector/TLV/option/grouped bodies, add or extend the existing variant/body decoder instead of adding another generic value field.
                - Preserve existing parser registration, top-level dispatch, and unrelated decoded fields.
                - Return a small diff when possible.
                """
            ).strip()
            changed = repair_agent.repair_files(
                files=current_files,
                diagnostics=diagnostics,
                codegen_profile=codegen_profile,
                extra_rules=extra_rules,
                max_attempts=repair_attempts,
                allow_new_files=False,
            )
            if not changed:
                blocked_groups.update(selected_group_names)
                continue

            merged = dict(current_files)
            merged.update(changed)
            current_files = merged
            changed_any = True
            blocked_groups.clear()

        _violations, current_missing = self._validation_result(summary, current_files, parse_ir, host_ir)
        if changed_any and len(current_missing) < len(missing_paths):
            return current_files
        return current_files if changed_any and not current_missing else None

    def _split_files(self, raw: str) -> Dict[str, str]:
        """Extract ```file:<name>``` fenced blocks into a dict."""
        files = split_fenced_files(raw)
        if not files and raw.strip():
            return {"plugin.txt": raw.strip()}
        return files

    def _parse_response(self, raw: str) -> Tuple[Dict[str, object], Dict[str, str]]:
        """Extract summary JSON + file blocks."""
        summary: Dict[str, object] = {}
        summary_pattern = re.compile(r"```summary\s*(\{.*?\})\s*```", re.DOTALL)
        match = summary_pattern.search(raw)
        if match:
            try:
                summary = json.loads(match.group(1))
            except Exception:
                summary = {}
        # Remove summary block before parsing files to avoid accidental capture.
        cleaned_raw = summary_pattern.sub("", raw) if match else raw
        files = self._split_files(cleaned_raw)
        return summary, files

    def _leaf_path_nodes(self, parse_ir: ParseIR) -> Dict[str, Node]:
        """
        Compute dot-separated paths for leaf nodes in the format tree,
        returning path -> Node (uses name_clean on every segment).
        """
        children_map: Dict[int, List[int]] = {}
        for node in parse_ir.nodes.values():
            if node.parent_id is None:
                continue
            children_map.setdefault(node.parent_id, []).append(node.id)

        parent_map: Dict[int, Optional[int]] = {node.id: node.parent_id for node in parse_ir.nodes.values()}

        def build_path(node_id: int) -> str:
            node = parse_ir.nodes[node_id]
            parts: List[str] = [name_clean(node.name)]
            cur = parent_map.get(node_id)
            while cur is not None and cur in parse_ir.nodes:
                parts.append(name_clean(parse_ir.nodes[cur].name))
                cur = parent_map.get(cur)
            return ".".join(reversed(parts))

        leaf_ids = [
            node.id
            for node in parse_ir.nodes.values()
            if not children_map.get(node.id) and self._is_required_leaf_node(node)
        ]
        path_to_node: Dict[str, Node] = {}
        for leaf_id in leaf_ids:
            path_to_node[build_path(leaf_id)] = parse_ir.nodes[leaf_id]
        return path_to_node

    def _leaf_paths(self, parse_ir: ParseIR) -> List[str]:
        """Compute dot-separated paths for leaf nodes in the format tree."""
        return list(self._leaf_path_nodes(parse_ir).keys())

    def _is_required_leaf_node(self, node: Node) -> bool:
        """Return true for leaf nodes that represent decoded values, not empty control structure."""
        kind = str(node.kind or "").strip().lower()
        # A leaf declared 0 bits wide occupies nothing on the wire, so there is nothing for the
        # plugin to decode (a placeholder), so it is not required.
        if str(node.size_bits).strip() == "0":
            return False
        if kind == "payload":
            size = str(node.size_bits or "").strip().lower()
            data_type = str(node.data_type or "").strip().lower()
            if size in {"", "variable", "var", "*"} and data_type in {"", "binary", "bytes", "payload"}:
                return False
        return kind in {"field", "selector", "length", "payload"}

    def _node_paths(self, parse_ir: ParseIR) -> Dict[int, str]:
        parent_map: Dict[int, Optional[int]] = {node.id: node.parent_id for node in parse_ir.nodes.values()}

        def build_path(node_id: int) -> str:
            node = parse_ir.nodes[node_id]
            parts: List[str] = [name_clean(node.name)]
            cur = parent_map.get(node_id)
            while cur is not None and cur in parse_ir.nodes:
                parts.append(name_clean(parse_ir.nodes[cur].name))
                cur = parent_map.get(cur)
            return ".".join(reversed(parts))

        return {node_id: build_path(node_id) for node_id in parse_ir.nodes}

    def _is_wireshark_lua_host(self, host_ir: HostIR) -> bool:
        host_name = str(getattr(host_ir, "host_name", "") or "").strip().lower().replace("-", "_")
        return host_name == "wireshark_lua"

    def _selector_value_from_edge(self, edge_formula: str, selector_id: int, variant_name: str) -> Optional[int]:
        formula = str(edge_formula or "").replace(" ", "")
        if f"val({int(selector_id)})" in formula:
            m = re.search(r"==\s*(0x[0-9a-fA-F]+|\d+)", formula)
            if m:
                try:
                    return int(m.group(1), 0)
                except Exception:
                    return None
        name_matches = re.findall(r"(?:^|_)(\d+)(?:_|$)", str(variant_name or ""))
        if len(name_matches) == 1:
            return int(name_matches[0])
        return None

    def _is_tlv_like_selector_node(self, selector: Node, selector_path: str) -> bool:
        label = name_clean(selector.name).lower()
        path = str(selector_path or "").lower()
        selector_text = f"{path}.{label}"
        if re.search(r"\b(?:message|function|opcode|operation)[ _.-]*type\b", selector_text):
            return False
        return bool(
            "tlv" in selector_text
            or "information element" in selector_text
            or re.search(r"(?:^|[._ -])ie[._ -]*type(?:$|[._ -])", selector_text)
            or re.search(r"(?:^|[._ -])option[._ -]*(?:type|code|kind)(?:$|[._ -])", selector_text)
            or re.search(r"(?:^|[._ -])tag(?:$|[._ -])", selector_text)
        )

    def _generated_source_text(self, files: Dict[str, str]) -> str:
        return "\n".join(str(content or "") for content in (files or {}).values())

    def _leaf_has_source_evidence(self, *, source_text: str, path: str, node: Node) -> bool:
        if self._leaf_has_lexical_source_evidence(source_text=source_text, path=path, node=node):
            return True
        # Table-driven registration: ProtoFields created in a loop over a table of field
        # definitions ({id, "Name", "type"}, ...) name each field only in that table.
        names = _table_driven_protofield_names(str(source_text or ""))
        if not names:
            return False
        for raw in (node.name, path.rsplit(".", 1)[-1] if path else ""):
            cleaned = name_clean(str(raw or "")).strip("_").lower()
            if cleaned and {cleaned, cleaned.replace("_", " "), cleaned.replace("_", ".")} & names:
                return True
        return False

    def _leaf_has_lexical_source_evidence(self, *, source_text: str, path: str, node: Node) -> bool:
        """
        Return true when generated source visibly declares/adds a concrete field for this leaf.

        This is intentionally lexical: it prevents a generic TLV `IE Value` field from satisfying
        selector-specific Parse-IR leaves, while still accepting ordinary ProtoField/filter/display
        naming conventions such as `foo_field`, `Foo Field`, or `x.foo_field`.
        """
        text = str(source_text or "").lower()
        if not text:
            return False

        candidates: set[str] = set()
        for raw in (node.name, path.rsplit(".", 1)[-1] if path else ""):
            cleaned = name_clean(str(raw or "")).strip("_").lower()
            if cleaned:
                candidates.add(cleaned)
                candidates.add(cleaned.replace("_", "."))
                candidates.add(cleaned.replace("_", " "))

        # Also accept a variant-prefixed concrete tail (common in deterministic TLV scaffolds).
        segments = [seg for seg in str(path or "").split(".") if seg]
        if len(segments) >= 2:
            tail = ".".join(segments[-2:])
            cleaned_tail = name_clean(tail).strip("_").lower()
            if cleaned_tail:
                candidates.add(cleaned_tail)
                candidates.add(cleaned_tail.replace("_", "."))
                candidates.add(cleaned_tail.replace("_", " "))

        for candidate in sorted(candidates, key=len, reverse=True):
            if not candidate:
                continue
            for match in re.finditer(re.escape(candidate), text):
                window = text[max(0, match.start() - 160) : match.end() + 160]
                if (
                    "protofield" in window
                    or ":add" in window
                    or "add_packet_field" in window
                    or ".fields" in window
                ):
                    return True
        return False

    def _wireshark_lua_materialized_leaf_paths(
        self, parse_ir: ParseIR, host_ir: HostIR, files: Optional[Dict[str, str]] = None
    ) -> set[str]:
        """
        Leaf paths that the Wireshark-Lua backend can deterministically materialize
        from generic backend scaffolds after LLM codegen.
        """
        if not self._is_wireshark_lua_host(host_ir):
            return set()

        children_map: Dict[int, List[int]] = {}
        for node in parse_ir.nodes.values():
            if node.parent_id is None:
                continue
            children_map.setdefault(node.parent_id, []).append(node.id)
        node_paths = self._node_paths(parse_ir)
        path_to_node_id = {path: node_id for node_id, path in node_paths.items()}

        def descendant_required_leaf_paths(root_id: int) -> List[str]:
            paths: List[str] = []
            stack: List[int] = [root_id]
            while stack:
                node_id = stack.pop()
                node = parse_ir.nodes.get(node_id)
                if node is None:
                    continue
                children = children_map.get(node_id, [])
                if children:
                    stack.extend(reversed(children))
                    continue
                if self._is_required_leaf_node(node):
                    path = node_paths.get(node_id)
                    if path:
                        paths.append(path)
            return paths

        materialized: set[str] = set()
        source_text = self._generated_source_text(files or {})
        for path, node in self._leaf_path_nodes(parse_ir).items():
            if self._leaf_has_source_evidence(source_text=source_text, path=path, node=node):
                materialized.add(path)
        for edge in parse_ir.edges:
            if str(edge.rel or "").strip() != "condition_on":
                continue
            selector = parse_ir.nodes.get(int(edge.src))
            variant = parse_ir.nodes.get(int(edge.dst))
            if selector is None or variant is None:
                continue
            if str(selector.kind or "").strip().lower() != "selector":
                continue
            if not children_map.get(int(variant.id)):
                continue
            selector_path = node_paths.get(int(selector.id), "")
            if not self._is_tlv_like_selector_node(selector, selector_path):
                continue
            if self._selector_value_from_edge(str(edge.formula or ""), int(selector.id), variant.name) is None:
                continue
            for path in descendant_required_leaf_paths(int(variant.id)):
                node = parse_ir.nodes.get(path_to_node_id.get(path, -1))
                if node is not None and self._leaf_has_source_evidence(source_text=source_text, path=path, node=node):
                    materialized.add(path)

        def is_fixed_tail_leaf(node: Node) -> bool:
            size_bits = node.size_bits
            if isinstance(size_bits, bool):
                return False
            if isinstance(size_bits, int):
                size = size_bits
            elif isinstance(size_bits, str) and size_bits.strip().isdigit():
                size = int(size_bits.strip())
            else:
                return False
            if size <= 0 or size % 8 != 0:
                return False
            label = name_clean(node.name).lower().replace("_", " ")
            return bool(re.search(r"\b(?:sequence|seq|spare|reserved|padding|pad)\b", label))

        for parent in parse_ir.nodes.values():
            if str(parent.kind or "").strip().lower() != "header":
                continue
            suffix: List[int] = []
            for child_id in reversed(children_map.get(parent.id, [])):
                child = parse_ir.nodes.get(child_id)
                if child is None:
                    if suffix:
                        break
                    continue
                if children_map.get(child.id) or not self._is_required_leaf_node(child) or not is_fixed_tail_leaf(child):
                    if suffix:
                        break
                    continue
                suffix.append(child.id)
            for node_id in reversed(suffix):
                path = node_paths.get(node_id)
                if path:
                    materialized.add(path)
        return materialized

    def _field_alias_key(self, name: str, protocol_name: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", str(name or "").lower())
        if "common" in tokens and "header" in tokens:
            header_idx = max(idx for idx, tok in enumerate(tokens) if tok == "header")
            if header_idx + 1 < len(tokens):
                tokens = tokens[header_idx + 1 :]
        protocol_tokens = set(re.findall(r"[a-z0-9]+", str(protocol_name or "").lower()))
        noise = {
            "common",
            "header",
            "packet",
            "protocol",
            "field",
            "format",
            "message",
        }
        kept = [tok for tok in tokens if tok not in protocol_tokens and tok not in noise]
        return "".join(kept or tokens)

    def _same_parent_alias_covered(
        self,
        path: str,
        node: Node,
        *,
        counts: Counter[str],
        leaf_map: Dict[str, Node],
        protocol_name: str,
    ) -> bool:
        parent_path, sep, _leaf = path.rpartition(".")
        if not sep:
            return False
        target_key = self._field_alias_key(node.name, protocol_name)
        if not target_key:
            return False
        for other_path, other_node in leaf_map.items():
            if other_path == path or counts[other_path] <= 0:
                continue
            other_parent, other_sep, _ = other_path.rpartition(".")
            if not other_sep or other_parent != parent_path:
                continue
            if self._field_alias_key(other_node.name, protocol_name) == target_key:
                return True
        return False

    def _concept_tokens(self, path: str, node: Node, protocol_name: str) -> set[str]:
        segments = [seg for seg in str(path or "").split(".") if seg]
        lower_segments = [seg.lower() for seg in segments]
        start = len(segments) - 1
        for idx, seg in enumerate(lower_segments):
            if "ie_value_body" in seg or "value_body" in seg:
                start = idx + 1
        for idx, seg in enumerate(lower_segments):
            if "variant" in seg:
                start = max(start, idx + 1)
        selected = segments[start:] if 0 <= start < len(segments) else [node.name]
        tokens = re.findall(r"[a-z0-9]+", " ".join(selected).lower())
        protocol_tokens = set(re.findall(r"[a-z0-9]+", str(protocol_name or "").lower()))
        noise = {
            "common",
            "header",
            "packet",
            "protocol",
            "field",
            "format",
            "message",
            "variant",
            "request",
            "response",
            "core",
            "body",
            "value",
            "item",
            "tlv",
            "ie",
            "type",
            "length",
        }
        return {tok for tok in tokens if tok not in protocol_tokens and tok not in noise}

    def _message_variant_requirement_alias_covered(
        self,
        path: str,
        node: Node,
        *,
        counts: Counter[str],
        leaf_map: Dict[str, Node],
        materialized_paths: set[str],
        protocol_name: str,
    ) -> bool:
        if "variant" not in str(path or "").lower():
            return False
        target = self._concept_tokens(path, node, protocol_name)
        if not target:
            return False
        for other_path, other_node in leaf_map.items():
            if other_path == path:
                continue
            if counts[other_path] <= 0 and other_path not in materialized_paths:
                continue
            candidate = self._concept_tokens(other_path, other_node, protocol_name)
            if not candidate:
                continue
            overlap = target & candidate
            if not overlap:
                continue
            if target.issubset(candidate) or candidate.issubset(target):
                if len(overlap) >= 2 or any(len(tok) >= 4 for tok in overlap):
                    return True
        return False

    def _source_marker_materialized_leaf_paths(self, files: Dict[str, str], leaf_map: Dict[str, Node]) -> set[str]:
        """Collect exact Parse-IR leaf paths marked in generated source."""

        required = set(leaf_map.keys())
        materialized: set[str] = set()
        marker_re = re.compile(r"\bBS_FIELD\s+([^\n*]+)")
        for source in files.values():
            text = str(source or "")
            for match in marker_re.finditer(text):
                raw_path = match.group(1).strip()
                raw_path = raw_path.split("*/", 1)[0].strip()
                raw_path = raw_path.strip("`'\"")
                if raw_path in required:
                    materialized.add(raw_path)
        return materialized

    def _source_evidence_tokens(self, text: str) -> set[str]:
        def norm(tok: str) -> str:
            tok = str(tok or "").strip().lower()
            if tok == "identifier":
                return "id"
            if len(tok) > 3 and tok.endswith("s"):
                return tok[:-1]
            return tok

        return {norm(tok) for tok in re.findall(r"[a-z0-9]+", str(text or "").lower())}

    def _wireshark_c_materialized_leaf_paths(
        self,
        parse_ir: ParseIR,
        host_ir: HostIR,
        files: Dict[str, str],
        leaf_map: Dict[str, Node],
    ) -> set[str]:
        """Infer field coverage from concrete Wireshark-C source evidence."""

        if str(getattr(host_ir, "host_name", "") or "").strip().lower() != "wireshark_c":
            return set()
        source_text = "\n".join(str(src or "") for src in files.values())
        if "proto_tree_add_item" not in source_text and "hf_register_info" not in source_text:
            return set()
        source_tokens = self._source_evidence_tokens(source_text)
        if not source_tokens:
            return set()

        materialized: set[str] = set()
        noise = {"field", "message", "packet", "protocol", "pdu", "adu", "tcp", "area", "container", "body"}
        protocol_tokens = self._source_evidence_tokens(parse_ir.protocol_name)
        for path, node in leaf_map.items():
            name_tokens = self._source_evidence_tokens(node.name)
            name_tokens = {tok for tok in name_tokens if tok and tok not in noise and tok not in protocol_tokens}
            if not name_tokens:
                continue
            if name_tokens.issubset(source_tokens):
                materialized.add(path)
        return materialized

    def _validate(self, summary: Dict[str, object], files: Dict[str, str], parse_ir: ParseIR, host_ir: HostIR) -> None:
        """Consistency checks between summary.mapping, Parse-IR, and Host-IR."""
        artifacts_declared = {art.id for art in host_ir.artifacts}

        artifacts_used: List[str] = []
        mapping_list: List[Dict[str, str]] = []
        if isinstance(summary, dict):
            if isinstance(summary.get("artifacts_used"), list):
                artifacts_used = [str(x) for x in summary.get("artifacts_used")]
            if isinstance(summary.get("field_to_code_mapping"), list):
                mapping_list = [m for m in summary.get("field_to_code_mapping") if isinstance(m, dict)]

        if artifacts_used:
            unknown_artifacts = [a for a in artifacts_used if a not in artifacts_declared]
            if unknown_artifacts:
                raise ValueError(f"Generated artifacts_used not in host_ir: {unknown_artifacts}")
            missing_artifacts = [a for a in artifacts_declared if a not in artifacts_used]
            if missing_artifacts:
                raise ValueError(f"Missing artifacts from summary: {missing_artifacts}")

        # Field coverage: each leaf path must appear exactly once.
        leaf_map = self._leaf_path_nodes(parse_ir)
        required_paths = list(leaf_map.keys())

        counts: Counter[str] = Counter()
        ops_by_path: DefaultDict[str, List[str]] = defaultdict(list)

        for m in mapping_list:
            path = str(m.get("format_path") or "").strip()
            if not path:
                continue
            counts[path] += 1
            op = str(m.get("operation") or "").strip()
            if op:
                ops_by_path[path].append(op)

        materialized_paths = self._wireshark_lua_materialized_leaf_paths(parse_ir, host_ir, files)
        materialized_paths |= self._source_marker_materialized_leaf_paths(files, leaf_map)
        materialized_paths |= self._wireshark_c_materialized_leaf_paths(parse_ir, host_ir, files, leaf_map)
        missing_paths = [
            p
            for p in required_paths
            if counts[p] == 0
            and p not in materialized_paths
            and not self._same_parent_alias_covered(
                p,
                leaf_map[p],
                counts=counts,
                leaf_map=leaf_map,
                protocol_name=parse_ir.protocol_name,
            )
            and not self._message_variant_requirement_alias_covered(
                p,
                leaf_map[p],
                counts=counts,
                leaf_map=leaf_map,
                materialized_paths=materialized_paths,
                protocol_name=parse_ir.protocol_name,
            )
        ]
        duplicate_paths = [p for p in required_paths if counts[p] > 1]

        if required_paths and missing_paths:
            raise FieldCoverageError(
                "Field coverage missing paths (required leaf fields not mapped): "
                + self._format_missing_leaf_paths(missing_paths),
                missing_paths=missing_paths,
            )

        if duplicate_paths:
            raise ValueError(f"Field coverage duplicate mappings (leaf fields mapped more than once): {duplicate_paths}")

        # kind vs operation: leaf nodes with kind=='field' must NOT use container_op/selector_op.
        wrong_kind_paths: List[str] = []
        for path, node in leaf_map.items():
            ops = ops_by_path.get(path, [])
            if not ops:
                continue
            if node.kind == "field":
                bad_ops = [op for op in ops if op in {"container_op", "selector_op"}]
                if bad_ops:
                    wrong_kind_paths.append(f"{path} (kind={node.kind}, ops={sorted(set(bad_ops))})")

        if wrong_kind_paths:
            raise ValueError(
                "Leaf nodes with kind=='field' must only use field_op in field_to_code_mapping; "
                f"violations: {wrong_kind_paths}"
            )

        # Ensure files were produced for declared artifacts (len match is a cheap check).
        if artifacts_declared and len(files) < len(artifacts_declared):
            raise ValueError("Fewer files than declared artifacts; likely missing outputs.")
