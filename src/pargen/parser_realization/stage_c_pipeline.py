from __future__ import annotations

"""New Stage C orchestrator: Parse-IR + Host-IR → host plugin code."""

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pargen.paths import DATA_DIR, LOGS_DIR

from .code_repair_agent import CodeRepairAgent, RepairDiagnostics
from .codegen_agent import CodegenAgent
from .constraint_contract import build_contract, find_missing_tags, format_contract_for_prompt
from .profile_utils import strip_protocol_hints
from .fence_utils import resolve_plugin_file_path, validate_fenced_filename
from .host_docs import load_host_docs
from .host_ir_agent import HostIRAgent
from .llm_client import LLMClient, LLMConfig, create_llm_client, resolve_effective_llm_request
from .lowering_plan import build_stage_c_lowering_plan, stage_c_lowering_plan_to_dict
from .parse_ir import build_parse_ir
from .source_sanity import generated_source_violations


@dataclass
class StageCConfig:
    format_tree_path: Optional[Path]
    host_name: str
    protocol_family: Optional[str]
    host_docs_dir: Path
    output_dir: Path

    host_ir_cache_path: Optional[Path] = None
    codegen_profile_path: Optional[Path] = None
    api_doc_paths: Optional[List[Path]] = None
    doc_summary_text: Optional[str] = None
    verify: bool = False
    llm_config: Optional[LLMConfig] = None
    constraints_mode: str = "codegen"
    contract_repair_max_rounds: int = 2
    allow_missing_contract_tags: bool = False
    layout_guard_state: Optional[Dict[str, Any]] = None
    source_layout_context: Optional[Dict[str, Any]] = None
    generated_source_resume: Optional[Dict[str, Any]] = None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sha256_directory(root: Optional[Path]) -> Optional[str]:
    if not root:
        return None
    try:
        base = Path(root)
        if not base.exists() or not base.is_dir():
            return None
        h = hashlib.sha256()
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = path.relative_to(base).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            digest = _sha256_file(path)
            if digest:
                h.update(digest.encode("ascii"))
            h.update(b"\0")
        return h.hexdigest()
    except OSError:
        return None


def build_stage_c_input_signature(
    cfg: StageCConfig,
    *,
    input_spec: str,
    constraints_mode: str,
    constraints_mode_requested: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a stable fingerprint for deciding whether Stage C output is reusable."""
    mode = str(constraints_mode or "codegen").strip().lower()
    if mode not in {"inject", "auto", "codegen"}:
        mode = "codegen"
    requested = str(constraints_mode_requested or mode).strip().lower() or mode
    llm_cfg = cfg.llm_config
    signature: Dict[str, Any] = {
        "schema": 2,
        "input_spec": str(input_spec or "format_tree"),
        "host": str(cfg.host_name or ""),
        "protocol_family": str(cfg.protocol_family or "") or None,
        "constraints_mode": mode,
        "constraints_mode_requested": requested,
        "contract_repair_max_rounds": int(getattr(cfg, "contract_repair_max_rounds", 2) or 2),
        "allow_missing_contract_tags": bool(getattr(cfg, "allow_missing_contract_tags", False)),
        "verify": bool(getattr(cfg, "verify", False)),
        "fixed_layout_guard": cfg.layout_guard_state is not None,
        "fixed_layout_checker": (cfg.layout_guard_state.get('checker_version')
                                 if cfg.layout_guard_state is not None else None),
        "generated_source_resume_origin_sha256": (cfg.generated_source_resume.get('origin_sha256')
                                                 if cfg.generated_source_resume is not None else None),
        "source_layout_context_sha256": (_sha256_text(json.dumps(cfg.source_layout_context, sort_keys=True))
                                         if cfg.source_layout_context is not None else None),
        "format_tree_sha256": None if input_spec == "doc_summary" else _sha256_file(cfg.format_tree_path),
        "doc_summary_sha256": _sha256_text(str(cfg.doc_summary_text or "")) if input_spec == "doc_summary" else None,
        "host_docs_sha256": _sha256_directory(cfg.host_docs_dir),
        "host_ir_cache_sha256": _sha256_file(cfg.host_ir_cache_path),
        "codegen_profile_sha256": _sha256_file(cfg.codegen_profile_path),
    }
    if llm_cfg is not None:
        signature["llm"] = resolve_effective_llm_request(llm_cfg)
    if cfg.api_doc_paths:
        signature["api_docs_sha256"] = [
            {"path": Path(path).name, "sha256": _sha256_file(path)}
            for path in cfg.api_doc_paths
        ]
    return {k: v for k, v in signature.items() if v is not None}


def _write_plugin_files(plugin_dir: Path, files: Dict[str, str]) -> List[str]:
    source_violations = generated_source_violations(files)
    if source_violations:
        sample = "; ".join(source_violations[:10])
        raise RuntimeError(f"Generated plugin source failed static completeness checks: {sample}")

    normalized: Dict[str, str] = {}
    for name, content in files.items():
        rel = validate_fenced_filename(name)
        if rel in normalized:
            raise ValueError(f"duplicate generated plugin filename after normalization: {rel!r}")
        normalized[rel] = content

    if plugin_dir.exists():
        shutil.rmtree(plugin_dir)
    plugin_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in normalized.items():
        path = resolve_plugin_file_path(plugin_dir, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return sorted(normalized.keys())


def _write_failed_codegen_candidate(
    *,
    out_dir: Path,
    host_name: str,
    files: Optional[Dict[str, str]],
    llm_summary: Optional[Dict[str, Any]],
    error: BaseException,
) -> Optional[Path]:
    if not files:
        return None
    failed_dir = out_dir / "_failed_codegen_candidate" / str(host_name or "unknown_host")
    if failed_dir.exists():
        shutil.rmtree(failed_dir)
    failed_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name, content in files.items():
        try:
            rel = validate_fenced_filename(name)
            path = resolve_plugin_file_path(failed_dir, rel)
        except Exception:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "candidate.txt")).strip("._") or "candidate.txt"
            path = failed_dir / safe
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content or ""), encoding="utf-8")
        written.append(path.resolve().relative_to(failed_dir.resolve()).as_posix())
    meta = {
        "error": str(error),
        "error_type": type(error).__name__,
        "files": sorted(written),
    }
    if llm_summary:
        meta["llm_summary"] = llm_summary
    (failed_dir / "_failure.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return failed_dir


def _infer_protocol_root(*, tree_path: Path, protocol_family: Optional[str]) -> Path:
    """Infer protocol root (data/<protocol>/) from stage artifact paths."""
    family = (protocol_family or "").strip()
    if family:
        cand = DATA_DIR / family
        if cand.exists():
            return cand.resolve()
        for parent in tree_path.parents:
            if parent.name == family:
                return parent

    for parent in tree_path.parents:
        if (parent / "inputs").exists() and (parent / "outputs").exists():
            return parent

    for parent in tree_path.parents:
        if parent.name == "outputs":
            return parent.parent

    return tree_path.parent.parent


def run_stage_c(cfg: StageCConfig, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.verify:
        raise RuntimeError(
            "--stage-c-verify is not implemented as a real verification hook. "
            "Use --stage-c-build-ws, --stage-c-semantic-verify, or --stage-c-exp-constraint-verify instead."
        )

    llm = llm_client or create_llm_client(cfg.llm_config)

    docs = load_host_docs(cfg.host_docs_dir)
    doc_summary = str(getattr(cfg, "doc_summary_text", "") or "").strip()
    # Code generation sees the PKG and the platform context only: no traffic text and no specification sections.
    api_doc_text = _load_api_docs_explicit_only(cfg)
    if not docs and not (cfg.host_ir_cache_path and cfg.host_ir_cache_path.exists()):
        raise RuntimeError(
            f"No host docs found under {cfg.host_docs_dir} and no Host-IR cache specified; cannot learn Host-IR."
        )

    host_ir_agent = HostIRAgent(llm)
    try:
        host_ir = host_ir_agent.build_host_ir(
            host_name=cfg.host_name,
            docs=docs,
            cache_path=cfg.host_ir_cache_path,
        )
    except Exception:
        raw = getattr(host_ir_agent, "last_raw_host_ir", None)
        if raw:
            stage_c_logs = LOGS_DIR / "stage_c"
            stage_c_logs.mkdir(parents=True, exist_ok=True)
            fail_path = stage_c_logs / "host_ir_raw_failed.yaml"
            fail_path.write_text(raw, encoding="utf-8")
        raise

    codegen_profile = _load_codegen_profile(cfg.codegen_profile_path)
    if cfg.source_layout_context is not None:
        if (cfg.layout_guard_state is None or cfg.source_layout_context.get('tree_sha256') != _sha256_file(cfg.format_tree_path)
                or not isinstance(cfg.source_layout_context.get('relationships'), list)
                or not isinstance(cfg.source_layout_context.get('evidence_locations'), list)):
            raise ValueError('Source-layout clarification context needs the verified current tree')
        codegen_profile = dict(codegen_profile or {}, _source_layout_context=cfg.source_layout_context)
    if cfg.layout_guard_state is not None:
        if cfg.host_name != 'wireshark_c' or doc_summary:
            raise ValueError('Fixed-layout guard requires a source tree and Wireshark C')
        codegen_profile = dict(codegen_profile or {}, _fixed_layout_guard=True)

    code_agent = CodegenAgent(llm)
    if doc_summary:
        requested_mode = "codegen"
        constraints_mode = "codegen"

        target_proto = cfg.protocol_family or "unknown_protocol"
        files = code_agent.generate_plugin_from_doc_summary(
            protocol_name=target_proto,
            document_summary=doc_summary,
            host_ir=host_ir,
            target_protocol=target_proto,
            api_doc=api_doc_text,
            codegen_profile=codegen_profile,
        )

        plugin_dir = out_dir / "plugin" / cfg.host_name
        written_files = _write_plugin_files(plugin_dir, files)
        input_signature = build_stage_c_input_signature(
            cfg,
            input_spec="doc_summary",
            constraints_mode=constraints_mode,
            constraints_mode_requested=requested_mode,
        )
        llm_usage = _llm_usage_summary(llm)
        llm_usage_paths = _write_llm_usage_files(out_dir, llm_usage)

        meta = {
            "host": cfg.host_name,
            "protocol": target_proto,
            "target_protocol": target_proto,
            "constraints_mode": constraints_mode,
            "constraints_mode_requested": requested_mode,
            "input_spec": "doc_summary",
            "doc_summary_chars": len(doc_summary),
            "input_signature": input_signature,
            "generated_at": int(time.time()),
        }
        if llm_usage:
            meta["llm_usage"] = llm_usage
        if llm_usage_paths:
            meta.update(llm_usage_paths)
        if cfg.codegen_profile_path:
            meta["codegen_profile_path"] = str(cfg.codegen_profile_path)
        try:
            (plugin_dir / "_pargen_stage_c_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

        summary: Dict[str, Any] = {
            "host": cfg.host_name,
            "protocol": target_proto,
            "target_protocol": target_proto,
            "plugin_dir": str(plugin_dir),
            "files": written_files,
            "docs_loaded": len(docs),
            "constraints_mode": constraints_mode,
            "constraints_mode_requested": requested_mode,
            "input_spec": "doc_summary",
            "doc_summary_chars": len(doc_summary),
            "input_signature": input_signature,
        }
        if llm_usage:
            summary["llm_usage"] = llm_usage
        if llm_usage_paths:
            summary.update(llm_usage_paths)
        if cfg.codegen_profile_path:
            summary["codegen_profile_path"] = str(cfg.codegen_profile_path)
        if getattr(code_agent, "last_summary", None):
            summary["llm_summary"] = code_agent.last_summary
        summary_path = out_dir / "stage_c_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    if cfg.format_tree_path is None:
        raise RuntimeError("Stage C requires format_tree_path unless doc_summary_text is provided.")
    tree_json = json.loads(cfg.format_tree_path.read_text(encoding="utf-8"))
    parse_ir = build_parse_ir(tree_json)

    target_proto = cfg.protocol_family or parse_ir.protocol_name
    try:
        if cfg.generated_source_resume is not None:
            from scripts.c_layout_resume import validate_source_resume
            validate_source_resume(tree_json, cfg.layout_guard_state, cfg.generated_source_resume)
            files = dict(cfg.generated_source_resume['files'])
            code_agent.last_summary = cfg.generated_source_resume['llm_summary']
        else:
            files = code_agent.generate_plugin(
                parse_ir,
                host_ir,
                target_protocol=target_proto,
                api_doc=api_doc_text,
                codegen_profile=codegen_profile,
                validate=True,
            )
    except Exception as exc:
        _write_failed_codegen_candidate(
            out_dir=out_dir,
            host_name=cfg.host_name,
            files=getattr(code_agent, "last_files", None),
            llm_summary=getattr(code_agent, "last_summary", None),
            error=exc,
        )
        raise

    # Route 2: optionally require the LLM output to carry runtime constraint validation.
    contract_items = build_contract(parse_ir)
    mode = "codegen"
    allow_missing_contract_tags = bool(getattr(cfg, "allow_missing_contract_tags", False))

    missing = find_missing_tags(files=files, items=contract_items, host_name=cfg.host_name)
    contract_repaired = False
    contract_repair_rounds = 0

    if cfg.generated_source_resume is not None and (missing or mode != 'codegen'):
        raise ValueError('Generated-source resume requires the existing complete constraint contract; no reinjection')


    if mode == "codegen" and missing and not allow_missing_contract_tags:
        repair_agent = CodeRepairAgent(llm)
        protocol_root = _infer_protocol_root(tree_path=cfg.format_tree_path, protocol_family=cfg.protocol_family)
        logs_dir = protocol_root / "logs" / "stage_c" / "contract_repair" / cfg.host_name
        logs_dir.mkdir(parents=True, exist_ok=True)

        for round_idx in range(max(1, int(getattr(cfg, "contract_repair_max_rounds", 2) or 2))):
            contract_repair_rounds = round_idx + 1
            missing_items = [it for it in contract_items if it.id in set(missing)]
            contract_text = format_contract_for_prompt(host_name=cfg.host_name, items=missing_items)
            ts = time.strftime("%Y%m%d_%H%M%S")
            prompt_path = logs_dir / f"round{round_idx:02d}_{ts}_prompt.txt"
            response_path = logs_dir / f"round{round_idx:02d}_{ts}_response.txt"

            diagnostics = RepairDiagnostics(
                tool_name="constraint_contract",
                protocol_name=parse_ir.protocol_name,
                errors=tuple([f"Missing constraint tags ({len(missing)}): {', '.join(missing[:25])}"]),
                log_tail="",
                host_docs=contract_text,
            )
            extra_rules = (
                "Constraint contract repair (STRICT):\n"
                "- Add/repair runtime validation so that every missing contract item emits its tag.\n"
                "- A tag literal alone is not enough: attach each check to the node read, selected selector branch, or length_of check that owns the contract item.\n"
                "- For condition_on items, validate branch consistency inside the selected branch: selected branch bodies must be present and bounded; unselected branches must not be consumed as if selected.\n"
                "- For length_of items, validate expected bytes against available/consumed bytes before accepting the packet as normal; truncation and leftover bytes relevant to the relation must emit the contract tag.\n"
                "- Keep existing parsing logic and field offsets intact.\n"
                "- Do NOT remove deterministic constraint checks if they already exist.\n"
            )

            changed = repair_agent.repair_files(
                files=files,
                diagnostics=diagnostics,
                codegen_profile=codegen_profile,
                extra_rules=extra_rules,
                max_attempts=2,
                prompt_path=prompt_path,
                response_path=response_path,
                allow_new_files=False,
            )
            if not changed:
                break
            files = dict(files)
            files.update(changed)
            missing = find_missing_tags(files=files, items=contract_items, host_name=cfg.host_name)
            contract_repaired = True
            if not missing:
                break



    if contract_items and missing and not allow_missing_contract_tags:
        sample = ", ".join(missing[:10])
        raise RuntimeError(
            "Stage C constraint contract not satisfied; refusing to write plugin files "
            f"(constraints_mode={mode}, missing_tags={len(missing)}, sample=[{sample}])."
        )

    if cfg.layout_guard_state is not None:
        from .layout_guard import repair_c_layout
        try:
            files = repair_c_layout(tree_json, files, llm, cfg.layout_guard_state,
                                    out_dir / 'layout_guard', codegen_profile, api_doc_text)
            remaining = find_missing_tags(files=files, items=contract_items, host_name=cfg.host_name)
            if remaining and not allow_missing_contract_tags:
                raise RuntimeError('Layout repair removed required constraint tags')
        except Exception as exc:
            _write_failed_codegen_candidate(out_dir=out_dir, host_name=cfg.host_name,
                files=cfg.layout_guard_state.get('files', files),
                llm_summary=getattr(code_agent, 'last_summary', None), error=exc)
            raise

    plugin_dir = out_dir / "plugin" / cfg.host_name
    written_files = _write_plugin_files(plugin_dir, files)
    if cfg.layout_guard_state is not None:
        receipt = dict(version=1, tree_sha256=_sha256_file(cfg.format_tree_path),
            scope=cfg.layout_guard_state['report']['scope'],
            repair_rounds=cfg.layout_guard_state['rounds'],
            report=cfg.layout_guard_state['report'])
        (out_dir / 'fixed_layout_contract.json').write_text(json.dumps(receipt, indent=2))
    input_signature = build_stage_c_input_signature(
        cfg,
        input_spec="format_tree",
        constraints_mode=mode,
        constraints_mode_requested=mode,
    )
    llm_usage = _llm_usage_summary(llm)
    llm_usage_paths = _write_llm_usage_files(out_dir, llm_usage)

    meta = {
        "host": cfg.host_name,
        "protocol": parse_ir.protocol_name,
        "target_protocol": target_proto,
        "constraints_mode": mode,
        "input_signature": input_signature,
        "generated_at": int(time.time()),
    }
    if llm_usage:
        meta["llm_usage"] = llm_usage
    if llm_usage_paths:
        meta.update(llm_usage_paths)
    if contract_items:
        meta["constraint_contract_total"] = len(contract_items)
    if missing:
        meta["constraint_contract_missing_count"] = len(missing)
        meta["constraint_contract_missing_sample"] = list(missing[:50])
    if contract_repaired:
        meta["constraint_contract_repaired"] = True
        meta["constraint_contract_repair_rounds"] = int(contract_repair_rounds)
    if cfg.codegen_profile_path:
        meta["codegen_profile_path"] = str(cfg.codegen_profile_path)
    try:
        (plugin_dir / "_pargen_stage_c_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        pass

    summary: Dict[str, Any] = {
        "host": cfg.host_name,
        "protocol": parse_ir.protocol_name,
        "target_protocol": target_proto,
        "plugin_dir": str(plugin_dir),
        "files": written_files,
        "docs_loaded": len(docs),
        "constraints_mode": mode,
        "input_signature": input_signature,
    }
    if llm_usage:
        summary["llm_usage"] = llm_usage
    if llm_usage_paths:
        summary.update(llm_usage_paths)
    if contract_items:
        summary["constraint_contract_total"] = len(contract_items)
    if missing:
        summary["constraint_contract_missing_count"] = len(missing)
        summary["constraint_contract_missing_sample"] = list(missing[:50])
    if contract_repaired:
        summary["constraint_contract_repaired"] = True
        summary["constraint_contract_repair_rounds"] = int(contract_repair_rounds)
    if cfg.codegen_profile_path:
        summary["codegen_profile_path"] = str(cfg.codegen_profile_path)

    if getattr(code_agent, "last_summary", None):
        summary["llm_summary"] = code_agent.last_summary

    summary_path = out_dir / "stage_c_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def _llm_usage_summary(llm: LLMClient) -> Optional[Dict[str, Any]]:
    usage_fn = getattr(llm, "usage_summary", None)
    if callable(usage_fn):
        try:
            usage = usage_fn()
        except Exception:
            usage = None
        if isinstance(usage, dict) and int(usage.get("calls") or 0) > 0:
            return usage
    records = getattr(llm, "usage_records", None)
    if isinstance(records, list) and records:
        return {"calls": len(records), "records": records}
    return None


def _write_llm_usage_files(out_dir: Path, llm_usage: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(llm_usage, dict) or int(llm_usage.get("calls") or 0) <= 0:
        return {}
    records = llm_usage.get("records")
    if not isinstance(records, list):
        records = []

    usage_jsonl = out_dir / "stage_c_llm_usage.jsonl"
    usage_summary = out_dir / "stage_c_llm_usage_summary.json"
    try:
        with usage_jsonl.open("w", encoding="utf-8") as handle:
            for idx, record in enumerate(records):
                payload = dict(record) if isinstance(record, dict) else {"value": record}
                payload.setdefault("call_index", idx)
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

        summary_payload = {k: v for k, v in llm_usage.items() if k != "records"}
        summary_payload["records_path"] = str(usage_jsonl)
        summary_payload["records_count"] = len(records)
        usage_summary.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        return {}

    return {
        "llm_usage_path": str(usage_jsonl),
        "llm_usage_summary_path": str(usage_summary),
    }


def _load_codegen_profile(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    profile_path = Path(path)
    if not profile_path.exists() or not profile_path.is_file():
        return None
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return strip_protocol_hints(payload)


def build_stage_c_backend_config(
    *,
    format_tree_path: Optional[Path],
    codegen_profile: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build backend repair/check config with the host-neutral lowering plan."""

    cfg: Dict[str, Any] = dict(codegen_profile or {})
    if isinstance(extra, dict):
        cfg.update(extra)
    if format_tree_path is not None:
        cfg["format_tree_path"] = str(format_tree_path)
        try:
            tree_json = json.loads(Path(format_tree_path).read_text(encoding="utf-8"))
            parse_ir = build_parse_ir(tree_json)
            plan = build_stage_c_lowering_plan(parse_ir, codegen_profile=codegen_profile)
            cfg["stage_c_lowering_plan"] = stage_c_lowering_plan_to_dict(plan)
        except Exception:
            cfg.setdefault("stage_c_lowering_plan_error", "unavailable")
    else:
        cfg.setdefault("format_tree_path", None)
    return cfg


def _load_api_docs_explicit_only(cfg: StageCConfig) -> str:
    """Load ONLY user-provided `api_doc_paths` (no heuristics)."""
    texts: List[str] = []
    max_total_chars = 20000
    for path in cfg.api_doc_paths or []:
        if path and Path(path).is_file():
            try:
                texts.append(Path(path).read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
        if sum(len(t) for t in texts) > max_total_chars:
            break
    return "\n\n".join(texts).strip()


