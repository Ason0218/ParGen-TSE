from __future__ import annotations

"""Build exp-constraint suites from PKG-path-generated seed messages."""

import hashlib
import json
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from pargen.protocol_knowledge.constraint_ir import is_constraint_ir_text
from pargen.protocol_knowledge.tree_utils import (
    parse_numeric_calculated_upper_range_constraint_text,
    parse_numeric_enum_constraint_text,
    parse_numeric_range_constraint_text,
)
from pargen.parser_realization.pkg_seed_generator import (
    build_pkg_constraints_catalog,
    generate_pkg_path_seed_suite,
    is_upper_boundary_id,
)
from pargen.parser_realization.target_platforms.extraction_profile import infer_ip_proto, infer_pcap_layer
from pargen.parser_realization.traffic_exp_constraint_generator import (
    _mutate_frame_protocol_bytes,
    _write_single_frame_pcap,
)


_DLT_EN10MB = 1
_ETHERTYPE_IPV4 = 0x0800
_IPPROTO_TCP = 6
_IPPROTO_UDP = 17
_SYNTHETIC_CLIENT_PORT = 49152
_SYNTHETIC_CLIENT_MAC = bytes.fromhex("020000000001")
_SYNTHETIC_SERVER_MAC = bytes.fromhex("020000000002")
_SYNTHETIC_CLIENT_IPV4 = bytes((192, 0, 2, 1))
_SYNTHETIC_SERVER_IPV4 = bytes((198, 51, 100, 1))
_PROFILE_HOST_KEYS = ("wireshark_c", "wireshark_lua", "scapy", "zeek_spicy")
# Every capture layer the extraction profiles use. The envelope (Ethernet, IPv4 header, UDP/TCP header) is built
# with valid lengths and checksums; bytes inside the PKG-described layer, including a checksum field there, take
# the values message construction assigns, because the PKG interpreter is the reference and checks only the
# rules the PKG represents (Sec. IV-D).
_SUPPORTED_PCAP_LAYERS = {"transport_payload", "ip_payload", "l3_packet", "tcp_header"}


@dataclass(frozen=True)
class SyntheticEnvelopeSpec:
    """Deterministic outer-envelope inputs derived only from the codegen profile."""

    pcap_layer: str
    ip_proto: Optional[int]
    transport: Optional[str]
    registered_ports: Tuple[int, ...] = ()
    linktype: int = _DLT_EN10MB

    @property
    def registered_port(self) -> Optional[int]:
        return self.registered_ports[0] if self.registered_ports else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance": "synthetic_stdlib_ipv4_v1",
            "source": "codegen_profile",
            "deterministic": True,
            "linktype": self.linktype,
            "linktype_name": "DLT_EN10MB",
            "network": "ipv4",
            "pcap_layer": self.pcap_layer,
            "ip_proto": self.ip_proto,
            "transport": self.transport,
            "registered_ports": list(self.registered_ports),
            "selected_registered_port": self.registered_port,
            "base_client_port": _SYNTHETIC_CLIENT_PORT,
        }


@dataclass(frozen=True)
class GeneratedPkgConstraintSuite:
    protocol: str
    suite_dir: Path
    mapping_path: Path
    pcap_layer: str
    ip_proto: Optional[int]
    positive_samples_total: int
    positive_samples_usable: int
    negative_samples: int
    cases_generated: int
    paths_total: int
    paths_enumerated_total: int
    paths_truncated: bool
    skipped_paths: int
    skipped_no_frame: int
    skipped_unsupported_frame: int
    branch_constraints_total: int
    branch_constraints_covered: int
    negative_obligations_activated: int
    negative_obligations_total: int
    negative_obligations_covered: int
    condition_constraints_total: int
    condition_constraints_covered: int
    condition_negative_obligations_total: int
    condition_negative_obligations_covered: int
    replay_validated_paths: int
    replay_unverified_paths: int
    generation_complete: bool
    generation_incomplete_reasons: Tuple[str, ...]
    independent_negative_obligations_total: int = 0
    independent_negative_obligations_covered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "suite_dir": str(self.suite_dir),
            "mapping_path": str(self.mapping_path),
            "pcap_layer": self.pcap_layer,
            "ip_proto": self.ip_proto,
            "positive_samples_total": self.positive_samples_total,
            "positive_samples_usable": self.positive_samples_usable,
            "negative_samples": self.negative_samples,
            "cases_generated": self.cases_generated,
            "paths_total": self.paths_total,
            "paths_enumerated_total": self.paths_enumerated_total,
            "paths_truncated": self.paths_truncated,
            "skipped_paths": self.skipped_paths,
            "skipped_no_frame": self.skipped_no_frame,
            "skipped_unsupported_frame": self.skipped_unsupported_frame,
            "branch_constraints_total": self.branch_constraints_total,
            "branch_constraints_covered": self.branch_constraints_covered,
            "negative_obligations_activated": self.negative_obligations_activated,
            "negative_obligations_total": self.negative_obligations_total,
            "negative_obligations_covered": self.negative_obligations_covered,
            "condition_constraints_total": self.condition_constraints_total,
            "condition_constraints_covered": self.condition_constraints_covered,
            "condition_negative_obligations_total": self.condition_negative_obligations_total,
            "condition_negative_obligations_covered": self.condition_negative_obligations_covered,
            "replay_validated_paths": self.replay_validated_paths,
            "replay_unverified_paths": self.replay_unverified_paths,
            "generation_complete": self.generation_complete,
            "generation_incomplete_reasons": list(self.generation_incomplete_reasons),
            "independent_negative_obligations_total": self.independent_negative_obligations_total,
            "independent_negative_obligations_covered": self.independent_negative_obligations_covered,
        }


def generate_exp_constraint_suite_from_pkg_paths(
    *,
    protocol: str,
    tree_path: Path,
    out_dir: Path,
    codegen_profile: Optional[Mapping[str, Any]] = None,
    max_negatives_per_positive: int = 2,
    max_paths: Optional[int] = None,
) -> GeneratedPkgConstraintSuite:
    protocol_norm = str(protocol or "").strip().lower()
    if not protocol_norm:
        raise ValueError("protocol is required")
    if max_negatives_per_positive <= 0:
        raise ValueError("max_negatives_per_positive must be > 0")

    tree_path = Path(tree_path)
    tree_bytes = tree_path.read_bytes()
    tree = json.loads(tree_bytes.decode("utf-8"))
    tree_sha256 = hashlib.sha256(tree_bytes).hexdigest()
    seed_result = generate_pkg_path_seed_suite(
        tree,
        max_negatives_per_positive=max_negatives_per_positive,
        max_paths=max_paths,
    )
    base_constraints_catalog = build_pkg_constraints_catalog(tree)

    envelope = _resolve_synthetic_envelope(codegen_profile)
    pcap_layer = envelope.pcap_layer
    ip_proto = envelope.ip_proto

    requested_suite_dir = Path(out_dir).expanduser()
    _assert_requested_output_dir_no_symlinks(requested_suite_dir)
    suite_dir = requested_suite_dir.resolve()
    samples_dir = suite_dir / "samples"
    if samples_dir.is_symlink():
        raise ValueError(f"refusing symlinked PKG suite samples directory: {samples_dir}")
    samples_dir.mkdir(parents=True, exist_ok=True)

    cases: Dict[str, Any] = {}
    physical_counterexamples: set[tuple[str, bytes]] = set()
    seed_frames = []
    seed_frame_no: Dict[str, int] = {}
    skipped_no_frame = 0
    skipped_unsupported_frame = 0
    positive_by_id = {sample.packet_id: sample for sample in seed_result.positive_samples}
    ordinal_by_id = {sample.packet_id: idx for idx, sample in enumerate(seed_result.positive_samples)}

    for ordinal, sample in enumerate(seed_result.positive_samples):
        try:
            linktype, frame = _build_synthetic_frame(
                envelope,
                layer_bytes=sample.raw_bytes,
                response=_sample_is_response(sample, tree=tree),
                ordinal=ordinal,
            )
        except Exception as exc:
            raise ValueError(
                f"failed to synthesize positive envelope for {sample.packet_id}: {type(exc).__name__}: {exc}"
            ) from exc
        seed_frames.append((linktype, frame))
        seed_frame_no[sample.packet_id] = len(seed_frames)

    positive_pcap = suite_dir / f"{protocol_norm}_pkg_positive_seeds.pcap"
    if seed_frames:
        _assert_suite_output_path(suite_dir, positive_pcap)
        _write_multi_frame_pcap(positive_pcap, seed_frames)

    if len(seed_result.negative_qualifications) != len(seed_result.negative_samples):
        raise ValueError("Every generated negative requires an isolation audit")
    for neg, qualification in zip(seed_result.negative_samples, seed_result.negative_qualifications):
        frame_no = seed_frame_no.get(neg.base_packet_id)
        if frame_no is None:
            skipped_no_frame += 1
            continue
        base_sample = positive_by_id.get(neg.base_packet_id)
        if base_sample is None:
            skipped_no_frame += 1
            continue
        try:
            linktype, frame = _build_synthetic_frame(
                envelope,
                layer_bytes=neg.raw_bytes,
                response=_sample_is_response(base_sample, tree=tree),
                ordinal=ordinal_by_id[neg.base_packet_id],
            )
        except Exception:
            skipped_unsupported_frame += 1
            continue

        case_idx = len(cases) + 1
        case_name = f"pkg_path_neg_{case_idx:04d}"
        pcap_name = f"{case_name}.pcap"
        pcap_path = samples_dir / pcap_name
        _assert_suite_output_path(suite_dir, pcap_path)
        _write_single_frame_pcap(pcap_path, linktype=linktype, frame=frame)
        cases[case_name] = {
            "pcap": f"samples/{pcap_name}",
            "frame_no": 1,
            "seed_frame_no": int(frame_no),
            "base_packet_id": neg.base_packet_id,
            "target_constraints": list(neg.expected_violated_constraints),
            "mutation_note": neg.note,
            "counterexample_qualification": qualification,
        }
        physical_counterexamples.add((str(neg.base_packet_id), bytes(neg.raw_bytes)))

    mapping_path = suite_dir / f"{protocol_norm}_pkg_negpcaps_constraints_map.json"
    candidate_negative_obligations = {
        str(cid)
        for sample in seed_result.positive_samples
        for cid in sample.activated_constraints
        if not str(cid).startswith("edge:condition_on:")
    }
    generated_runtime_obligations = {
        str(cid)
        for case in cases.values()
        for cid in case.get("target_constraints") or []
    }
    generated_negative_obligations = {
        cid
        for cid in generated_runtime_obligations
        if not cid.startswith("edge:condition_on:")
    }
    activated_branch_ids = {
        str(cid)
        for sample in seed_result.positive_samples
        for cid in sample.activated_constraints
        if str(cid).startswith("edge:condition_on:")
    }
    positive_only_branch_ids = set(seed_result.positive_only_branch_constraints)
    required_condition_obligations = {
        cid
        for cid in activated_branch_ids - positive_only_branch_ids
        if bool(base_constraints_catalog.get(cid, {}).get("present_in_codegen_contract"))
    }
    generated_condition_obligations = {
        cid
        for cid in generated_runtime_obligations
        if cid.startswith("edge:condition_on:")
    }
    positive_only_or_unmutatable = list(seed_result.positive_only_node_constraints)
    # Upper-boundary positives re-materialize an already covered path.
    materialized_paths = sum(1 for sample in seed_result.positive_samples if not is_upper_boundary_id(sample.packet_id))
    path_coverage = {
        "representative_paths": {
            "total": seed_result.paths_enumerated_total,
            "selected": seed_result.paths_total,
            "truncated": seed_result.paths_truncated,
            "materialized": materialized_paths,
            "boundary_positives": len(seed_result.positive_samples) - materialized_paths,
            "skipped": len(seed_result.skipped_paths),
            "coverage": (
                materialized_paths / seed_result.paths_enumerated_total
                if seed_result.paths_enumerated_total
                else 0.0
            ),
        },
        "branch_variants": {
            "total": seed_result.branch_constraints_total,
            "covered": seed_result.branch_constraints_covered,
            "coverage": (
                seed_result.branch_constraints_covered / seed_result.branch_constraints_total
                if seed_result.branch_constraints_total
                else 0.0
            ),
        },
        "condition_edges": {
            "total": seed_result.condition_constraints_total,
            "activated": seed_result.condition_constraints_covered,
            "coverage": (
                seed_result.condition_constraints_covered / seed_result.condition_constraints_total
                if seed_result.condition_constraints_total
                else 0.0
            ),
            "negative_conformance_required": len(required_condition_obligations),
            "counterexamples_generated": len(generated_condition_obligations),
            "negative_coverage": (
                len(generated_condition_obligations) / len(required_condition_obligations)
                if required_condition_obligations
                else 0.0
            ),
            "diagnostic_only": False,
            "positive_only": list(seed_result.positive_only_branch_constraints),
        },
        "tree_replay": {
            "attempted": seed_result.paths_total,
            "enumerated": seed_result.paths_enumerated_total,
            "truncated": seed_result.paths_truncated,
            "validated": seed_result.replay_validated_paths,
            "rejected": len(seed_result.skipped_paths),
            "replay_rejected": len(seed_result.replay_unverified_paths),
            "coverage": (
                seed_result.replay_validated_paths / seed_result.paths_total
                if seed_result.paths_total
                else 0.0
            ),
            "rejected_paths": [
                {"path_id": item.path_id, "reason": item.reason}
                for item in seed_result.skipped_paths
            ],
        },
        "negative_obligations": {
            "scope": "node_and_length_constraints_excluding_condition_on",
            "activated_total": seed_result.negative_obligations_activated,
            "mutatable_total": seed_result.negative_obligations_mutatable,
            "generated": len(generated_negative_obligations),
            "coverage": (
                len(generated_negative_obligations) / seed_result.negative_obligations_mutatable
                if seed_result.negative_obligations_mutatable
                else 0.0
            ),
            "positive_only_or_unmutatable": positive_only_or_unmutatable,
            "unsupported": list(seed_result.unsupported_negative_constraints),
        },
    }
    constraints_catalog = _annotated_constraints_catalog(
        tree,
        base_catalog=base_constraints_catalog,
        activated_ids=candidate_negative_obligations.union(activated_branch_ids),
        generated_ids=generated_runtime_obligations,
        positive_only_ids=set(seed_result.positive_only_branch_constraints).union(positive_only_or_unmutatable),
        unsupported_ids=set(seed_result.unsupported_negative_constraints),
    )
    required_negative_ids = {
        cid for cid, spec in constraints_catalog.items() if _catalog_spec_requires_negative_conformance(spec)
    }
    unlowered_required_ids = {
        cid
        for cid in required_negative_ids
        if not bool(constraints_catalog[cid].get("present_in_codegen_contract"))
    }
    lowered_required_ids = required_negative_ids - unlowered_required_ids
    independent_ids = {
        str(cid) for case in cases.values()
        if case["counterexample_qualification"].get("independent_target_test") is True
        for cid in case["target_constraints"]
    }
    unverified_independent_ids = required_negative_ids - independent_ids
    path_coverage["independent_negative_obligations"] = {
        "required": len(required_negative_ids),
        "verified": len(independent_ids & required_negative_ids),
        "coverage": len(independent_ids & required_negative_ids) / len(required_negative_ids)
            if required_negative_ids else 0.0,
        "unverified": sorted(unverified_independent_ids),
        "multi_violation_cases": sum(c["counterexample_qualification"]["status"] == "multi_violation"
                                     for c in cases.values()),
        "unverified_cases": sum(c["counterexample_qualification"]["status"] == "unverified"
                                for c in cases.values()),
        "candidate_counts_are_not_independent_coverage": True,
    }
    activated_catalog_ids = candidate_negative_obligations.union(activated_branch_ids)
    unactivated_required_ids = lowered_required_ids - activated_catalog_ids
    missing_counterexample_ids = lowered_required_ids - generated_runtime_obligations
    path_coverage["tree_catalog"] = {
        "total": len(constraints_catalog),
        "negative_conformance_required": len(required_negative_ids),
        "negative_conformance_unlowered": sorted(unlowered_required_ids),
        "negative_conformance_not_activated": sorted(unactivated_required_ids),
        "negative_conformance_without_counterexample": sorted(missing_counterexample_ids),
        "activated_on_replay_valid_paths": sum(
            1 for spec in constraints_catalog.values() if spec.get("activated_on_generated_path")
        ),
        "negative_obligations_activated": len(activated_catalog_ids),
        "non_condition_obligations_activated": len(candidate_negative_obligations),
        "negative_obligations_with_counterexample": len(generated_runtime_obligations),
        "unique_counterexamples": len(physical_counterexamples),
        "target_specific_cases": len(cases),
        "multi_target_counterexamples": sum(
            1 for case in cases.values() if len(case.get("target_constraints") or []) > 1
        ),
    }
    generation_incomplete_reasons: list[str] = []
    if unverified_independent_ids:
        generation_incomplete_reasons.append(
            "negative obligations lack independently isolated counterexamples: "
            f"count={len(unverified_independent_ids)} sample={sorted(unverified_independent_ids)[:5]}"
        )
    if seed_result.paths_enumerated_total <= 0:
        generation_incomplete_reasons.append("representative path enumeration produced no paths")
    if not seed_result.positive_samples:
        generation_incomplete_reasons.append("no replay-valid representative positive samples were generated")
    if seed_result.paths_truncated:
        generation_incomplete_reasons.append(
            "representative path enumeration was truncated by max_paths: "
            f"selected={seed_result.paths_total}/{seed_result.paths_enumerated_total}"
        )
    if materialized_paths != seed_result.paths_total or seed_result.skipped_paths:
        generation_incomplete_reasons.append(
            "representative path materialization is incomplete: "
            f"materialized={materialized_paths}/{seed_result.paths_total}"
        )
    if seed_result.branch_constraints_covered != seed_result.branch_constraints_total:
        generation_incomplete_reasons.append(
            "branch coverage is incomplete: "
            f"covered={seed_result.branch_constraints_covered}/{seed_result.branch_constraints_total}"
        )
    if seed_result.condition_constraints_covered != seed_result.condition_constraints_total:
        generation_incomplete_reasons.append(
            "condition-edge coverage is incomplete: "
            f"covered={seed_result.condition_constraints_covered}/{seed_result.condition_constraints_total}"
        )
    if len(generated_negative_obligations) != seed_result.negative_obligations_mutatable:
        generation_incomplete_reasons.append(
            "negative obligation generation is incomplete: "
            f"generated={len(generated_negative_obligations)}/{seed_result.negative_obligations_mutatable}"
        )
    if unactivated_required_ids:
        generation_incomplete_reasons.append(
            "executable negative obligations were not activated on representative paths: "
            f"count={len(unactivated_required_ids)} sample={sorted(unactivated_required_ids)[:5]}"
        )
    if missing_counterexample_ids:
        generation_incomplete_reasons.append(
            "executable runtime obligations have no exact negative counterexample: "
            f"count={len(missing_counterexample_ids)} sample={sorted(missing_counterexample_ids)[:5]}"
        )
    if unlowered_required_ids:
        generation_incomplete_reasons.append(
            "executable tree constraints are absent from the Stage-C codegen contract: "
            f"count={len(unlowered_required_ids)} sample={sorted(unlowered_required_ids)[:5]}"
        )
    if skipped_no_frame or skipped_unsupported_frame:
        generation_incomplete_reasons.append(
            "synthetic envelope generation skipped cases: "
            f"missing_frame={skipped_no_frame} unsupported_frame={skipped_unsupported_frame}"
        )

    runtime_disable_protocols = _profile_runtime_disable_protocols(codegen_profile)
    mapping_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol_norm,
        "generator": "pkg_path_conformance_counterexample_v4_targeted_isolation",
        "negative_construction_audit": list(seed_result.negative_construction_audit),
        "base_pcap": positive_pcap.name if seed_frames else None,
        "base_pcap_scope": "suite_local_required",
        "positive_seed_scope": "all_replay_valid_representative_paths",
        "positive_seeds": [
            {
                "packet_id": sample.packet_id,
                "path_id": sample.path_id,
                "frame_no": int(seed_frame_no[sample.packet_id]),
            }
            for sample in seed_result.positive_samples
            if sample.packet_id in seed_frame_no
        ],
        "generation_completeness": {
            "complete": not generation_incomplete_reasons,
            "reasons": generation_incomplete_reasons,
        },
        "binary_require_exact_target_tags": True,
        "counterexample_semantics": "target_violation_candidate_plus_independent_isolation_audit",
        "envelope": envelope.to_dict(),
        "tree": str(tree_path),
        "tree_sha256": tree_sha256,
        "pcap_layer": pcap_layer,
        "ip_proto": ip_proto,
        "coverage": path_coverage,
        "constraints_from_tree": constraints_catalog,
        "cases": cases,
    }
    if runtime_disable_protocols:
        mapping_payload["tshark"] = {
            "disable_protocols": list(runtime_disable_protocols),
        }
    _assert_suite_output_path(suite_dir, mapping_path)
    mapping_path.write_text(json.dumps(mapping_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = GeneratedPkgConstraintSuite(
        protocol=protocol_norm,
        suite_dir=suite_dir,
        mapping_path=mapping_path,
        pcap_layer=pcap_layer,
        ip_proto=ip_proto,
        positive_samples_total=seed_result.paths_total,
        positive_samples_usable=len(seed_frame_no),
        negative_samples=len(seed_result.negative_samples),
        cases_generated=len(cases),
        paths_total=seed_result.paths_total,
        paths_enumerated_total=seed_result.paths_enumerated_total,
        paths_truncated=seed_result.paths_truncated,
        skipped_paths=len(seed_result.skipped_paths),
        skipped_no_frame=skipped_no_frame,
        skipped_unsupported_frame=skipped_unsupported_frame,
        branch_constraints_total=seed_result.branch_constraints_total,
        branch_constraints_covered=seed_result.branch_constraints_covered,
        negative_obligations_activated=seed_result.negative_obligations_activated,
        negative_obligations_total=seed_result.negative_obligations_mutatable,
        negative_obligations_covered=len(generated_negative_obligations),
        condition_constraints_total=seed_result.condition_constraints_total,
        condition_constraints_covered=seed_result.condition_constraints_covered,
        condition_negative_obligations_total=len(required_condition_obligations),
        condition_negative_obligations_covered=len(generated_condition_obligations),
        replay_validated_paths=seed_result.replay_validated_paths,
        replay_unverified_paths=len(seed_result.replay_unverified_paths),
        generation_complete=not generation_incomplete_reasons,
        generation_incomplete_reasons=tuple(generation_incomplete_reasons),
        independent_negative_obligations_total=len(required_negative_ids),
        independent_negative_obligations_covered=len(independent_ids & required_negative_ids),
    )
    summary_path = suite_dir / "summary.json"
    _assert_suite_output_path(suite_dir, summary_path)
    summary_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _assert_suite_output_path(suite_dir: Path, output_path: Path) -> None:
    """Reject lexical escapes and symlink traversal for generated suite outputs."""

    suite_root = Path(suite_dir).resolve()
    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = suite_root / candidate

    try:
        relative = candidate.relative_to(suite_root)
    except ValueError as exc:
        raise ValueError(f"refusing PKG suite output outside suite directory: {candidate}") from exc
    if not relative.parts or any(part == ".." for part in relative.parts):
        raise ValueError(f"refusing unsafe PKG suite output path: {candidate}")

    current = suite_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"refusing symlinked PKG suite output path: {current}")

    try:
        candidate.parent.resolve().relative_to(suite_root)
    except ValueError as exc:
        raise ValueError(f"refusing PKG suite output outside suite directory: {candidate}") from exc


def _assert_requested_output_dir_no_symlinks(requested_dir: Path) -> None:
    """Reject pre-existing symlinks below a trusted cwd/temp path anchor."""

    requested = Path(requested_dir)
    if ".." in requested.parts:
        raise ValueError(f"refusing unsafe PKG suite output directory traversal: {requested}")
    absolute = requested.absolute()
    anchors = (Path.cwd().absolute(), Path(tempfile.gettempdir()).absolute())
    anchor: Optional[Path] = None
    for candidate in anchors:
        try:
            absolute.relative_to(candidate)
        except ValueError:
            continue
        if anchor is None or len(candidate.parts) > len(anchor.parts):
            anchor = candidate
    if anchor is None:
        anchor = Path(absolute.anchor)

    relative = absolute.relative_to(anchor)
    current = anchor
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"refusing symlinked PKG suite output directory component: {current}")


def _resolve_synthetic_envelope(
    codegen_profile: Optional[Mapping[str, Any]],
) -> SyntheticEnvelopeSpec:
    """Resolve all outer-envelope parameters without consulting captured traffic."""

    profile = codegen_profile if isinstance(codegen_profile, Mapping) else {}
    pcap_layer = infer_pcap_layer(profile, host=None)
    if pcap_layer not in _SUPPORTED_PCAP_LAYERS:
        raise ValueError(
            f"tree-only PKG synthesis does not know pcap_layer={pcap_layer!r}"
        )

    registrations = []
    for host in _PROFILE_HOST_KEYS:
        section = profile.get(host)
        if not isinstance(section, Mapping):
            continue
        registration = section.get("registration")
        if isinstance(registration, Mapping):
            registrations.append((host, registration))

    transport_candidates = set()
    registered_ports = set()
    ethertype_candidates = set()
    for _host, registration in registrations:
        transport = str(registration.get("transport") or "").strip().lower()
        table = str(registration.get("dissector_table") or "").strip().lower()
        if transport in {"tcp", "udp"}:
            transport_candidates.add(transport)
        if table in {"tcp.port", "tcp.port_range"}:
            transport_candidates.add("tcp")
        elif table in {"udp.port", "udp.port_range"}:
            transport_candidates.add("udp")

        if transport in {"tcp", "udp"} or table.startswith(("tcp.port", "udp.port")):
            for port in _registration_ports(registration):
                registered_ports.add(port)

        selector_type = str(registration.get("selector_type") or "").strip().lower()
        if selector_type == "ethertype" or table == "ethertype":
            ethertype = _coerce_int(registration.get("selector"))
            if ethertype is not None:
                ethertype_candidates.add(ethertype)

    if len(transport_candidates) > 1:
        raise ValueError(f"ambiguous transport registrations: {sorted(transport_candidates)}")
    transport = next(iter(transport_candidates), None)

    if pcap_layer == "transport_payload":
        if transport not in {"tcp", "udp"}:
            raise ValueError("transport_payload requires an explicit TCP or UDP codegen registration")
        if not registered_ports:
            raise ValueError("transport_payload requires at least one registered TCP/UDP port")
        ip_proto = _IPPROTO_TCP if transport == "tcp" else _IPPROTO_UDP
        inferred = infer_ip_proto(profile, host=None)
        if inferred is not None and inferred != ip_proto:
            raise ValueError(f"transport/ip_proto conflict: transport={transport}, ip_proto={inferred}")
    elif pcap_layer == "tcp_header":
        transport = "tcp"
        inferred = infer_ip_proto(profile, host=None)
        if inferred is not None and inferred != _IPPROTO_TCP:
            raise ValueError(f"tcp_header requires ip_proto=6, got {inferred}")
        ip_proto = _IPPROTO_TCP
        registered_ports.clear()
    elif pcap_layer == "ip_payload":
        ip_proto = infer_ip_proto(profile, host=None)
        if ip_proto is None:
            raise ValueError("ip_payload requires an explicit IP protocol number in the codegen profile")
        transport = None
        registered_ports.clear()
    else:
        if ethertype_candidates and ethertype_candidates != {_ETHERTYPE_IPV4}:
            raise ValueError(f"only IPv4 l3_packet synthesis is supported, got ethertypes={sorted(ethertype_candidates)}")
        ip_proto = None
        transport = None
        registered_ports.clear()

    return SyntheticEnvelopeSpec(
        pcap_layer=pcap_layer,
        ip_proto=ip_proto,
        transport=transport,
        registered_ports=tuple(sorted(registered_ports)),
    )


def _profile_runtime_disable_protocols(
    codegen_profile: Optional[Mapping[str, Any]],
) -> Tuple[str, ...]:
    """Preserve host runtime isolation required by the effective codegen profile."""

    profile = codegen_profile if isinstance(codegen_profile, Mapping) else {}
    values: set[str] = set()

    def _collect(raw: Any) -> None:
        if isinstance(raw, bool) or raw is None:
            return
        items = raw if isinstance(raw, (list, tuple, set)) else (raw,)
        for item in items:
            value = str(item or "").strip()
            if value:
                values.add(value)

    keys = (
        "runtime_disable_protocols",
        "runtime_disable_protocol",
        "disable_builtin_protocols",
        "disable_builtin_protocol",
        "disable_protocols",
        "disable_protocol",
    )
    for host in _PROFILE_HOST_KEYS:
        section = profile.get(host)
        if not isinstance(section, Mapping):
            continue
        registration = section.get("registration")
        if not isinstance(registration, Mapping):
            continue
        for key in keys:
            _collect(registration.get(key))

    return tuple(sorted(values))


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip(), 0)
    except Exception:
        return None


def _registration_ports(registration: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = registration.get("ports")
    if raw is None or raw == [] or raw == ():
        raw = registration.get("port")
    values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
    ports = []
    for value in values:
        port = _coerce_int(value)
        if port is not None and 0 < port <= 65535:
            ports.append(port)
    return tuple(sorted(set(ports)))


def _sample_is_response(sample: Any, *, tree: Optional[Mapping[str, Any]] = None) -> bool:
    explicit_message_type = _sample_tree_message_type(sample, tree=tree)
    if explicit_message_type is not None:
        return explicit_message_type == "response"

    path_id = str(getattr(sample, "path_id", "") or "").lower()
    if any(token in path_id for token in ("response", "reply", "server_to_client")):
        return True
    return False


def _sample_tree_message_type(
    sample: Any,
    *,
    tree: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Resolve request/response direction from the selected tree scope."""

    variant_id = getattr(sample, "variant_id", None)
    if variant_id is None or not isinstance(tree, Mapping):
        return None
    payload = tree.get("protocol_tree")
    if not isinstance(payload, Mapping):
        payload = tree
    nodes = {
        str(node.get("node_id")): node
        for node in (payload.get("nodes") or [])
        if isinstance(node, Mapping) and node.get("node_id") is not None
    }
    current: Optional[str] = str(variant_id)
    seen: set[str] = set()
    directions: set[str] = set()
    while current is not None and current not in seen:
        seen.add(current)
        node = nodes.get(current)
        if node is None:
            break
        message_type = str(node.get("message_type") or "").strip().lower()
        if message_type in {"request", "response"}:
            directions.add(message_type)
        parent_id = node.get("parent_id")
        current = None if parent_id is None else str(parent_id)
    if len(directions) > 1:
        raise ValueError(
            f"conflicting request/response message_type values for variant {variant_id}: "
            f"{sorted(directions)}"
        )
    return next(iter(directions)) if directions else None


def _build_synthetic_frame(
    envelope: SyntheticEnvelopeSpec,
    *,
    layer_bytes: bytes,
    response: bool,
    ordinal: int,
) -> Tuple[int, bytes]:
    """Wrap tree-generated bytes in a deterministic Ethernet/IPv4 envelope."""

    linktype, base_frame = _build_synthetic_base_frame(envelope, response=response, ordinal=ordinal)
    frame = _mutate_frame_protocol_bytes(
        frame=base_frame,
        linktype=linktype,
        pcap_layer=envelope.pcap_layer,
        ip_proto=envelope.ip_proto,
        new_layer_bytes=bytes(layer_bytes),
    )
    return linktype, frame


def _build_synthetic_base_frame(
    envelope: SyntheticEnvelopeSpec,
    *,
    response: bool,
    ordinal: int,
) -> Tuple[int, bytes]:
    if response:
        src_mac, dst_mac = _SYNTHETIC_SERVER_MAC, _SYNTHETIC_CLIENT_MAC
        src_ip, dst_ip = _SYNTHETIC_SERVER_IPV4, _SYNTHETIC_CLIENT_IPV4
    else:
        src_mac, dst_mac = _SYNTHETIC_CLIENT_MAC, _SYNTHETIC_SERVER_MAC
        src_ip, dst_ip = _SYNTHETIC_CLIENT_IPV4, _SYNTHETIC_SERVER_IPV4

    proto = envelope.ip_proto
    l4 = b""
    if envelope.pcap_layer == "transport_payload":
        registered_port = envelope.registered_port
        if registered_port is None or envelope.transport not in {"tcp", "udp"}:
            raise ValueError("transport envelope is incomplete")
        client_port = _synthetic_client_port(registered_port, ordinal)
        sport, dport = (registered_port, client_port) if response else (client_port, registered_port)
        if envelope.transport == "udp":
            proto = _IPPROTO_UDP
            l4 = struct.pack("!HHHH", sport, dport, 8, 0)
        else:
            proto = _IPPROTO_TCP
            seq = 1 + max(0, int(ordinal))
            ack = 1
            off_flags = (5 << 12) | 0x18
            l4 = struct.pack("!HHIIHHHH", sport, dport, seq, ack, off_flags, 65535, 0, 0)
    elif envelope.pcap_layer == "tcp_header":
        proto = _IPPROTO_TCP
        client_port = _synthetic_client_port(80, ordinal)
        sport, dport = (80, client_port) if response else (client_port, 80)
        l4 = struct.pack("!HHIIHHHH", sport, dport, 1, 1, (5 << 12) | 0x18, 65535, 0, 0)
    elif envelope.pcap_layer == "ip_payload":
        if proto is None:
            raise ValueError("ip_payload envelope is missing ip_proto")
    elif envelope.pcap_layer == "l3_packet":
        # The placeholder is used only to let the existing mutation helper retain
        # the deterministic Ethernet prefix. The tree owns the replacement L3 bytes.
        proto = 253
    else:  # pragma: no cover - guarded by _resolve_synthetic_envelope
        raise ValueError(f"unsupported pcap_layer={envelope.pcap_layer!r}")

    if proto is None or not (0 <= int(proto) <= 255):
        raise ValueError(f"invalid synthetic IPv4 protocol number: {proto!r}")

    total_len = 20 + len(l4)
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_len,
        int(ordinal) & 0xFFFF,
        0,
        64,
        int(proto),
        0,
        src_ip,
        dst_ip,
    )
    ethernet = dst_mac + src_mac + struct.pack("!H", _ETHERTYPE_IPV4)
    return envelope.linktype, ethernet + ip_header + l4


def _synthetic_client_port(registered_port: int, ordinal: int) -> int:
    port = _SYNTHETIC_CLIENT_PORT + (max(0, int(ordinal)) % (65536 - _SYNTHETIC_CLIENT_PORT))
    if port == int(registered_port):
        port = _SYNTHETIC_CLIENT_PORT if port != _SYNTHETIC_CLIENT_PORT else _SYNTHETIC_CLIENT_PORT + 1
    return port


def _write_multi_frame_pcap(path: Path, frames: Sequence[Tuple[int, bytes]]) -> None:
    if not frames:
        return
    linktype = int(frames[0][0])
    chunks = [struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype)]
    for idx, (_link, frame) in enumerate(frames):
        chunks.append(struct.pack("<IIII", idx, 0, len(frame), len(frame)))
        chunks.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))


def _annotated_constraints_catalog(
    tree: Mapping[str, Any],
    *,
    base_catalog: Optional[Mapping[str, Mapping[str, Any]]] = None,
    activated_ids: Sequence[str] = (),
    generated_ids: Sequence[str] = (),
    positive_only_ids: Sequence[str] = (),
    unsupported_ids: Sequence[str] = (),
) -> Dict[str, Dict[str, Any]]:
    """Keep the complete tree metadata and add runtime generation status."""

    catalog = (
        {str(cid): dict(spec) for cid, spec in base_catalog.items()}
        if base_catalog is not None
        else build_pkg_constraints_catalog(tree)
    )
    activated = {str(cid) for cid in activated_ids}
    generated = {str(cid) for cid in generated_ids}
    positive_only = {str(cid) for cid in positive_only_ids}
    unsupported = {str(cid) for cid in unsupported_ids}
    missing = sorted((activated | generated | positive_only | unsupported) - set(catalog))
    if missing:
        raise ValueError(f"PKG constraint IDs missing from node-scoped tree catalog: {missing}")
    missing_contract_tags = sorted(
        cid for cid in generated if not str(catalog[cid].get("contract_tag") or "").strip()
    )
    if missing_contract_tags:
        raise ValueError(
            "generated PKG targets are absent from the executable Stage-C contract: "
            f"{missing_contract_tags}"
        )

    out: Dict[str, Dict[str, Any]] = {}
    for cid, raw_spec in catalog.items():
        spec = dict(raw_spec)
        is_positive_only = cid in positive_only
        spec["activated_on_generated_path"] = cid in activated
        spec["negative_case_generated"] = cid in generated
        spec["positive_only"] = is_positive_only
        spec["negative_generation_unsupported"] = cid in unsupported
        # Tree-only suites require the exact node/edge contract tag. Generic
        # malformed/violation messages must not satisfy a different target.
        spec["strict_contract_tag"] = cid in generated
        if is_positive_only:
            status = "positive_only"
        elif cid in generated:
            status = "counterexample_generated"
        elif cid in unsupported:
            status = "activated_but_unsupported"
        elif cid in activated:
            status = "activated_without_counterexample"
        else:
            status = "not_activated_on_representative_paths"
        spec["generation_status"] = status
        out[cid] = spec
    return out


def _catalog_spec_requires_negative_conformance(spec: Mapping[str, Any]) -> bool:
    """Return whether a raw tree obligation needs a negative runtime case."""

    if bool(spec.get("positive_only")):
        return False
    if bool(spec.get("present_in_codegen_contract")):
        return True
    if str(spec.get("kind") or "") == "edge_constraint":
        # length_of is executable tree semantics even when contract lowering
        # failed.  Fallback condition_on edges may be intentionally omitted.
        return str(spec.get("rel") or "").strip().lower() == "length_of"
    if str(spec.get("kind") or "") != "node_constraint":
        return False
    if str(spec.get("node_type") or "").strip().lower() not in {
        "field",
        "selector",
        "payload",
        "length",
        "type",
        "checksum",
        "count",
    }:
        return False
    raw = str(spec.get("constraint") or "").strip()
    lowered = raw.lower()
    return bool(
        parse_numeric_enum_constraint_text(raw) is not None
        or parse_numeric_range_constraint_text(raw) is not None
        or parse_numeric_calculated_upper_range_constraint_text(raw) is not None
        or lowered.startswith("complement:")
        or lowered.startswith("formula:")
        or is_constraint_ir_text(raw)
    )
