"""Loss-accounted conformance evidence, separate from acceptance/scoring."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

POLICY = "conformance_detection_and_attribution_v3_isolation_v2"
TAG = re.compile(r"\bBS_CONSTRAINT\s+[A-Za-z0-9_:.\-]+")


def evaluation_gaps(suite_result):
    """Keep unqualified tests in results/denominators, not in parser repair."""
    gaps = []
    for case in suite_result.results:
        q = (getattr(case, "observations", None) or {}).get("counterexample_qualification") or {}
        if q.get("independent_target_test") is not True:
            gaps.append(dict(case=case.case, target_constraints=list(case.target_constraints),
                kind="counterexample_not_independently_qualified", qualification=q, parser_fault_proven=False))
    return gaps


def metric_summary(suite_result):
    names = ("scoring_policy", "acceptance_semantics", "confusion_semantics", "negative_detection_success",
             "negative_target_diagnostic_success", "negative_independently_verified", "negative_isolation_unverified")
    return {name: getattr(suite_result, name, None) for name in names}


def matches_expectation(expectation, output, catalog):
    strict = bool((catalog.get(expectation.constraint_id) or {}).get("strict_contract_tag"))
    for needle in expectation.patterns_any:
        if not needle:
            continue
        if strict:
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(needle) + r"(?![A-Za-z0-9_])", output):
                return True
        elif needle in output:
            return True
    return False


def observe_case(exp, case_info, output, returncode, *, logs_dir=None, pcap_layer=None):
    tags = sorted(set(TAG.findall(output)))
    diagnostics = [line.strip() for line in output.splitlines()
                   if "BS_CONSTRAINT" in line or "expert info" in line.lower()]
    evidence = dict(policy=POLICY, observed_tags=tags,
        output_sha256=hashlib.sha256(output.encode()).hexdigest(), output_chars=len(output),
        output_excerpt=output[:2048], output_excerpt_truncated=len(output) > 2048,
        diagnostic_lines=[line[:512] for line in diagnostics[:12]],
        diagnostic_lines_omitted=max(0, len(diagnostics)-12), returncode=returncode,
        mutation_note=str(case_info.get("mutation_note") or ""),
        target_patterns={c.constraint_id: list(c.patterns_any) for c in exp.constraints})
    packet = dict(status="unsupported_layer", pcap_layer=pcap_layer)
    if pcap_layer == "transport_payload":
        from .target_platforms.pcap_payloads import extract_tcp_udp_payloads_with_ports_from_pcap
        try:
            rows = extract_tcp_udp_payloads_with_ports_from_pcap(exp.pcap, [exp.frame_no])
            if exp.frame_no not in rows:
                packet = dict(status="frame_payload_not_found")
            else:
                raw, transport, _sport, _dport = rows[exp.frame_no]
                packet = dict(status="observed", byte_length=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(), hex_prefix=raw[:256].hex(),
                    prefix_truncated=len(raw) > 256, transport=transport)
        except (OSError, ValueError, TypeError) as exc:
            packet = dict(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
    evidence["packet_payload"] = packet
    qualification = case_info.get("counterexample_qualification")
    if not isinstance(qualification, Mapping):
        qualification = dict(status="unverified", independent_target_test=False,
                             reason="mapping_has_no_isolation_audit")
    else:
        qualification = dict(qualification)
        from .pkg_negative_qualification import POLICY as ISOLATION_POLICY
        if (qualification.get("policy") != ISOLATION_POLICY
                or packet.get("status") != "observed"
                or qualification.get("payload_sha256") != packet.get("sha256")
                or sorted(qualification.get("target_constraints") or []) !=
                    sorted(c.constraint_id for c in exp.constraints)):
            qualification.update(status="unverified", independent_target_test=False,
                                 reason="isolation_audit_not_bound_to_executed_payload_and_targets")
        elif qualification.get("independent_target_test") is True and (
                qualification.get("status") != "isolated"
                or qualification.get("target_violation_proven") is not True
                or qualification.get("source_positive_qualified") is not True
                or qualification.get("coincident_constraints")
                or qualification.get("unresolved_constraints")
                or sorted(qualification.get("known_violated_constraints") or []) !=
                    sorted(qualification.get("target_constraints") or [])):
            qualification.update(status="unverified", independent_target_test=False,
                                 reason="inconsistent_isolation_audit")
    evidence["counterexample_qualification"] = qualification
    if logs_dir is not None:
        folder = Path(logs_dir) / "observations"
        if folder.is_symlink():
            raise ValueError("Linked conformance observations directory")
        folder.mkdir(parents=True, exist_ok=True)
        # Never interpolate an untrusted case name as a path; never overwrite a
        # prior run's evidence. Full output lives here, not in the model prompt.
        key = hashlib.sha256(f"{exp.case}:{exp.frame_no}".encode()).hexdigest()
        path = folder / (key + ".json")
        receipt = dict(case=exp.case, frame_no=exp.frame_no, pcap=str(exp.pcap),
                       evidence=evidence, raw_output=output)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        evidence["raw_output_artifact"] = str(path)
    return evidence
