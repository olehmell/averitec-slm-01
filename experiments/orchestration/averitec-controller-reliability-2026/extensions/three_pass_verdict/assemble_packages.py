"""Seal a hash-pinned, mixed-source selector matrix into gold-free packages."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

import downstream
import selector_passes

ARMS = tuple(selector_passes.ARMS)
PASSES = (1, 2, 3)
PACKAGE_SCHEMA = "three-pass-verdict-packages/v1"
MANIFEST_SCHEMA = "three-pass-composite-selector-inputs/v1"
FROZEN_CANDIDATES_SHA256 = "219c745aac6f50501a55c4f2511d383505317fd75338b0efaed25f44bbe23b98"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
JOURNALS = {"receipt.json", "run-metadata.json", "preflight.json",
            "intents.jsonl", "results.jsonl", "traces.jsonl"}
ENTRY_KEYS = {"arm", "pass", "source", "candidates", "selector_launch",
              "selector_gate", "selector_output", "accounting"}
UNAVAILABLE_ENTRY_KEYS = {"arm", "pass", "status", "source", "candidates", "selector_launch",
                          "astra_no_go", "unavailable", "accounting"}
FINAL_SLURM_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file(spec: Any) -> Path:
    if (not isinstance(spec, dict) or set(spec) != {"path", "sha256"}
            or not isinstance(spec["path"], str) or not Path(spec["path"]).is_absolute()
            or not isinstance(spec["sha256"], str) or not HEX64.fullmatch(spec["sha256"])):
        raise ValueError("composite_file_spec")
    path = Path(spec["path"])
    if not path.is_file() or path.is_symlink() or _sha256(path) != spec["sha256"]:
        raise ValueError(f"composite_file_drift:{path}")
    return path


def _output(spec: Any) -> Path:
    if (not isinstance(spec, dict) or set(spec) != {"path", "files"}
            or not isinstance(spec["path"], str) or not Path(spec["path"]).is_absolute()
            or not isinstance(spec["files"], dict) or set(spec["files"]) != JOURNALS):
        raise ValueError("composite_output_spec")
    directory = Path(spec["path"])
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("composite_output_directory")
    for name, expected in spec["files"].items():
        _file({"path": str(directory / name), "sha256": expected})
    return directory


def _source_receipt(path: Path, entry: dict) -> None:
    receipt = _json(path)
    launch = _json(Path(entry["selector_launch"]["path"]))
    code = launch.get("code_sha256")
    if (not isinstance(receipt, dict) or receipt.get("gold_staged") is not False
            or receipt.get("manifest_sha256") != entry["selector_launch"]["sha256"]
            or receipt.get("gate_sha256") != entry.get("selector_gate", entry.get("astra_no_go"))["sha256"]
            or not isinstance(receipt.get("inputs_sha256"), dict)
            or receipt["inputs_sha256"].get("candidates.json") != entry["candidates"]["sha256"]
            or not isinstance(code, dict) or not code
            or receipt.get("source_commit") != launch.get("source_commit")
            or not isinstance(receipt.get("source_sha256"), dict)
            or any(receipt["source_sha256"].get(name) != digest for name, digest in code.items())):
        raise ValueError("composite_source_binding")


def _unavailable(entry: dict, launch: Path, candidates: Path) -> dict:
    """Require a scoped NO-GO, a no-run receipt, and zero ledger calls."""
    if entry["arm"] not in {"jev", "gemini"}:
        raise ValueError("composite_unavailable_hosted_only")
    gate = _json(_file(entry["astra_no_go"]))
    if (not isinstance(gate, dict)
            or set(gate) != {"schema", "decision", "reviewer", "manifest_sha256",
                             "allowed_arms", "allowed_passes", "reason"}
            or gate["schema"] != "three-pass-selector-no-go/v1"
            or gate["decision"] != "no_go" or gate["reviewer"] != "gpt-6-astra"
            or gate["manifest_sha256"] != entry["selector_launch"]["sha256"]
            or gate["allowed_arms"] != [entry["arm"]]
            or gate["allowed_passes"] != [entry["pass"]]
            or type(gate["allowed_passes"][0]) is not int
            or not isinstance(gate["reason"], str) or not gate["reason"].strip()):
        raise ValueError("composite_no_go_scope")
    source_receipt = _json(Path(entry["source"]["path"]))
    if (type(source_receipt.get("model_calls")) is not int or source_receipt["model_calls"] != 0
            or type(source_receipt.get("submitted_jobs")) is not int or source_receipt["submitted_jobs"] != 0):
        raise ValueError("composite_unavailable_source_not_zero")
    launch_value = _json(launch)
    evidence = _json(_file(entry["unavailable"]))
    if (not isinstance(evidence, dict)
            or set(evidence) != {"schema", "arm", "pass", "reason", "source_commit",
                                 "launch_sha256", "candidates_sha256", "astra_no_go_sha256",
                                 "model_calls", "dispatch_claim", "ledger_audit_sha256"}
            or evidence["schema"] != "three-pass-selector-unavailable/v1"
            or evidence["arm"] != entry["arm"] or evidence["pass"] != entry["pass"]
            or evidence["reason"] != gate["reason"]
            or evidence["source_commit"] != launch_value.get("source_commit")
            or evidence["launch_sha256"] != entry["selector_launch"]["sha256"]
            or evidence["candidates_sha256"] != entry["candidates"]["sha256"]
            or evidence["astra_no_go_sha256"] != entry["astra_no_go"]["sha256"]
            or type(evidence["model_calls"]) is not int or evidence["model_calls"] != 0
            or evidence["dispatch_claim"] is not None):
        raise ValueError("composite_unavailable_binding")
    accounting = entry["accounting"]
    if not isinstance(accounting, dict) or set(accounting) != {"kind", "evidence"} or accounting["kind"] != "hosted":
        raise ValueError("composite_unavailable_accounting_shape")
    account = _json(_file(accounting["evidence"]))
    if (not isinstance(account, dict)
            or set(account) != {"schema", "arm", "pass", "kind", "ledger", "audit"}
            or account["schema"] != "three-pass-selector-no-run-accounting/v1"
            or account["arm"] != entry["arm"] or account["pass"] != entry["pass"]
            or account["kind"] != "hosted"):
        raise ValueError("composite_unavailable_accounting_binding")
    ledger = _file(account["ledger"])
    audit = _json(_file(account["audit"]))
    if (not isinstance(audit, dict)
            or set(audit) != {"schema", "arm", "pass", "ledger_sha256", "selector_calls", "reviewer"}
            or audit["schema"] != "three-pass-hosted-slot-audit/v1"
            or audit["arm"] != entry["arm"] or audit["pass"] != entry["pass"]
            or audit["ledger_sha256"] != account["ledger"]["sha256"]
            or type(audit["selector_calls"]) is not int or audit["selector_calls"] != 0
            or audit["reviewer"] != "gpt-6-astra"
            or evidence["ledger_audit_sha256"] != account["audit"]["sha256"]):
        raise ValueError("composite_unavailable_ledger_audit")
    try:
        connection = sqlite3.connect(f"{ledger.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            metadata = {key: json.loads(value) for key, value in
                        connection.execute("SELECT key,value FROM metadata")}
            rows = connection.execute("SELECT request_id FROM calls WHERE arm=? AND block='selector'",
                                      (entry["arm"],)).fetchall()
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise ValueError("composite_unavailable_ledger_unreadable") from exc
    if (metadata.get("schema") != "averitec-three-pass-hosted-budget/v1"
            or metadata.get("experiment_id") != "averitec-controller-three-pass-verdict-2026"):
        raise ValueError("composite_unavailable_ledger_identity")
    prefix = f"selector/{entry['arm']}/pass-{entry['pass']}/"
    if any(isinstance(row[0], str) and row[0].startswith(prefix) for row in rows):
        raise ValueError("composite_unavailable_ledger_has_calls")
    return {"reason": evidence["reason"], "unavailable_sha256": entry["unavailable"]["sha256"],
            "no_go_sha256": entry["astra_no_go"]["sha256"],
            "ledger_sha256": account["ledger"]["sha256"],
            "ledger_audit_sha256": account["audit"]["sha256"]}


def _accounting(entry: dict) -> dict:
    accounting = entry["accounting"]
    kind = "hosted" if entry["arm"] in {"jev", "gemini"} else "slurm"
    if not isinstance(accounting, dict) or set(accounting) != {"kind", "evidence"} or accounting["kind"] != kind:
        raise ValueError("composite_accounting_shape")
    evidence = _json(_file(accounting["evidence"]))
    if (not isinstance(evidence, dict) or evidence.get("schema") != "three-pass-selector-accounting/v1"
            or evidence.get("arm") != entry["arm"] or evidence.get("pass") != entry["pass"]
            or evidence.get("kind") != kind):
        raise ValueError("composite_accounting_binding")
    if kind == "hosted":
        if set(evidence) != {"schema", "arm", "pass", "kind", "ledger"}:
            raise ValueError("composite_hosted_accounting_shape")
        return {"ledger_sha256": _sha256(_file(evidence["ledger"]))}
    if set(evidence) != {"schema", "arm", "pass", "kind", "accepted", "sacct"}:
        raise ValueError("composite_slurm_accounting_shape")
    accepted = _json(_file(evidence["accepted"]))
    if (not isinstance(accepted, dict)
            or accepted.get("schema") != "averitec-three-pass-selector-submit-accepted/v1"
            or accepted.get("arm") != entry["arm"] or accepted.get("pass") != entry["pass"]
            or accepted.get("manifest_sha256") != entry["selector_launch"]["sha256"]
            or accepted.get("gate_sha256") != entry["selector_gate"]["sha256"]
            or not isinstance(accepted.get("job_id"), str) or not accepted["job_id"].isdigit()):
        raise ValueError("composite_submission_binding")
    sacct = _file(evidence["sacct"]).read_text(encoding="utf-8").strip().splitlines()
    if len(sacct) != 1:
        raise ValueError("composite_sacct_shape")
    parts = sacct[0].split("|")
    if (len(parts) != 4 or parts[0] != accepted["job_id"]
            or parts[1].split()[0] not in FINAL_SLURM_STATES
            or not parts[2].isdigit() or "gres/gpu=1" not in parts[3].split(",")):
        raise ValueError("composite_sacct_binding")
    return {"job_id": accepted["job_id"], "slurm_state": parts[1],
            "sacct_sha256": evidence["sacct"]["sha256"],
            "accepted_sha256": evidence["accepted"]["sha256"]}


def _measured(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("selector_result_row_invalid")
        if row.get("phase") == "measured":
            rows.append({"case_id": row.get("case_id"), "id": row.get("id"),
                         "outcome": row.get("outcome"), "action": row.get("action")})
    return rows


def assemble(manifest_file: Path, *, expected_cases: int = 100) -> dict:
    """Verify all 21 frozen sources and build the complete 27-arm matrix."""
    manifest = _json(manifest_file)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "entries"}
            or manifest["schema"] != MANIFEST_SCHEMA or not isinstance(manifest["entries"], list)
            or len(manifest["entries"]) != len(ARMS) * len(PASSES)):
        raise ValueError("composite_manifest_shape")
    planned = {(arm, number) for number in PASSES for arm in ARMS}
    entries = {}
    for entry in manifest["entries"]:
        if (not isinstance(entry, dict)
                or set(entry) not in (ENTRY_KEYS, UNAVAILABLE_ENTRY_KEYS)
                or not isinstance(entry["arm"], str) or type(entry["pass"]) is not int
                or (set(entry) == UNAVAILABLE_ENTRY_KEYS and entry["status"] != "unavailable")):
            raise ValueError("composite_entry_shape")
        key = (entry["arm"], entry["pass"])
        if key not in planned or key in entries:
            raise ValueError("composite_duplicate_or_unknown_slot")
        entries[key] = entry

    packages: list[dict] = []
    receipts: list[dict] = []
    candidate_hash: str | None = None
    source_document: dict | None = None
    for number in PASSES:
        for arm in ARMS:
            entry = entries[arm, number]
            source = _file(entry["source"])
            candidates = _file(entry["candidates"])
            launch = _file(entry["selector_launch"])
            if candidate_hash is None:
                candidate_hash = entry["candidates"]["sha256"]
                if expected_cases == 100 and candidate_hash != FROZEN_CANDIDATES_SHA256:
                    raise ValueError("composite_protocol_candidates_mismatch")
                source_document = _json(candidates)
            elif entry["candidates"]["sha256"] != candidate_hash:
                raise ValueError("composite_candidate_drift")
            if entry.get("status") == "unavailable":
                _source_receipt(source, entry)
                no_run = _unavailable(entry, launch, candidates)
                for case in downstream._validate_candidates(source_document, expected_cases, 10):
                    packages.append({"schema": "three-pass-selected-passages/v1",
                                     "case_id": case["case_id"], "group_id": case["group_id"],
                                     "claim": case["claim"], "mode": "selector", "status": "failed",
                                     "errors": [f"selector_unavailable:{no_run['reason']}"],
                                     "selected_passages": None, "arm": arm, "pass": number})
                receipts.append({"arm": arm, "pass": number, "status": "unavailable",
                                 "source_sha256": entry["source"]["sha256"],
                                 "candidates_sha256": candidate_hash,
                                 "selector_launch_sha256": entry["selector_launch"]["sha256"],
                                 "accounting_evidence_sha256": entry["accounting"]["evidence"]["sha256"],
                                 "model_calls": 0, "dispatch_claim": None,
                                 "complete": False, "measured_results": 0, **no_run})
                continue
            gate = _file(entry["selector_gate"])
            directory = _output(entry["selector_output"])
            _source_receipt(source, entry)
            accounting_details = _accounting(entry)
            receipt = selector_passes.verify(directory, candidates, launch, gate, frozen_source=True)
            metadata = _json(directory / "run-metadata.json")
            if metadata.get("qualification") != "real_runtime" or receipt.get("qualification") != "real_runtime":
                raise ValueError("test_selector_receipt_prohibited")
            if metadata.get("arm") != arm or metadata.get("pass") != number:
                raise ValueError("selector_receipt_layout_mismatch")
            measured = _measured(directory / "results.jsonl")
            for item in downstream.build_packages(source_document, measured, expected_cases=expected_cases):
                packages.append({**item, "arm": arm, "pass": number})
            receipts.append({"arm": arm, "pass": number,
                             "source_sha256": entry["source"]["sha256"],
                             "candidates_sha256": candidate_hash,
                             "selector_launch_sha256": entry["selector_launch"]["sha256"],
                             "selector_gate_sha256": entry["selector_gate"]["sha256"],
                             "accounting_kind": entry["accounting"]["kind"],
                             "accounting_evidence_sha256": entry["accounting"]["evidence"]["sha256"],
                             **accounting_details,
                             "receipt_sha256": entry["selector_output"]["files"]["receipt.json"],
                             "complete": receipt["complete"], "measured_results": len(measured),
                             "historical_native_usage_audit": receipt.get("historical_native_usage_audit")})
    assert source_document is not None and candidate_hash is not None
    for number in PASSES:
        for mode in ("include_all", "include_none"):
            for item in downstream.build_packages(source_document, mode=mode, expected_cases=expected_cases):
                packages.append({**item, "arm": mode, "pass": number})
    if len(packages) != 27 * expected_cases or len(receipts) != 21:
        raise ValueError("package_matrix_incomplete")
    runnable = sum(item["status"] == "ready" for item in packages)
    failure_reasons = Counter(error.split(":", 1)[0] for item in packages
                              if item["status"] == "failed" for error in item["errors"])
    return {"schema": PACKAGE_SCHEMA, "composite_manifest_sha256": _sha256(manifest_file),
            "candidates_sha256": candidate_hash, "planned_slots": len(packages),
            "runnable_packages": runnable, "failed_slots": len(packages) - runnable,
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "selector_receipts": receipts, "packages": packages}


def seal(output: Path, value: dict) -> str:
    """Write once; no overwrite or partial JSON after an interrupted run."""
    body = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(body).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = assemble(args.composite_manifest)
    digest = seal(args.output, value)
    print(json.dumps({"planned_slots": value["planned_slots"], "runnable_packages": value["runnable_packages"],
                      "failed_slots": value["failed_slots"], "failure_reasons": value["failure_reasons"],
                      "receipts": len(value["selector_receipts"]),
                      "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
