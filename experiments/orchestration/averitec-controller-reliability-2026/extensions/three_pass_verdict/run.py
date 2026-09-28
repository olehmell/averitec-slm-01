"""Read-only input gate for the three-pass extension; no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[5]
CONFIG = Path(__file__).with_name("config.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_file(root: Path, relative: str, expected_hash: str) -> Path:
    path = root / relative
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError(f"missing_or_changed_input:{relative}")
    return path


def check_inputs(input_root: Path, *, verify_evaluator_reference: bool = False,
                 verify_source_provenance: bool = False) -> dict:
    """Verify frozen identities and scope, without opening gold in the run path."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    if config["status"] != "planned" or config["protocol_version"] != "three-pass-verdict-v1":
        raise ValueError("unexpected_protocol")
    if not config["authorization"]["launch_blocked_until_exact_manifest_astra_review_and_numeric_caps"]:
        raise ValueError("launch_gate_must_remain_closed")

    audit_path = _checked_file(input_root, inputs["workflow_audit_path"], inputs["workflow_audit_sha256"])
    freeze_path = _checked_file(input_root, inputs["workflow_freeze_path"], inputs["workflow_freeze_file_sha256"])
    _checked_file(input_root, inputs["workflow_preparation_traces_path"], inputs["workflow_preparation_traces_sha256"])
    candidates_path = _checked_file(input_root, inputs["candidates_path"], inputs["candidates_sha256"])
    if verify_evaluator_reference:
        _checked_file(input_root, inputs["reference_evaluator_only_path"], inputs["reference_evaluator_only_sha256"])
        gold_path = _checked_file(input_root, inputs["verdict_gold_evaluator_only_path"], inputs["verdict_gold_evaluator_only_sha256"])
        verdict_rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        verdict_gold_by_id = {row["case_id"]: row for row in verdict_rows}
        if len(verdict_rows) != len(verdict_gold_by_id):
            raise ValueError("duplicate_verdict_gold_case_id")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not audit.get("complete") or audit.get("cases") != 100 or audit.get("trials") != 1500:
        raise ValueError("workflow_audit_incomplete")
    if audit.get("freeze_file_sha256") != inputs["workflow_freeze_file_sha256"]:
        raise ValueError("workflow_audit_freeze_mismatch")
    if audit.get("preparation_trace_sha256") != inputs["workflow_preparation_traces_sha256"]:
        raise ValueError("workflow_audit_traces_mismatch")
    original_arms = {arm["controller"]: arm for arm in audit["arms"]}
    if set(original_arms) != {"jev", "lfm", "qwen", "lfm26", "gemini"}:
        raise ValueError("workflow_original_arms")
    if any(arm["observed_decisions"] != 2299 or arm["trials"] != 300 for arm in original_arms.values()):
        raise ValueError("workflow_checkpoint_count")
    if freeze.get("preparation_trace_sha256") != inputs["workflow_preparation_traces_sha256"]:
        raise ValueError("freeze_traces_mismatch")
    if freeze["selection"].get("gold_used") is not False:
        raise ValueError("gold_in_workflow_selection")

    freeze_cases = freeze["cases"]
    candidate_cases = candidates["cases"]
    if len(freeze_cases) != inputs["cases"] or len(candidate_cases) != inputs["cases"]:
        raise ValueError("case_count")
    freeze_by_id = {row["case_id"]: row for row in freeze_cases}
    candidates_by_id = {row["case_id"]: row for row in candidate_cases}
    if len(freeze_by_id) != 100 or set(freeze_by_id) != set(candidates_by_id):
        raise ValueError("case_id_mismatch")
    if verify_evaluator_reference and not set(freeze_by_id) <= set(verdict_gold_by_id):
        raise ValueError("verdict_gold_missing_cases")
    if len({row["group_id"] for row in freeze_cases}) != 100:
        raise ValueError("duplicate_source_group")
    if candidates.get("gold_included") is not False:
        raise ValueError("gold_in_candidates")
    if candidates.get("source_freeze_sha256") != inputs["workflow_freeze_file_sha256"]:
        raise ValueError("candidate_freeze_mismatch")
    if candidates.get("source_traces_sha256") != inputs["workflow_preparation_traces_sha256"]:
        raise ValueError("candidate_traces_mismatch")
    candidate_count = 0
    for case_id, case in candidates_by_id.items():
        if case["group_id"] != freeze_by_id[case_id]["group_id"]:
            raise ValueError("candidate_group_mismatch")
        rows = case["candidates"]
        if len(rows) != inputs["passages_per_case"] or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("candidate_ids")
        if any(not row["text"] or not row["url"] or not row["passage_id"] for row in rows):
            raise ValueError("empty_candidate_identity")
        candidate_count += len(rows)
    if candidate_count != 1000:
        raise ValueError("candidate_count")
    if verify_source_provenance:
        from source_rejoin import rejoin_candidates
        joined = rejoin_candidates(
            Path(inputs["candidates_path"]), input_root=input_root,
            expected_candidate_sha256=inputs["candidates_sha256"],
            source_manifest_path=Path(inputs["source_manifest_path"]),
            expected_manifest_sha256=inputs["source_manifest_sha256"],
            expected_archive_sha256=inputs["dev_source_archive_sha256"],
        )
        if len(joined["cases"]) != 100 or sum(len(case["passages"]) for case in joined["cases"]) != 1000:
            raise ValueError("source_rejoin_count")

    workflow = config["workflow"]
    selection = config["selection"]
    if (len(workflow["arms"]) * workflow["repetitions"] * inputs["cases"] * len(workflow["scenarios"])
            != workflow["planned_trials_per_mode"]):
        raise ValueError("workflow_trial_arithmetic")
    if (len(workflow["arms"]) * workflow["repetitions"] * workflow["expected_checkpoints_per_arm_pass"]
            != workflow["maximum_checkpoint_decisions"]):
        raise ValueError("workflow_decision_arithmetic")
    if (len(selection["arms"]) * selection["repetitions"] * candidate_count
            != selection["maximum_measured_decisions"]):
        raise ValueError("selection_decision_arithmetic")
    if (len(selection["arms"]) * selection["repetitions"] * inputs["cases"]
            != config["downstream"]["maximum_selector_verdict_calls"]):
        raise ValueError("verdict_call_arithmetic")
    return {
        "schema": "three-pass-verdict-input-check/v1",
        "status": "inputs_verified_no_model_calls",
        "cases": len(freeze_cases),
        "candidate_pairs": candidate_count,
        "planned_workflow_trials_per_mode": workflow["planned_trials_per_mode"],
        "planned_selector_decisions": selection["maximum_measured_decisions"],
        "evaluator_reference_hash_checked": verify_evaluator_reference,
        "source_provenance_checked": verify_source_provenance,
        "launch_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--input-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--verify-evaluator-reference", action="store_true")
    parser.add_argument("--verify-source-provenance", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_inputs(args.input_root,
                                  verify_evaluator_reference=args.verify_evaluator_reference,
                                  verify_source_provenance=args.verify_source_provenance), sort_keys=True))


if __name__ == "__main__":
    main()
