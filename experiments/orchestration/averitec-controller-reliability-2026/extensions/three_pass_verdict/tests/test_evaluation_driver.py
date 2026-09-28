"""Synthetic-only checks of the final denominator and inference seal."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation_driver as driver


def _file(tmp_path: Path, name: str, value, *, lines: bool = False) -> dict:
    path = tmp_path / name
    body = ("".join(json.dumps(row, sort_keys=True) + "\n" for row in value) if lines
            else json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(body)
    return {"path": str(path), "sha256": hashlib.sha256(body).hexdigest()}


def _fixture(tmp_path: Path) -> dict:
    cases = [("c1", "g1"), ("c2", "g2")]
    trials = []
    checkpoints = []
    for case, group in cases:
        for scenario in driver.SCENARIOS:
            trial_id = f"base-{case}-{scenario}"
            trials.append({"trial_id": trial_id, "case_id": case, "group_id": group, "scenario": scenario})
            checkpoints.append({"checkpoint_id": trial_id + "-0", "trial_id": trial_id,
                                "step": 0, "expected_action": "decompose"})
    plan = _file(tmp_path, "base-plan.json", {"trials": trials, "checkpoints": checkpoints})
    candidates = {"gold_included": False, "cases": [
        {"case_id": case, "group_id": group, "candidates": [{"id": f"{case}-p{index}"} for index in range(1, 11)]}
        for case, group in cases]}
    candidate_ref = _file(tmp_path, "candidates.json", candidates)
    packages = [{"arm": arm, "pass": number, "case_id": case, "group_id": group, "status": "ready"}
                for arm in (*driver.ARMS, *driver.evaluate.CONTROLS) for number in driver.PASSES
                for case, group in cases]
    packages_ref = _file(tmp_path, "packages.json", {"schema": "three-pass-verdict-packages/v1",
                                                    "composite_manifest_sha256": "same", "packages": packages})
    choices = {"schema": "three-pass-sealed-selection-choices/v1", "sealed": True,
               "composite_manifest_sha256": "same",
               "source_receipts": [{"arm": arm, "pass": number, "source_sha256": "a" * 64,
                                    "accounting_evidence_sha256": "b" * 64,
                                    "selector_launch_sha256": "c" * 64, "receipt_sha256": "d" * 64}
                                   for arm in driver.ARMS for number in driver.PASSES],
               "rows": [{"arm": arm, "pass": number, "case_id": case, "id": f"{case}-p{index}",
                         "outcome": "ok", "action": "include" if index == 1 else "exclude"}
                        for arm in driver.ARMS for number in driver.PASSES for case, _ in cases for index in range(1, 11)]}
    choices_ref = _file(tmp_path, "choices.json", choices)
    verdict_rows = [{"phase": "measured", "arm": row["arm"], "pass": row["pass"],
                     "case_id": row["case_id"], "outcome": "ok", "verdict": "Supported"}
                    for row in packages]
    verdict_ref = _file(tmp_path, "verdict.jsonl", verdict_rows, lines=True)
    verdict_receipt = _file(tmp_path, "verdict-receipt.json", {"schema": "three-pass-verdict-run/v1",
                                                             "results_sha256": verdict_ref["sha256"],
                                                             "measured_results": len(verdict_rows)})
    slots = []
    for arm in driver.ARMS:
        for number in driver.PASSES:
            if arm in {"jeff", "laya_typed"}:
                own_trials = [{**row, "trial_id": f"{arm}-{number}-" + row["trial_id"]} for row in trials]
                own_points = [{**row, "trial_id": f"{arm}-{number}-" + row["trial_id"],
                               "checkpoint_id": f"{arm}-{number}-" + row["checkpoint_id"]} for row in checkpoints]
                own_plan = _file(tmp_path, f"{arm}-{number}-plan.json",
                                 {"trials": own_trials, "checkpoints": own_points})
                ledger = [{"kind": "response", "checkpoint_id": point["checkpoint_id"],
                           "trial_id": point["trial_id"], "expected_action": "decompose",
                           "action": "decompose", "provider_outcome": "ok", "compliant": True}
                          for point in own_points]
                file = _file(tmp_path, f"{arm}-{number}-ledger.jsonl", ledger, lines=True)
                receipt = _file(tmp_path, f"{arm}-{number}-receipt.json",
                                {"schema": "averitec-native-workflow-pass-receipt/v1", "arm": arm,
                                 "pass": number, "ledger_sha256": file["sha256"]})
                slots.append({"arm": arm, "pass": number, "kind": "native_ledger", "file": file,
                              "receipt": receipt, "plan": own_plan})
            else:
                rows = [{"case_id": case, "group_id": group, "scenario": scenario,
                         "events": [{"step": 0, "expected_action": "decompose", "action": "decompose",
                                     "provider_outcome": "ok", "compliant": True}]}
                        for case, group in cases for scenario in driver.SCENARIOS]
                if arm == "jev" and number == 3:
                    rows = rows[:1]  # terminal partial: five planned trials remain missing
                file = _file(tmp_path, f"{arm}-{number}-results.jsonl", rows, lines=True)
                guard = _file(tmp_path, f"{arm}-{number}-guard.jsonl", [], lines=True)
                run = _file(tmp_path, f"{arm}-{number}-run.json",
                            {"schema": "averitec-controller-run/v1", "planned_tasks": trials})
                receipt = _file(tmp_path, f"{arm}-{number}-receipt.json",
                                {"schema": "averitec-original-workflow-repeat-receipt/v1", "arm": arm,
                                 "pass": number, "ledger_sha256": guard["sha256"]})
                slots.append({"arm": arm, "pass": number, "kind": "trial_results", "file": file,
                              "receipt": receipt, "run": run, "guard_ledger": guard})
    return {"schema": "three-pass-evaluation-inputs/v1", "inference_sealed": True,
            "packages": packages_ref, "candidates": candidate_ref, "sealed_choices": choices_ref,
            "verdict_results": verdict_ref, "verdict_receipt": verdict_receipt,
            "workflow_plan": plan, "workflow_slots": slots, "_sha256": "synthetic"}


def test_partial_pass_stays_in_planned_denominator(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    reference = {"reference_type": "synthetic", "ratings": [
        {"case_id": case, "id": f"{case}-p{index}", "grade": 2 if index == 1 else 0}
        for case in ("c1", "c2") for index in range(1, 11)]}
    gold = {"cases": [{"case_id": "c1", "group_id": "g1", "label": "Supported"},
                      {"case_id": "c2", "group_id": "g2", "label": "Refuted"}]}
    report = driver.evaluate_all(manifest, reference, gold, expected_cases=2, bootstrap_replicates=10)
    partial = report["workflow"]["by_arm_pass"]["jev"][3]
    assert partial["planned_checkpoints"] == 6
    assert partial["observed_checkpoints"] == 1
    assert partial["outcomes"] == {"compliant": 1, "missing": 5}
    assert report["verdict"]["by_arm_pass"]["jev"][3]["accuracy_over_planned"] == 0.5
    assert report["selection"]["by_arm_pass"]["jev"][3]["known_grades"] == 20
    graded = report["selection"]["by_arm_pass"]["jev"][3]["graded_gain_sensitivity"][2]
    assert graded["grade_one_gain"] == 0.5
    assert graded["selected_known_denominator"] == 2
    assert graded["available_gain_denominator"] == 2
    assert graded["graded_f1"] == 1.0
    assert report["workflow"]["group_bootstrap"]["unit"] == "source_group_with_all_scenarios_and_passes"
    assert report["verdict"]["group_bootstrap"]["groups"] == 2
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=False, allow_nan=False)
    parsed = json.loads(encoded)
    assert set(parsed["selection"]["by_arm_pass"]["jev"]["3"]["by_grade"]) == {"0", "1", "2", "U"}


def test_missing_or_changed_inference_receipt_blocks_gold(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["workflow_slots"][0]["receipt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input_hash_drift"):
        driver.validate_inference(manifest, expected_cases=2)
    manifest = _fixture(tmp_path)
    manifest["sealed_choices"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input_hash_drift"):
        driver.validate_inference(manifest, expected_cases=2)


def test_unsealed_and_wrong_matrix_are_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["inference_sealed"] = False
    with pytest.raises(ValueError, match="inference_not_sealed"):
        driver.validate_inference(manifest, expected_cases=2)
    manifest["inference_sealed"] = True
    manifest["workflow_slots"].pop()
    with pytest.raises(ValueError, match="workflow_slot_count"):
        driver.validate_inference(manifest, expected_cases=2)


def test_verdict_gold_jsonl_requires_exact_selected_join(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(''.join(json.dumps(row) + '\n' for row in [
        {"case_id": "c1", "label": "Supported"}, {"case_id": "unselected", "label": "Refuted"},
        {"case_id": "c2", "label": "Refuted"}]), encoding="utf-8")
    selected = driver.load_pinned_verdict_gold(path, [{"case_id": "c1", "group_id": "g1"},
                                                    {"case_id": "c2", "group_id": "g2"}], enforce_pin=False)
    assert selected == {"cases": [{"case_id": "c1", "group_id": "g1", "label": "Supported"},
                                  {"case_id": "c2", "group_id": "g2", "label": "Refuted"}]}
    with pytest.raises(ValueError, match="verdict_gold_pin_mismatch"):
        driver.load_pinned_verdict_gold(path, [{"case_id": "c1", "group_id": "g1"}])


def test_terminal_partial_verdict_rows_keep_planned_denominator(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    rows = [json.loads(line) for line in Path(manifest["verdict_results"]["path"]).read_text().splitlines()]
    rows = rows[:-3]
    verdict_ref = _file(tmp_path, "verdict-partial.jsonl", rows, lines=True)
    evidence = _file(tmp_path, "terminal-evidence.json", {"scheduler": "TIMEOUT"})
    receipt = _file(tmp_path, "verdict-partial-receipt.json", {"schema": "three-pass-verdict-terminal-seal/v1",
        "results_sha256": verdict_ref["sha256"], "packages_sha256": manifest["packages"]["sha256"],
        "outcome": "time_limit", "terminal_evidence_sha256": evidence["sha256"], "complete": False})
    manifest["verdict_results"] = verdict_ref
    manifest["verdict_receipt"] = receipt
    manifest["verdict_terminal_evidence"] = evidence
    *_, verdicts, _, _ = driver.validate_inference(manifest, expected_cases=2)
    assert len(verdicts["rows"]) == 54
    assert sum(row["outcome"] == "missing_after_terminal_stop" for row in verdicts["rows"]) == 3


def test_retry_abort_and_finish_are_distinct_decision_counts() -> None:
    actions = {"nominal": ["decompose", "finish"],
               "retrieval_timeout_once": ["retrieve", "retrieve", "finish"],
               "retrieval_timeout_persistent": ["retrieve", "retrieve", "abort"]}
    trials = {("c1", scenario): [{"step": step, "expected_action": action}
                                  for step, action in enumerate(sequence)]
              for scenario, sequence in actions.items()}
    observed = {(case, scenario, point["step"]): {"action": point["expected_action"],
                "expected_action": point["expected_action"], "provider_outcome": "ok", "compliant": True}
                for (case, scenario), points in trials.items() for point in points}
    slots = {(arm, number): observed for arm in driver.ARMS for number in driver.PASSES}
    targets = driver._workflow_scores(slots, trials)["by_arm_pass"]["jev"][1]["expected_action_counts"]
    assert targets == {"initial_retrieve": {"planned": 2, "correct": 2},
                       "retry_retrieve": {"planned": 2, "correct": 2},
                       "abort": {"planned": 1, "correct": 1},
                       "finish": {"planned": 2, "correct": 2}}


def test_zero_gain_with_defined_precision_and_recall_has_zero_f1(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    choices = json.loads(Path(manifest["sealed_choices"]["path"]).read_text())
    for row in choices["rows"]:
        row["action"] = "exclude" if row["id"].endswith("p1") else "include"
    manifest["sealed_choices"] = _file(tmp_path, "choices-zero-gain.json", choices)
    reference = {"reference_type": "synthetic", "ratings": [
        {"case_id": case, "id": f"{case}-p{index}", "grade": 2 if index == 1 else 0}
        for case in ("c1", "c2") for index in range(1, 11)]}
    gold = {"cases": [{"case_id": "c1", "group_id": "g1", "label": "Supported"},
                      {"case_id": "c2", "group_id": "g2", "label": "Refuted"}]}
    score = driver.evaluate_all(manifest, reference, gold, expected_cases=2)["selection"]["by_arm_pass"]["jev"][1]
    assert score["graded_gain_sensitivity"][2]["graded_f1"] == 0.0
