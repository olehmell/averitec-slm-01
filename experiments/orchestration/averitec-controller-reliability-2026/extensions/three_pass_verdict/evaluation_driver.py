"""Offline three-pass evaluation. Run only after all inference is sealed.

The manifest binds every inference input by SHA-256. ``--check-inference`` reads
no evaluator reference. The scoring command validates all inference inputs
before it opens either evaluator-only reference.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import evaluate
import evaluate_selection

ARMS = evaluate_selection.ARMS
PASSES = (1, 2, 3)
SCENARIOS = ("nominal", "retrieval_timeout_once", "retrieval_timeout_persistent")
PINNED_VERDICT_GOLD_SHA256 = "e7724886f9a7885dfa6c38710aef644b79b928692a3b89238879d3095b2ea71b"
PINNED_CANDIDATES_SHA256 = "219c745aac6f50501a55c4f2511d383505317fd75338b0efaed25f44bbe23b98"


def _bound(ref: dict, *, binary: bool = False) -> Any:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("input_binding_shape")
    path = Path(ref["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("input_binding_path")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("input_hash_drift")
    return data if binary else json.loads(data)


def _lines(ref: dict) -> list[dict]:
    rows = []
    for line in _bound(ref, binary=True).splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("jsonl_row_shape")
            rows.append(value)
    return rows


def _plan(native_plan: dict, cases: list[dict], expected_cases: int) -> tuple[dict, dict]:
    if not isinstance(native_plan, dict) or not isinstance(native_plan.get("trials"), list) or not isinstance(native_plan.get("checkpoints"), list):
        raise ValueError("workflow_plan_shape")
    roster = {row["case_id"]: row["group_id"] for row in cases}
    if len(roster) != expected_cases:
        raise ValueError("workflow_case_roster")
    trials = {}
    trial_ids = {}
    for trial in native_plan["trials"]:
        key = (trial.get("case_id"), trial.get("scenario"))
        if (key[0] not in roster or trial.get("group_id") != roster[key[0]]
                or key[1] not in SCENARIOS or key in trials):
            raise ValueError("workflow_trial_plan")
        trials[key] = []
        trial_ids[trial["trial_id"]] = key
    if len(trials) != expected_cases * len(SCENARIOS):
        raise ValueError("workflow_trial_count")
    checkpoint_ids = {}
    for checkpoint in native_plan["checkpoints"]:
        key = trial_ids.get(checkpoint.get("trial_id"))
        if key is None or type(checkpoint.get("step")) is not int or checkpoint["step"] < 0:
            raise ValueError("workflow_checkpoint_plan")
        if not isinstance(checkpoint.get("expected_action"), str):
            raise ValueError("workflow_expected_action")
        trials[key].append(checkpoint)
        checkpoint_ids[checkpoint["checkpoint_id"]] = (key, checkpoint)
    if len(checkpoint_ids) != len(native_plan["checkpoints"]):
        raise ValueError("workflow_checkpoint_duplicate")
    for key, points in trials.items():
        points.sort(key=lambda row: row["step"])
        if [point["step"] for point in points] != list(range(len(points))):
            raise ValueError(f"workflow_step_gap:{key}")
    if expected_cases == 100 and len(checkpoint_ids) != 2299:
        raise ValueError("workflow_checkpoint_count")
    return trials, checkpoint_ids


def _workflow_slot(slot: dict, trials: dict, roster: dict) -> dict:
    required = {"arm", "pass", "kind", "file", "receipt"}
    extras = ({"plan"} if slot.get("kind") == "native_ledger" else
              {"run", "guard_ledger"} if slot.get("kind") == "trial_results" else set())
    if set(slot) != required | extras:
        raise ValueError("workflow_slot_shape")
    receipt = _bound(slot["receipt"])
    if not isinstance(receipt, dict):
        raise ValueError("workflow_receipt_shape")
    rows = _lines(slot["file"])
    observed = {}
    if slot["kind"] == "native_ledger":
        own_plan = _bound(slot["plan"])
        own_trials, checkpoint_ids = _plan(own_plan,
                                           [{"case_id": case, "group_id": group} for case, group in roster.items()],
                                           len(roster))
        if {(case, scenario): [(point["step"], point["expected_action"]) for point in points]
                for (case, scenario), points in own_trials.items()} != {
                key: [(point["step"], point["expected_action"]) for point in points]
                for key, points in trials.items()}:
            raise ValueError("native_workflow_plan_drift")
        if (receipt.get("schema") != "averitec-native-workflow-pass-receipt/v1"
                or receipt.get("arm") != slot["arm"] or receipt.get("pass") != slot["pass"]
                or receipt.get("ledger_sha256") != slot["file"]["sha256"]
                or receipt.get("plan_sha256") != own_plan.get("plan_sha256")):
            raise ValueError("native_workflow_receipt_binding")
        for row in rows:
            if row.get("kind") not in {"response", "uncertain"}:
                continue
            binding = checkpoint_ids.get(row.get("checkpoint_id"))
            if binding is None:
                raise ValueError("workflow_native_unknown_checkpoint")
            key, point = binding
            if key + (point["step"],) in observed or row.get("trial_id") != point["trial_id"]:
                raise ValueError("workflow_native_duplicate_or_trial")
            if row.get("expected_action") != point["expected_action"]:
                raise ValueError("workflow_native_expected_action")
            observed[key + (point["step"],)] = row
    elif slot["kind"] == "trial_results":
        run = _bound(slot["run"])
        _lines(slot["guard_ledger"])
        if (receipt.get("schema") != "averitec-original-workflow-repeat-receipt/v1"
                or receipt.get("arm") != slot["arm"] or receipt.get("pass") != slot["pass"]
                or receipt.get("ledger_sha256") != slot["guard_ledger"]["sha256"]
                or run.get("schema") != "averitec-controller-run/v1"):
            raise ValueError("trial_workflow_receipt_binding")
        planned = {(row.get("case_id"), row.get("scenario")) for row in run.get("planned_tasks", [])}
        if planned != set(trials) or len(run.get("planned_tasks", [])) != len(trials):
            raise ValueError("trial_workflow_run_plan")
        seen = set()
        for row in rows:
            key = (row.get("case_id"), row.get("scenario"))
            if key not in trials or key in seen or row.get("group_id") != roster[key[0]]:
                raise ValueError("workflow_trial_unknown_or_duplicate")
            seen.add(key)
            points = trials[key]
            if not isinstance(row.get("events"), list) or len(row["events"]) > len(points):
                raise ValueError("workflow_trial_events")
            for event in row["events"]:
                step = event.get("step")
                if type(step) is not int or step < 0 or step >= len(points):
                    raise ValueError("workflow_event_step")
                if event.get("expected_action") != points[step]["expected_action"]:
                    raise ValueError("workflow_event_expected_action")
                identity = key + (step,)
                if identity in observed:
                    raise ValueError("workflow_event_duplicate")
                observed[identity] = event
    elif slot["kind"] != "unavailable":
        raise ValueError("workflow_slot_kind")
    if slot["kind"] == "unavailable":
        if rows or receipt.get("arm") != slot["arm"] or receipt.get("pass") != slot["pass"] or receipt.get("status") != "unavailable":
            raise ValueError("workflow_unavailable_receipt")
    expected = sum(map(len, trials.values()))
    if len(observed) > expected:
        raise ValueError("workflow_exceeds_plan")
    return observed


def _workflow_scores(by_slot: dict, trials: dict) -> dict:
    planned = sum(map(len, trials.values()))
    scores = {}
    for arm in ARMS:
        scores[arm] = {}
        for number in PASSES:
            observed = by_slot[arm, number]
            by_scenario = {}
            all_outcomes = Counter()
            compliant = 0
            action_targets = {name: {"planned": 0, "correct": 0}
                              for name in ("initial_retrieve", "retry_retrieve", "abort", "finish")}
            for scenario in SCENARIOS:
                keys = [(case_id, scen, point["step"])
                        for (case_id, scen), points in trials.items() if scen == scenario for point in points]
                status = Counter()
                for key in keys:
                    row = observed.get(key)
                    outcome = ("missing" if row is None else "compliant" if row.get("compliant") is True
                               and row.get("action") == row.get("expected_action") and row.get("provider_outcome") == "ok"
                               else "noncompliant")
                    status[outcome] += 1
                    all_outcomes[outcome] += 1
                    point = trials[key[:2]][key[2]]
                    expected_action = point["expected_action"]
                    target = ("retry_retrieve" if expected_action == "retrieve" and key[2] > 0
                              and trials[key[:2]][key[2] - 1]["expected_action"] == "retrieve" else
                              "initial_retrieve" if expected_action == "retrieve" else expected_action)
                    if target in action_targets:
                        action_targets[target]["planned"] += 1
                        action_targets[target]["correct"] += outcome == "compliant"
                compliant += status["compliant"]
                by_scenario[scenario] = {"planned": len(keys), "outcomes": dict(status),
                                         "compliance_over_planned": status["compliant"] / len(keys)}
            scores[arm][number] = {"planned_checkpoints": planned, "observed_checkpoints": len(observed),
                                   "outcomes": dict(all_outcomes), "first_attempt_oracle_compliance": compliant / planned,
                                   "by_scenario": by_scenario, "expected_action_counts": action_targets}
    repeat = {}
    for arm in ARMS:
        pairs = {}
        for a, b in combinations(PASSES, 2):
            same = both_correct = 0
            for key in [(case, scenario, point["step"]) for (case, scenario), points in trials.items() for point in points]:
                x, y = by_slot[arm, a].get(key), by_slot[arm, b].get(key)
                if (x is not None and y is not None and x.get("provider_outcome") == y.get("provider_outcome") == "ok"
                        and x.get("action") is not None and x.get("action") == y.get("action")):
                    same += 1
                    both_correct += x.get("compliant") is True and y.get("compliant") is True
            pairs[f"{a}-{b}"] = {"action_agreement_over_planned": same / planned,
                                  "both_correct_over_planned": both_correct / planned}
        all_same = all_correct = 0
        for key in [(case, scenario, point["step"]) for (case, scenario), points in trials.items() for point in points]:
            values = [by_slot[arm, number].get(key) for number in PASSES]
            if all(value is not None and value.get("provider_outcome") == "ok" and value.get("action") is not None
                   for value in values) and len({value["action"] for value in values}) == 1:
                all_same += 1
                all_correct += all(value.get("compliant") is True for value in values)
        repeat[arm] = {"planned_checkpoints": planned, "pairwise": pairs,
                       "all_three_action_agreement_over_planned": all_same / planned,
                       "all_three_correct_over_planned": all_correct / planned}
    return {"by_arm_pass": scores, "repeatability": repeat}


def validate_inference(manifest: dict, *, expected_cases: int = 100) -> tuple[dict, dict, dict, dict, dict]:
    if manifest.get("schema") != "three-pass-evaluation-inputs/v1" or manifest.get("inference_sealed") is not True:
        raise ValueError("inference_not_sealed")
    package_file = _bound(manifest["packages"])
    if package_file.get("schema") != "three-pass-verdict-packages/v1" or len(package_file.get("packages", [])) != 27 * expected_cases:
        raise ValueError("package_matrix_shape")
    packages = package_file["packages"]
    status_counts = Counter(row.get("status") for row in packages)
    if expected_cases == 100 and status_counts != {"ready": 2609, "failed": 91}:
        raise ValueError("package_status_count_drift")
    planned_cases = [{"case_id": row["case_id"], "group_id": row["group_id"]} for row in packages
                     if row["arm"] == "include_all" and row["pass"] == 1]
    if len(planned_cases) != expected_cases or len({row["case_id"] for row in planned_cases}) != expected_cases:
        raise ValueError("package_case_roster")
    choices = _bound(manifest["sealed_choices"])
    if choices.get("sealed") is not True or choices.get("schema") != "three-pass-sealed-selection-choices/v1":
        raise ValueError("selector_choices_unsealed")
    selector_receipts = choices.get("source_receipts")
    if (not isinstance(selector_receipts, list) or len(selector_receipts) != 21
            or {(row.get("arm"), row.get("pass")) for row in selector_receipts}
               != {(arm, number) for arm in ARMS for number in PASSES}):
        raise ValueError("selector_receipt_matrix")
    for source in selector_receipts:
        if (not isinstance(source.get("source_sha256"), str)
                or not isinstance(source.get("accounting_evidence_sha256"), str)
                or not isinstance(source.get("selector_launch_sha256"), str)
                or not (isinstance(source.get("receipt_sha256"), str)
                        or isinstance(source.get("unavailable_sha256"), str))):
            raise ValueError("selector_receipt_unbound")
    if choices.get("composite_manifest_sha256") != package_file.get("composite_manifest_sha256"):
        raise ValueError("package_selector_seal_mismatch")
    candidates = _bound(manifest["candidates"])
    if expected_cases == 100 and manifest["candidates"]["sha256"] != PINNED_CANDIDATES_SHA256:
        raise ValueError("candidate_protocol_pin_mismatch")
    if package_file.get("candidates_sha256") != manifest["candidates"]["sha256"] and expected_cases == 100:
        raise ValueError("package_candidate_hash_mismatch")
    if expected_cases == 100 and any(source.get("candidates_sha256") != manifest["candidates"]["sha256"]
                                     for source in selector_receipts):
        raise ValueError("selector_candidate_hash_mismatch")
    if candidates.get("gold_included") is not False or {row["case_id"] for row in candidates.get("cases", [])} != {row["case_id"] for row in planned_cases}:
        raise ValueError("candidate_roster")
    evaluate_selection._choices(choices, evaluate_selection._pool(candidates, expected_cases, 10)[0], ARMS, PASSES)
    verdict_rows = _lines(manifest["verdict_results"])
    verdict_receipt = _bound(manifest["verdict_receipt"])
    if (verdict_receipt.get("schema") not in {"three-pass-verdict-run/v1", "three-pass-verdict-terminal-seal/v1"}
            or verdict_receipt.get("results_sha256") != manifest["verdict_results"]["sha256"]):
        raise ValueError("verdict_receipt_binding")
    if verdict_receipt["schema"] == "three-pass-verdict-terminal-seal/v1":
        if (verdict_receipt.get("packages_sha256") != manifest["packages"]["sha256"]
                or verdict_receipt.get("outcome") not in {"partial", "time_limit", "failed"}
                or verdict_receipt.get("terminal_evidence_sha256") !=
                    manifest.get("verdict_terminal_evidence", {}).get("sha256")):
            raise ValueError("verdict_terminal_seal_binding")
        _bound(manifest["verdict_terminal_evidence"], binary=True)
    measured = [row for row in verdict_rows if row.get("phase") == "measured"]
    if len(measured) > 27 * expected_cases or len({(row.get("arm"), row.get("pass"), row.get("case_id")) for row in measured}) != len(measured):
        raise ValueError("verdict_result_matrix")
    package_map = {(row["arm"], row["pass"], row["case_id"]): row for row in packages}
    if len(package_map) != 27 * expected_cases or not {(row["arm"], row["pass"], row["case_id"]) for row in measured} <= set(package_map):
        raise ValueError("verdict_package_join")
    if any(package_map[row["arm"], row["pass"], row["case_id"]]["status"] == "failed" and row.get("outcome") != "upstream_package_failure" for row in measured):
        raise ValueError("failed_package_was_inferred")
    if len(measured) < 27 * expected_cases and verdict_receipt.get("complete") is True:
        raise ValueError("complete_verdict_receipt_has_missing_rows")
    if verdict_receipt["schema"] == "three-pass-verdict-run/v1" and verdict_receipt.get("measured_results") != len(measured):
        raise ValueError("verdict_receipt_result_count")
    observed_keys = {(row["arm"], row["pass"], row["case_id"]) for row in measured}
    measured.extend({"arm": arm, "pass": number, "case_id": case,
                     "outcome": "upstream_package_failure" if package["status"] == "failed" else "missing_after_terminal_stop",
                     "verdict": None}
                    for (arm, number, case), package in package_map.items() if (arm, number, case) not in observed_keys)
    native_plan = _bound(manifest["workflow_plan"])
    trials, checkpoint_ids = _plan(native_plan, planned_cases, expected_cases)
    roster = {row["case_id"]: row["group_id"] for row in planned_cases}
    slots = manifest.get("workflow_slots")
    if not isinstance(slots, list) or len(slots) != len(ARMS) * len(PASSES):
        raise ValueError("workflow_slot_count")
    by_slot = {}
    for slot in slots:
        key = (slot.get("arm"), slot.get("pass"))
        if key not in {(arm, number) for arm in ARMS for number in PASSES} or key in by_slot:
            raise ValueError("workflow_slot_duplicate_or_unknown")
        by_slot[key] = _workflow_slot(slot, trials, roster)
    if len(by_slot) != len(ARMS) * len(PASSES):
        raise ValueError("workflow_slot_missing")
    return choices, candidates, {"sealed": True, "rows": measured}, {"planned_cases": planned_cases, "trials": trials, "by_slot": by_slot}, verdict_receipt


def evaluate_all(manifest: dict, selection_reference: dict, verdict_gold: dict, *, expected_cases: int = 100,
                 expected_grade_counts: dict | None = None, bootstrap_replicates: int = 0) -> dict:
    choices, candidates, verdicts, workflow, receipt = validate_inference(manifest, expected_cases=expected_cases)
    selected = evaluate_selection.evaluate_selection(choices, candidates, selection_reference,
                                                       expected_cases=expected_cases,
                                                       expected_grade_counts=expected_grade_counts,
                                                       bootstrap_replicates=bootstrap_replicates)
    for arm in ARMS:
        for number in PASSES:
            for item in selected["by_arm_pass"][arm][number]["graded_gain_sensitivity"]:
                precision, recall = item["graded_precision"], item["graded_recall"]
                item["graded_f1"] = (2 * precision * recall / (precision + recall) if precision + recall else 0.0
                                     ) if precision is not None and recall is not None else None
                item["selected_known_denominator"] = item["selected_known"]
                item["available_gain_denominator"] = item["available_gain"]
    selected["graded_formula"] = {
        "gain": "g(0)=0, g(1)=w, g(2)=1; w in {0, 0.25, 0.5, 0.75, 1}",
        "precision": "sum(g(grade) for selected known) / number of selected known",
        "recall": "sum(g(grade) for selected known) / sum(g(grade) for all known)",
        "f1": "2 * precision * recall / (precision + recall)",
        "unknown_grade": "U excluded from gains and denominators",
        "invalid_or_missing_choice": "not converted to exclude or selected"}
    verdict = evaluate.evaluate_verdicts(verdicts, workflow["planned_cases"], verdict_gold,
                                         expected_cases=expected_cases)
    selected["correctness_repeatability"] = _selection_correctness_repeatability(choices, candidates, selection_reference,
                                                                                   expected_cases)
    verdict["repeatability"] = _verdict_repeatability(verdicts, verdict_gold, workflow["planned_cases"])
    workflow_result = _workflow_scores(workflow["by_slot"], workflow["trials"])
    if bootstrap_replicates:
        workflow_result["group_bootstrap"] = _workflow_bootstrap(workflow, bootstrap_replicates)
        verdict["group_bootstrap"] = _verdict_bootstrap(verdicts, verdict_gold, workflow["planned_cases"],
                                                        bootstrap_replicates)
    return {"schema": "three-pass-evaluation-report/v1", "inference_manifest_sha256": manifest["_sha256"],
            "verdict_receipt": receipt, "workflow": workflow_result,
            "selection": selected, "verdict": verdict,
            "limits": ["Fixed candidate pool; no full-corpus retrieval claim.",
                       "Repeated passes share 100 claims, not 300 independent cases.",
                       "Checkpoint choices do not establish end-to-end trajectory completion."]}


def _group_bootstrap_rates(buckets: dict, arms: tuple[str, ...], replicates: int) -> dict:
    """Resample whole source groups with every passage, scenario and pass."""
    groups = sorted({group for arm in arms for number in PASSES for group in buckets[arm, number]})
    if not groups or replicates < 1:
        raise ValueError("bootstrap_groups_or_replicates")
    def rates(sample: list[str]) -> dict[str, float]:
        answer = {}
        for arm in arms:
            passes = []
            for number in PASSES:
                numerator = sum(buckets[arm, number].get(group, (0, 0))[0] for group in sample)
                denominator = sum(buckets[arm, number].get(group, (0, 0))[1] for group in sample)
                passes.append(numerator / denominator if denominator else 0.0)
            answer[arm] = sum(passes) / len(passes)
        return answer
    point = rates(groups)
    draws = {arm: [] for arm in arms}
    differences = {f"{left}_minus_{right}": [] for left, right in combinations(arms, 2)}
    rng = random.Random(20260923)
    for _ in range(replicates):
        values = rates(rng.choices(groups, k=len(groups)))
        for arm in arms:
            draws[arm].append(values[arm])
        for left, right in combinations(arms, 2):
            differences[f"{left}_minus_{right}"].append(values[left] - values[right])
    percentile = evaluate_selection._percentile
    return {"unit": "source_group_with_all_scenarios_and_passes", "groups": len(groups),
            "replicates": replicates, "seed": 20260923,
            "by_arm": {arm: {"point": point[arm], "ci95": [percentile(draws[arm], 0.025),
                                                             percentile(draws[arm], 0.975)]} for arm in arms},
            "paired_differences": {name: {"point": point[name.split("_minus_")[0]] - point[name.split("_minus_")[1]],
                                           "ci95": [percentile(values, 0.025), percentile(values, 0.975)]}
                                   for name, values in differences.items()}}


def _workflow_bootstrap(workflow: dict, replicates: int) -> dict:
    groups = {row["case_id"]: row["group_id"] for row in workflow["planned_cases"]}
    buckets = defaultdict(lambda: [0, 0])
    for arm in ARMS:
        for number in PASSES:
            observed = workflow["by_slot"][arm, number]
            for (case, scenario), points in workflow["trials"].items():
                bucket = buckets[arm, number, groups[case]]
                for point in points:
                    row = observed.get((case, scenario, point["step"]))
                    bucket[1] += 1
                    bucket[0] += bool(row is not None and row.get("provider_outcome") == "ok"
                                      and row.get("action") == point["expected_action"] and row.get("compliant") is True)
    nested = {(arm, number): {group: tuple(values) for (a, n, group), values in buckets.items()
                              if a == arm and n == number} for arm in ARMS for number in PASSES}
    result = _group_bootstrap_rates(nested, ARMS, replicates)
    result["metric"] = "mean_of_pass_first_attempt_oracle_compliance"
    return result


def _verdict_bootstrap(verdicts: dict, gold: dict, planned_cases: list[dict], replicates: int) -> dict:
    roster = evaluate._roster(planned_cases, len(planned_cases))
    labels = evaluate._gold(gold, roster)
    observed = {(row["arm"], row["pass"], row["case_id"]): row for row in verdicts["rows"]}
    arms = (*ARMS, *evaluate.CONTROLS)
    buckets = defaultdict(lambda: [0, 0])
    for arm in arms:
        for number in PASSES:
            for case, group in roster.items():
                row = observed.get((arm, number, case))
                bucket = buckets[arm, number, group]
                bucket[1] += 1
                bucket[0] += bool(row is not None and row.get("outcome") == "ok" and row.get("verdict") == labels[case])
    nested = {(arm, number): {group: tuple(values) for (a, n, group), values in buckets.items()
                              if a == arm and n == number} for arm in arms for number in PASSES}
    result = _group_bootstrap_rates(nested, arms, replicates)
    result["metric"] = "mean_of_pass_valid_correct_verdict_over_planned"
    return result


def _selection_correctness_repeatability(choices: dict, candidates: dict, reference: dict,
                                         expected_cases: int) -> dict:
    roster, _ = evaluate_selection._pool(candidates, expected_cases, 10)
    ratings = evaluate_selection._ratings(reference, roster, None)
    observed = evaluate_selection._choices(choices, roster, ARMS, PASSES)
    known = [key for key, grade in ratings.items() if grade != "U"]
    by_arm = {}
    for arm in ARMS:
        pairs = {}
        for first, second in combinations(PASSES, 2):
            correct = sum(observed.get((arm, first, *key)) == observed.get((arm, second, *key))
                          == ("include" if ratings[key] == 2 else "exclude") for key in known)
            pairs[f"{first}-{second}"] = {"both_correct": correct, "known_denominator": len(known),
                                          "both_correct_over_known": correct / len(known)}
        three = sum(all(observed.get((arm, number, *key)) ==
                        ("include" if ratings[key] == 2 else "exclude") for number in PASSES) for key in known)
        by_arm[arm] = {"pairwise": pairs, "all_three_correct": three,
                       "all_three_correct_over_known": three / len(known)}
    return {"target": "grade_2_vs_0_or_1", "known_denominator": len(known), "by_arm": by_arm}


def _verdict_repeatability(verdicts: dict, gold: dict, planned_cases: list[dict]) -> dict:
    labels = evaluate._gold(gold, evaluate._roster(planned_cases, len(planned_cases)))
    observed = {(row["arm"], row["pass"], row["case_id"]):
                row["verdict"] if row.get("outcome") == "ok" and row.get("verdict") in evaluate.VERDICT_LABELS else None
                for row in verdicts["rows"]}
    result = {}
    for arm in (*ARMS, *evaluate.CONTROLS):
        pairs = {}
        for first, second in combinations(PASSES, 2):
            same = both_correct = 0
            for case in labels:
                x, y = observed.get((arm, first, case)), observed.get((arm, second, case))
                same += x is not None and x == y
                both_correct += x == y == labels[case]
            pairs[f"{first}-{second}"] = {"same_valid_verdict_over_planned": same / len(labels),
                                          "both_correct_over_planned": both_correct / len(labels)}
        three_same = three_correct = 0
        for case in labels:
            values = [observed.get((arm, number, case)) for number in PASSES]
            three_same += values[0] is not None and values[0] == values[1] == values[2]
            three_correct += all(value == labels[case] for value in values)
        result[arm] = {"pairwise": pairs, "all_three_same_valid_verdict_over_planned": three_same / len(labels),
                       "all_three_correct_over_planned": three_correct / len(labels)}
    return {"planned_cases": len(labels), "by_arm": result}


def load_pinned_verdict_gold(path: Path, planned_cases: list[dict], *, enforce_pin: bool = True) -> dict:
    """Reduce the pinned 500-row JSONL to the exact selected case IDs."""
    data = path.read_bytes()
    if enforce_pin and hashlib.sha256(data).hexdigest() != PINNED_VERDICT_GOLD_SHA256:
        raise ValueError("verdict_gold_pin_mismatch")
    roster = {row["case_id"]: row["group_id"] for row in planned_cases}
    if len(roster) != len(planned_cases):
        raise ValueError("verdict_gold_planned_case_duplicates")
    seen = set()
    selected = []
    source_rows = 0
    for line in data.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError("verdict_gold_source_duplicate_or_id")
        seen.add(case_id)
        source_rows += 1
        if case_id in roster:
            selected.append({"case_id": case_id, "group_id": roster[case_id], "label": row.get("label")})
    if enforce_pin and source_rows != 500:
        raise ValueError("verdict_gold_source_count")
    if len(selected) != len(roster):
        raise ValueError("verdict_gold_exact_join")
    return {"cases": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--check-inference", action="store_true")
    parser.add_argument("--selection-reference", type=Path)
    parser.add_argument("--verdict-gold", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    body = args.manifest.read_bytes()
    manifest = json.loads(body)
    manifest["_sha256"] = hashlib.sha256(body).hexdigest()
    validate_inference(manifest)
    if args.check_inference:
        print(json.dumps({"inference_valid": True, "manifest_sha256": manifest["_sha256"]}, sort_keys=True))
        return
    if not all((args.selection_reference, args.verdict_gold, args.output)):
        parser.error("scoring requires both evaluator references and --output")
    reference = json.loads(args.selection_reference.read_text(encoding="utf-8"))
    package_file = _bound(manifest["packages"])
    planned_cases = [{"case_id": row["case_id"], "group_id": row["group_id"]}
                     for row in package_file["packages"] if row["arm"] == "include_all" and row["pass"] == 1]
    gold = load_pinned_verdict_gold(args.verdict_gold, planned_cases)
    result = evaluate_all(manifest, reference, gold, expected_grade_counts={0: 247, 1: 514, 2: 230, "U": 9},
                          bootstrap_replicates=2000)
    result["evaluator_input_sha256"] = {"selection_reference": hashlib.sha256(args.selection_reference.read_bytes()).hexdigest(),
                                        "verdict_gold": hashlib.sha256(args.verdict_gold.read_bytes()).hexdigest()}
    # `by_grade` has numeric grades 0/1/2 and the string grade U. JSON permits
    # both, but sorting mixed Python key types raises TypeError.
    output = (json.dumps(result, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(output)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": result["schema"], "output_sha256": hashlib.sha256(output).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
