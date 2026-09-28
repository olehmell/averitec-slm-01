"""Evaluator-only, offline four-class verdict scoring after inference is sealed.

This module must never be imported by an inference runner. Its ``gold`` input
is a separate verdict reference, not the passage-relevance ``ratings`` file.
The published evidence/Q/A scorer is deliberately outside this module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


VERDICT_LABELS = (
    "Supported", "Refuted", "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)
SELECTOR_ARMS = ("jev", "lfm", "laya_typed", "gemini", "qwen", "lfm26", "jeff")
CONTROLS = ("include_all", "include_none")
FAILURE_COLUMN = "__missing_or_invalid__"


def _roster(planned_cases: Sequence[Mapping[str, Any]], expected_cases: int) -> dict[str, str]:
    if not isinstance(planned_cases, list) or len(planned_cases) != expected_cases:
        raise ValueError("planned_case_count")
    roster: dict[str, str] = {}
    for row in planned_cases:
        if not isinstance(row, Mapping):
            raise ValueError("invalid_planned_case")
        case_id, group_id = row.get("case_id"), row.get("group_id")
        if not isinstance(case_id, str) or not case_id or not isinstance(group_id, str) or not group_id:
            raise ValueError("invalid_planned_case")
        if case_id in roster:
            raise ValueError("duplicate_planned_case")
        roster[case_id] = group_id
    return roster


def _gold(gold: Mapping[str, Any], roster: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(gold, Mapping) or not isinstance(gold.get("cases"), list):
        raise ValueError("verdict_gold_required_not_relevance_ratings")
    cases = gold["cases"]
    if len(cases) != len(roster):
        raise ValueError("verdict_gold_case_count")
    labels: dict[str, str] = {}
    for row in cases:
        if not isinstance(row, Mapping):
            raise ValueError("invalid_verdict_gold")
        case_id, group_id, label = row.get("case_id"), row.get("group_id"), row.get("label")
        if not isinstance(case_id, str) or case_id not in roster or case_id in labels:
            raise ValueError("verdict_gold_id_join")
        if group_id != roster[case_id] or label not in VERDICT_LABELS:
            raise ValueError("verdict_gold_group_or_label")
        labels[case_id] = label
    if set(labels) != set(roster):
        raise ValueError("verdict_gold_id_join")
    return labels


def _metrics(rows: list[dict], denominator: int) -> dict:
    columns = (*VERDICT_LABELS, FAILURE_COLUMN)
    confusion = {truth: {prediction: 0 for prediction in columns} for truth in VERDICT_LABELS}
    for row in rows:
        confusion[row["gold_label"]][row["predicted_verdict"] or FAILURE_COLUMN] += 1
    per_class = {}
    for label in VERDICT_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in VERDICT_LABELS if other != label)
        fn = sum(value for prediction, value in confusion[label].items() if prediction != label)
        per_class[label] = {"support": sum(confusion[label].values()), "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "recall": tp / (tp + fn) if tp + fn else 0.0,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {
        "planned_cases": denominator,
        "correct": sum(row["correct"] for row in rows),
        "accuracy_over_planned": sum(row["correct"] for row in rows) / denominator,
        "valid_verdicts": sum(row["predicted_verdict"] is not None for row in rows),
        "missing_predictions": sum(row["status"] == "missing" for row in rows),
        "invalid_predictions": sum(row["status"] == "invalid" for row in rows),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(VERDICT_LABELS),
        "per_class": per_class,
        "confusion": confusion,
    }


def evaluate_verdicts(
    sealed_predictions: Mapping[str, Any],
    planned_cases: list[Mapping[str, Any]],
    gold: Mapping[str, Any],
    *,
    arms: Sequence[str] = (*SELECTOR_ARMS, *CONTROLS),
    passes: Sequence[int] = (1, 2, 3),
    expected_cases: int = 100,
) -> dict:
    """Score every arm/pass against an exact, unique evaluator-only gold join.

    Input prediction rows use ``arm``, ``pass``, ``case_id``, ``outcome`` and
    ``verdict``. Absent rows and non-``ok``/unknown verdicts remain failures in
    the planned denominator. Duplicate or foreign IDs are rejected globally.
    """
    if not isinstance(sealed_predictions, Mapping) or sealed_predictions.get("sealed") is not True:
        raise ValueError("predictions_not_sealed")
    rows = sealed_predictions.get("rows")
    if not isinstance(rows, list):
        raise ValueError("prediction_rows_required")
    if (type(expected_cases) is not int or expected_cases < 1
            or not arms or len(set(arms)) != len(arms) or not passes or len(set(passes)) != len(passes)
            or any(not isinstance(arm, str) or not arm for arm in arms)
            or any(type(number) is not int or number < 1 for number in passes)):
        raise ValueError("invalid_evaluation_plan")
    roster = _roster(planned_cases, expected_cases)
    labels = _gold(gold, roster)
    allowed_arms, allowed_passes = set(arms), set(passes)
    observed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid_prediction_row")
        arm, number, case_id = row.get("arm"), row.get("pass"), row.get("case_id")
        if (not isinstance(arm, str) or arm not in allowed_arms or type(number) is not int
                or number not in allowed_passes or not isinstance(case_id, str) or case_id not in roster):
            raise ValueError("unknown_prediction_id")
        if "group_id" in row and row["group_id"] != roster[case_id]:
            raise ValueError("prediction_group_mismatch")
        key = (arm, number, case_id)
        if key in observed:
            raise ValueError("duplicate_prediction_id")
        observed[key] = row

    scored: dict[str, dict[int, dict]] = {}
    case_scores: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for arm in arms:
        scored[arm] = {}
        for number in passes:
            joined = []
            for case_id, group_id in roster.items():
                row = observed.get((arm, number, case_id))
                valid = row is not None and row.get("outcome") == "ok" and row.get("verdict") in VERDICT_LABELS
                status = "missing" if row is None else "valid" if valid else "invalid"
                prediction = row["verdict"] if valid else None
                item = {"case_id": case_id, "group_id": group_id, "gold_label": labels[case_id],
                        "predicted_verdict": prediction, "status": status,
                        "correct": bool(valid and prediction == labels[case_id])}
                joined.append(item)
                case_scores[arm][number][case_id] = item
            scored[arm][number] = _metrics(joined, expected_cases)

    # Paired comparisons retain source groups as the matching unit. No bootstrap
    # is inferred from repeated passes or from individual passages.
    paired = {}
    for arm in arms:
        if arm in CONTROLS:
            continue
        paired[arm] = {}
        for control in CONTROLS:
            if control not in scored:
                continue
            paired[arm][control] = {}
            for number in passes:
                groups: dict[str, list[int]] = defaultdict(list)
                for case_id, group_id in roster.items():
                    groups[group_id].append(int(case_scores[arm][number][case_id]["correct"])
                                            - int(case_scores[control][number][case_id]["correct"]))
                group_deltas = {group_id: sum(values) / len(values) for group_id, values in groups.items()}
                paired[arm][control][number] = {
                    "groups": len(group_deltas),
                    "mean_group_accuracy_difference": sum(group_deltas.values()) / len(group_deltas),
                    "group_differences": group_deltas,
                }
    return {"schema": "three-pass-verdict-evaluation/v1", "planned_cases": expected_cases,
            "labels": list(VERDICT_LABELS), "by_arm_pass": scored, "paired_vs_controls": paired,
            "official_qa_scorer": "unavailable_not_run_qa_contract_unverified"}
