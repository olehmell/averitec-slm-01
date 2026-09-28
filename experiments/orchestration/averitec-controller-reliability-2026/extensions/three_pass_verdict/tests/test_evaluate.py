"""Narrow evaluator-only checks with synthetic verdict gold."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("three_pass_evaluate", MODULE_PATH)
assert SPEC and SPEC.loader
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)

ROSTER = [{"case_id": "a", "group_id": "source-one"}, {"case_id": "b", "group_id": "source-two"}]
GOLD = {"cases": [{"case_id": "a", "group_id": "source-one", "label": "Supported"},
                  {"case_id": "b", "group_id": "source-two", "label": "Refuted"}]}


def row(arm: str, case_id: str, verdict: str | None, *, outcome: str = "ok", number: int = 1) -> dict:
    return {"arm": arm, "pass": number, "case_id": case_id, "outcome": outcome, "verdict": verdict}


def score(rows: list[dict], **kwargs) -> dict:
    return evaluate.evaluate_verdicts({"sealed": True, "rows": rows}, ROSTER, GOLD,
                                      arms=("selector", "include_all", "include_none"),
                                      passes=(1,), expected_cases=2, **kwargs)


def test_denominator_confusion_macro_f1_and_paired_groups() -> None:
    results = score([
        row("selector", "a", "Supported"),
        row("selector", "b", "Refuted", outcome="transport_error"),
        row("include_all", "a", "Refuted"), row("include_all", "b", "Refuted"),
        row("include_none", "a", None, outcome="invalid_output"),
        row("include_none", "b", "Supported"),
    ])
    selector = results["by_arm_pass"]["selector"][1]
    assert selector["planned_cases"] == 2
    assert selector["correct"] == 1 and selector["accuracy_over_planned"] == 0.5
    assert selector["valid_verdicts"] == 1 and selector["invalid_predictions"] == 1
    assert selector["macro_f1"] == 0.25  # Fixed four-class macro, absent classes score zero.
    assert selector["confusion"]["Refuted"][evaluate.FAILURE_COLUMN] == 1
    assert results["paired_vs_controls"]["selector"]["include_all"][1]["group_differences"] == {
        "source-one": 1.0, "source-two": -1.0,
    }
    assert results["official_qa_scorer"].startswith("unavailable")


def test_missing_row_counts_as_failure_for_every_planned_arm_pass() -> None:
    results = score([row("selector", "a", "Supported")])
    selector = results["by_arm_pass"]["selector"][1]
    assert selector["missing_predictions"] == 1 and selector["accuracy_over_planned"] == 0.5
    control = results["by_arm_pass"]["include_all"][1]
    assert control["missing_predictions"] == 2 and control["accuracy_over_planned"] == 0.0


@pytest.mark.parametrize("rows, message", [
    ([row("selector", "a", "Supported"), row("selector", "a", "Supported")], "duplicate_prediction_id"),
    ([row("selector", "foreign", "Supported")], "unknown_prediction_id"),
    ([row("unplanned_arm", "a", "Supported")], "unknown_prediction_id"),
    ([row("selector", "a", "Supported", number=4)], "unknown_prediction_id"),
])
def test_prediction_join_rejects_foreign_or_duplicate_ids(rows, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        score(rows)


def test_invalid_label_is_failure_not_class_or_correct() -> None:
    results = score([row("selector", "a", "supported"), row("selector", "b", "Refuted")])
    selector = results["by_arm_pass"]["selector"][1]
    assert selector["invalid_predictions"] == 1 and selector["correct"] == 1
    assert selector["confusion"]["Supported"][evaluate.FAILURE_COLUMN] == 1


def test_gold_boundary_exact_case_and_group_join() -> None:
    with pytest.raises(ValueError, match="predictions_not_sealed"):
        evaluate.evaluate_verdicts({"rows": []}, ROSTER, GOLD, arms=("selector",), passes=(1,), expected_cases=2)
    with pytest.raises(ValueError, match="not_relevance_ratings"):
        evaluate.evaluate_verdicts({"sealed": True, "rows": []}, ROSTER,
                                   {"ratings": [{"case_id": "a", "id": "C01", "grade": 2}]},
                                   arms=("selector",), passes=(1,), expected_cases=2)
    wrong = {"cases": [dict(GOLD["cases"][0]), {**GOLD["cases"][1], "case_id": "other"}]}
    with pytest.raises(ValueError, match="verdict_gold_id_join"):
        evaluate.evaluate_verdicts({"sealed": True, "rows": []}, ROSTER, wrong,
                                   arms=("selector",), passes=(1,), expected_cases=2)
    wrong_group = {"cases": [dict(GOLD["cases"][0]), {**GOLD["cases"][1], "group_id": "wrong"}]}
    with pytest.raises(ValueError, match="verdict_gold_group_or_label"):
        evaluate.evaluate_verdicts({"sealed": True, "rows": []}, ROSTER, wrong_group,
                                   arms=("selector",), passes=(1,), expected_cases=2)
    with pytest.raises(ValueError, match="prediction_group_mismatch"):
        score([{**row("selector", "a", "Supported"), "group_id": "wrong"}])
