"""The three-pass selector evaluator keeps abstentions and unknown grades distinct."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_selection as scoring  # noqa: E402


POOL = {"gold_included": False, "cases": [
    {"case_id": "a", "group_id": "g1", "candidates": [{"id": "C01"}, {"id": "C02"}]},
    {"case_id": "b", "group_id": "g2", "candidates": [{"id": "C01"}, {"id": "C02"}]},
]}
REFERENCE = {"reference_type": "synthetic", "ratings": [
    {"case_id": "a", "id": "C01", "grade": 2},
    {"case_id": "a", "id": "C02", "grade": 1},
    {"case_id": "b", "id": "C01", "grade": 0},
    {"case_id": "b", "id": "C02", "grade": "U"},
]}


def rows() -> list[dict]:
    result = []
    for number in (1, 2, 3):
        for case_id, candidate_id, action in (("a", "C01", "include"),
                                               ("a", "C02", "exclude"),
                                               ("b", "C01", "exclude"),
                                               ("b", "C02", "exclude")):
            if number == 2 and (case_id, candidate_id) == ("a", "C02"):
                continue
            result.append({"arm": "selector", "pass": number, "case_id": case_id,
                           "id": candidate_id, "outcome": "ok", "action": action})
    return result


def evaluate(predictions: list[dict]) -> dict:
    return scoring.evaluate_selection({"sealed": True, "rows": predictions, "source_receipts": []},
                                      POOL, REFERENCE, arms=("selector",), passes=(1, 2, 3),
                                      expected_cases=2, candidates_per_case=2,
                                      expected_grade_counts={0: 1, 1: 1, 2: 1, "U": 1})


def test_primary_secondary_graded_and_failure_denominators() -> None:
    result = evaluate(rows())
    first = result["by_arm_pass"]["selector"][1]
    assert first["primary_grade2"]["macro_f1"] == 1.0
    assert first["secondary_grade1_or_2"]["macro_f1"] == pytest.approx(2 / 3)
    assert first["graded_gain_sensitivity"][2]["graded_recall"] == pytest.approx(2 / 3)
    assert first["unknown_grades"] == 1 and first["outcomes"] == {"include": 1, "exclude": 3}
    second = result["by_arm_pass"]["selector"][2]
    assert second["primary_grade2"]["known_denominator"] == 3
    assert second["primary_grade2"]["macro_f1"] == pytest.approx(5 / 6)
    assert second["primary_grade2"]["confusion"]["negative"]["failure"] == 1
    assert second["by_grade"][1]["missing"] == 1
    assert second["by_grade"][1]["excluded"] == 0
    assert second["outcomes"] == {"include": 1, "missing": 1, "exclude": 2}


def test_invalid_choice_is_distinct_from_missing_and_never_becomes_exclude() -> None:
    decisions = rows()
    decisions[0]["outcome"] = "transport_error"
    result = evaluate(decisions)["by_arm_pass"]["selector"][1]
    assert result["by_grade"][2]["invalid"] == 1
    assert result["by_grade"][2]["excluded"] == 0
    assert result["primary_grade2"]["confusion"]["positive"]["failure"] == 1


def test_group_bootstrap_keeps_three_passes_paired_across_arms() -> None:
    original = rows()
    clone = [{**row, "arm": "clone"} for row in original]
    result = scoring.evaluate_selection(
        {"sealed": True, "rows": original + clone, "source_receipts": []},
        POOL, REFERENCE, arms=("selector", "clone"), passes=(1, 2, 3),
        expected_cases=2, candidates_per_case=2, bootstrap_replicates=20,
        bootstrap_seed=7)
    bootstrap = result["group_bootstrap"]
    assert bootstrap["unit"] == "source_group_with_all_candidates_and_passes"
    assert bootstrap["groups"] == 2 and bootstrap["replicates"] == 20
    assert bootstrap["paired_differences"]["selector_minus_clone"] == {
        "point": 0.0, "ci95": [0.0, 0.0]}


def test_repeatability_counts_incomplete_sets_as_failures_and_both_empty_as_one() -> None:
    repeat = evaluate(rows())["repeatability"]["selector"]
    one_two = repeat["pairwise"]["1-2"]
    assert one_two["planned_actions"] == 4 and one_two["valid_action_pairs"] == 3
    assert one_two["action_agreement_over_planned"] == 0.75
    assert one_two["valid_set_pairs"] == 1
    assert one_two["exact_set_agreement_over_planned"] == 0.5
    assert one_two["mean_jaccard_over_planned"] == 0.5
    assert one_two["mean_jaccard_conditional_valid_sets"] == 1.0
    assert repeat["all_pass_action_agreement_over_planned"] == 0.75


def test_seal_and_exact_joins_are_required() -> None:
    with pytest.raises(ValueError, match="not_sealed"):
        scoring.evaluate_selection({"rows": rows(), "source_receipts": []}, POOL, REFERENCE,
                                   arms=("selector",), expected_cases=2, candidates_per_case=2)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(rows() + [rows()[0]])
    bad = [dict(row) for row in rows()]
    bad[0]["id"] = "foreign"
    with pytest.raises(ValueError, match="unknown_id"):
        evaluate(bad)
    bad_ref = {**REFERENCE, "ratings": REFERENCE["ratings"][:-1]}
    with pytest.raises(ValueError, match="reference_join"):
        scoring.evaluate_selection({"sealed": True, "rows": rows(), "source_receipts": []},
                                   POOL, bad_ref, arms=("selector",), expected_cases=2,
                                   candidates_per_case=2)
