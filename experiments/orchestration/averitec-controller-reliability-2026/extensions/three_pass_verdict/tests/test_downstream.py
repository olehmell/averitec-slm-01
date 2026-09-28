"""Synthetic, offline checks for selected-passage package assembly."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "downstream.py"
SPEC = importlib.util.spec_from_file_location("three_pass_downstream", MODULE_PATH)
assert SPEC and SPEC.loader
downstream = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downstream)


def candidates() -> dict:
    cases = []
    for number in (1, 2):
        rows = []
        for rank in range(10):
            rows.append({
                "id": f"c{number}-p{rank}", "passage_id": f"passage-{number}-{rank}",
                "text": f"Exact source excerpt {number}/{rank}", "url": f"https://example.test/{number}/{rank}",
                "source_start": rank * 17, "source_text_length": 500 + number,
                "source_text_sha256": f"{number:064x}",
                "reference_grade": 2, "gold_answer": "must never leave this input",
            })
        cases.append({"case_id": f"case-{number}", "group_id": f"group-{number}",
                      "claim": f"Claim {number}", "gold_verdict": "Supported", "candidates": rows})
    return {"gold_included": False, "cases": cases, "reference_labels": {"case-1": "Supported"}}


def outcomes(document: dict, included: set[str] | None = None) -> list[dict]:
    included = included or set()
    return [{"case_id": case["case_id"], "id": candidate["id"],
             "action": "include" if candidate["id"] in included else "exclude", "outcome": "ok"}
            for case in document["cases"] for candidate in case["candidates"]]


def build(document: dict, rows: list[dict] | None = None, *, mode: str = "selector") -> list[dict]:
    return downstream.build_packages(document, rows, mode=mode, expected_cases=2)


def test_exact_join_canonical_order_metadata_and_gold_boundary() -> None:
    document = candidates()
    rows = outcomes(document, {"c1-p8", "c1-p1", "c2-p3"})
    rows.reverse()  # Model output order cannot reorder source passages.
    result = build(document, rows)
    assert len(result) == 2
    assert [case["case_id"] for case in result] == ["case-1", "case-2"]
    assert [p["id"] for p in result[0]["selected_passages"]] == ["c1-p1", "c1-p8"]
    assert result[0]["selected_passages"][0] == {
        key: document["cases"][0]["candidates"][1][key] for key in downstream.PASSAGE_FIELDS
    }
    assert result[0]["group_id"] == "group-1" and result[0]["claim"] == "Claim 1"
    assert all(case["status"] == "ready" for case in result)
    assert "gold" not in json.dumps(result) and "reference" not in json.dumps(result)


def test_valid_all_exclude_and_deterministic_controls() -> None:
    document = candidates()
    result = build(document, outcomes(document))
    assert all(case["status"] == "ready" and case["selected_passages"] == [] for case in result)
    all_control = build(document, mode="include_all")
    none_control = build(document, mode="include_none")
    assert [len(case["selected_passages"]) for case in all_control] == [10, 10]
    assert [case["selected_passages"] for case in none_control] == [[], []]
    assert all_control == build(document, mode="include_all")
    with pytest.raises(ValueError, match="control_modes"):
        build(document, outcomes(document), mode="include_all")


@pytest.mark.parametrize("mutate, expected", [
    (lambda rows: rows.append(rows[0].copy()), "duplicate_candidate_id:c1-p0"),
    (lambda rows: rows.pop(0), "missing_candidate_id:c1-p0"),
    (lambda rows: rows[0].update(id="not-in-pool"), "unknown_candidate_id:not-in-pool"),
    (lambda rows: rows[0].update(outcome="transport_error", action="exclude"), "invalid_outcome:c1-p0"),
    (lambda rows: rows[0].update(outcome="ok", action="maybe"), "invalid_action:c1-p0"),
    (lambda rows: rows[0].update(outcome="ok", action=None), "invalid_action:c1-p0"),
])
def test_bad_measurement_fails_its_case_without_partial_package(mutate, expected: str) -> None:
    document = candidates()
    rows = outcomes(document, {"c1-p5", "c2-p5"})
    mutate(rows)
    result = build(document, rows)
    assert result[0]["status"] == "failed"
    assert expected in result[0]["errors"]
    assert result[0]["selected_passages"] is None
    assert result[1]["status"] == "ready"
    assert [p["id"] for p in result[1]["selected_passages"]] == ["c2-p5"]


def test_unknown_case_rejected_and_every_planned_case_reported() -> None:
    document = candidates()
    rows = outcomes(document)
    with pytest.raises(ValueError, match="unknown_measured_case_id"):
        build(document, rows + [{"case_id": "ghost", "id": "x", "action": "exclude", "outcome": "ok"}])
    result = build(document, rows[:10])
    assert len(result) == 2
    assert result[0]["status"] == "ready"
    assert result[1]["status"] == "failed"
    assert result[1]["selected_passages"] is None
    assert len(result[1]["errors"]) == 10


def test_source_shape_and_gold_marker_are_required() -> None:
    document = candidates()
    document["gold_included"] = True
    with pytest.raises(ValueError, match="gold_free"):
        build(document, outcomes(document))
    document["gold_included"] = False
    document["cases"][0]["candidates"][0]["source_start"] = -1
    with pytest.raises(ValueError, match="invalid_source_metadata"):
        build(document, outcomes(document))
