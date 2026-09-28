"""Offline relevance and repeatability scoring for sealed three-pass choices.

The reference is evaluator-only. Inference packages must never import this file
or contain reference grades. Missing and invalid choices remain abstentions,
not inferred ``exclude`` decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
from typing import Any


ARMS = ("jev", "lfm", "laya_typed", "gemini", "qwen", "lfm26", "jeff")
PASSES = (1, 2, 3)
GRADE_ONE_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0)
ACTIONS = ("include", "exclude")


def _pool(candidates: Mapping[str, Any], expected_cases: int,
          candidates_per_case: int) -> tuple[dict[tuple[str, str], str], dict[str, list[str]]]:
    if candidates.get("gold_included") is not False or not isinstance(candidates.get("cases"), list):
        raise ValueError("selection_gold_free_pool_required")
    cases = candidates["cases"]
    if len(cases) != expected_cases:
        raise ValueError("selection_case_count")
    roster: dict[tuple[str, str], str] = {}
    by_case: dict[str, list[str]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("selection_case_invalid")
        case_id, group_id, rows = case.get("case_id"), case.get("group_id"), case.get("candidates")
        if (not isinstance(case_id, str) or not case_id or not isinstance(group_id, str)
                or not group_id or not isinstance(rows, list) or len(rows) != candidates_per_case
                or case_id in by_case):
            raise ValueError("selection_case_invalid")
        ids = []
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError("selection_candidate_invalid")
            key = (case_id, row["id"])
            if key in roster:
                raise ValueError("selection_candidate_duplicate")
            roster[key] = group_id
            ids.append(row["id"])
        by_case[case_id] = ids
    return roster, by_case


def _ratings(reference: Mapping[str, Any], roster: Mapping[tuple[str, str], str],
             expected_counts: Mapping[Any, int] | None) -> dict[tuple[str, str], int | str]:
    if not isinstance(reference.get("ratings"), list):
        raise ValueError("selection_reference_required")
    ratings: dict[tuple[str, str], int | str] = {}
    for row in reference["ratings"]:
        if not isinstance(row, Mapping):
            raise ValueError("selection_reference_row")
        key = (row.get("case_id"), row.get("id"))
        grade = row.get("grade")
        if key not in roster or key in ratings or not ((type(grade) is int and grade in (0, 1, 2)) or grade == "U"):
            raise ValueError("selection_reference_join_or_grade")
        ratings[key] = grade
    if set(ratings) != set(roster):
        raise ValueError("selection_reference_join_or_grade")
    if expected_counts is not None and Counter(ratings.values()) != Counter(expected_counts):
        raise ValueError("selection_reference_counts")
    return ratings


def _choices(sealed: Mapping[str, Any], roster: Mapping[tuple[str, str], str],
             arms: Sequence[str], passes: Sequence[int]) -> dict[tuple[str, int, str, str], str | None]:
    if sealed.get("sealed") is not True or not isinstance(sealed.get("rows"), list):
        raise ValueError("selection_predictions_not_sealed")
    if "source_receipts" not in sealed or not isinstance(sealed["source_receipts"], list):
        raise ValueError("selection_source_receipts_required")
    observed: dict[tuple[str, int, str, str], str | None] = {}
    for row in sealed["rows"]:
        if not isinstance(row, Mapping):
            raise ValueError("selection_prediction_row")
        arm, number, case_id, candidate_id = (row.get(name) for name in ("arm", "pass", "case_id", "id"))
        if (arm not in arms or type(number) is not int or number not in passes
                or (case_id, candidate_id) not in roster):
            raise ValueError("selection_prediction_unknown_id")
        key = (arm, number, case_id, candidate_id)
        if key in observed:
            raise ValueError("selection_prediction_duplicate")
        if "group_id" in row and row["group_id"] != roster[(case_id, candidate_id)]:
            raise ValueError("selection_prediction_group")
        observed[key] = row.get("action") if row.get("outcome") == "ok" and row.get("action") in ACTIONS else None
    return observed


def _binary(known: list[tuple[int, str | None]], positive_grades: set[int]) -> dict[str, Any]:
    counts = {"positive": {"include": 0, "exclude": 0, "failure": 0},
              "negative": {"include": 0, "exclude": 0, "failure": 0}}
    for grade, action in known:
        truth = "positive" if grade in positive_grades else "negative"
        counts[truth][action or "failure"] += 1
    def class_f1(truth: str, predicted: str) -> float:
        other = "negative" if truth == "positive" else "positive"
        tp = counts[truth][predicted]
        fp = counts[other][predicted]
        fn = sum(counts[truth].values()) - tp
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"known_denominator": len(known), "confusion": counts,
            "macro_f1": (class_f1("positive", "include") + class_f1("negative", "exclude")) / 2,
            "positive_f1": class_f1("positive", "include"),
            "negative_f1": class_f1("negative", "exclude")}


def _graded(known: list[tuple[int, str | None]], weight: float) -> dict[str, Any]:
    gain = {0: 0.0, 1: weight, 2: 1.0}
    selected = [grade for grade, action in known if action == "include"]
    retrieved_gain = sum(gain[grade] for grade in selected)
    available_gain = sum(gain[grade] for grade, _ in known)
    precision = retrieved_gain / len(selected) if selected else None
    recall = retrieved_gain / available_gain if available_gain else None
    return {"grade_one_gain": weight, "selected_known": len(selected),
            "retrieved_gain": retrieved_gain, "available_gain": available_gain,
            "graded_precision": precision, "graded_recall": recall}


def _repeatability(by_case: Mapping[str, list[str]],
                   observed: Mapping[tuple[str, int, str, str], str | None],
                   arm: str, passes: Sequence[int]) -> dict[str, Any]:
    pairs = {}
    for first, second in combinations(passes, 2):
        action_agree = exact_sets = valid_sets = 0
        jaccard_sum = 0.0
        valid_actions = 0
        for case_id, ids in by_case.items():
            left = [observed.get((arm, first, case_id, item)) for item in ids]
            right = [observed.get((arm, second, case_id, item)) for item in ids]
            action_agree += sum(a is not None and b is not None and a == b for a, b in zip(left, right))
            valid_actions += sum(a is not None and b is not None for a, b in zip(left, right))
            if any(action is None for action in (*left, *right)):
                continue
            valid_sets += 1
            a = {item for item, action in zip(ids, left) if action == "include"}
            b = {item for item, action in zip(ids, right) if action == "include"}
            exact_sets += a == b
            jaccard_sum += len(a & b) / len(a | b) if a | b else 1.0
        planned_actions = sum(map(len, by_case.values()))
        planned_cases = len(by_case)
        pairs[f"{first}-{second}"] = {
            "planned_actions": planned_actions, "valid_action_pairs": valid_actions,
            "action_agreement_over_planned": action_agree / planned_actions,
            "planned_cases": planned_cases, "valid_set_pairs": valid_sets,
            "exact_set_agreement_over_planned": exact_sets / planned_cases,
            "mean_jaccard_over_planned": jaccard_sum / planned_cases,
            "mean_jaccard_conditional_valid_sets": jaccard_sum / valid_sets if valid_sets else None,
            "both_empty_jaccard": 1.0,
        }
    all_agree = 0
    for case_id, ids in by_case.items():
        for item in ids:
            values = [observed.get((arm, number, case_id, item)) for number in passes]
            all_agree += all(value is not None and value == values[0] for value in values)
    return {"pairwise": pairs, "all_pass_action_agreement_over_planned":
            all_agree / sum(map(len, by_case.values()))}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def _group_bootstrap(ratings: Mapping[tuple[str, str], int | str],
                     roster: Mapping[tuple[str, str], str],
                     observed: Mapping[tuple[str, int, str, str], str | None],
                     arms: Sequence[str], passes: Sequence[int],
                     replicates: int, seed: int) -> dict[str, Any]:
    groups = sorted(set(roster.values()))
    grouped: dict[tuple[str, int, str], list[tuple[int, str | None]]] = defaultdict(list)
    for (case_id, item), grade in ratings.items():
        if grade == "U":
            continue
        group_id = roster[(case_id, item)]
        for arm in arms:
            for number in passes:
                grouped[(arm, number, group_id)].append((grade, observed.get((arm, number, case_id, item))))

    def metric(sample: Sequence[str], arm: str) -> float:
        by_pass = []
        for number in passes:
            known = [pair for group_id in sample for pair in grouped[(arm, number, group_id)]]
            by_pass.append(_binary(known, {2})["macro_f1"])
        return sum(by_pass) / len(by_pass)

    point = {arm: metric(groups, arm) for arm in arms}
    draws = {arm: [] for arm in arms}
    differences = {f"{left}_minus_{right}": [] for left, right in combinations(arms, 2)}
    generator = random.Random(seed)
    for _ in range(replicates):
        sample = generator.choices(groups, k=len(groups))
        values = {arm: metric(sample, arm) for arm in arms}
        for arm, value in values.items():
            draws[arm].append(value)
        for left, right in combinations(arms, 2):
            differences[f"{left}_minus_{right}"].append(values[left] - values[right])
    return {"unit": "source_group_with_all_candidates_and_passes", "groups": len(groups),
            "replicates": replicates, "seed": seed, "metric": "mean_of_pass_grade2_macro_f1",
            "by_arm": {arm: {"point": value, "ci95": [_percentile(draws[arm], 0.025),
                                                   _percentile(draws[arm], 0.975)]}
                       for arm, value in point.items()},
            "paired_differences": {name: {"point": point[name.split("_minus_")[0]]
                                                    - point[name.split("_minus_")[1]],
                                           "ci95": [_percentile(values, 0.025),
                                                    _percentile(values, 0.975)]}
                                   for name, values in differences.items()}}


def evaluate_selection(sealed: Mapping[str, Any], candidates: Mapping[str, Any],
                       reference: Mapping[str, Any], *, arms: Sequence[str] = ARMS,
                       passes: Sequence[int] = PASSES, expected_cases: int = 100,
                       candidates_per_case: int = 10,
                       expected_grade_counts: Mapping[Any, int] | None = None,
                       bootstrap_replicates: int = 0, bootstrap_seed: int = 20260923) -> dict[str, Any]:
    """Join sealed choices to evaluator-only grades without changing failures."""
    if (not arms or len(set(arms)) != len(arms) or not passes or len(set(passes)) != len(passes)
            or any(not isinstance(arm, str) or not arm for arm in arms)
            or any(type(number) is not int or number < 1 for number in passes)
            or type(bootstrap_replicates) is not int or bootstrap_replicates < 0
            or type(bootstrap_seed) is not int):
        raise ValueError("selection_plan")
    roster, by_case = _pool(candidates, expected_cases, candidates_per_case)
    ratings = _ratings(reference, roster, expected_grade_counts)
    observed = _choices(sealed, roster, arms, passes)
    scored: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in arms:
        scored[arm] = {}
        for number in passes:
            labeled: list[tuple[int, str | None]] = []
            outcomes = Counter()
            strata = {grade: {"planned": 0, "included": 0, "excluded": 0, "missing": 0,
                              "invalid": 0} for grade in (0, 1, 2, "U")}
            for (case_id, item), grade in ratings.items():
                key = (arm, number, case_id, item)
                action = observed.get(key)
                status = action if action is not None else "invalid" if key in observed else "missing"
                outcomes[status] += 1
                strata[grade]["planned"] += 1
                strata[grade][{"include": "included", "exclude": "excluded",
                                "missing": "missing", "invalid": "invalid"}[status]] += 1
                if grade != "U":
                    labeled.append((grade, action))
            for row in strata.values():
                row["inclusion_over_planned"] = row["included"] / row["planned"] if row["planned"] else None
            scored[arm][number] = {
                "planned_candidates": len(roster), "known_grades": len(labeled),
                "unknown_grades": len(roster) - len(labeled), "outcomes": dict(outcomes),
                "primary_grade2": _binary(labeled, {2}),
                "secondary_grade1_or_2": _binary(labeled, {1, 2}),
                "by_grade": strata,
                "graded_gain_sensitivity": [_graded(labeled, weight) for weight in GRADE_ONE_GAINS],
            }
    repeatability = {arm: _repeatability(by_case, observed, arm, passes) for arm in arms}
    bootstrap = (_group_bootstrap(ratings, roster, observed, arms, passes,
                                  bootstrap_replicates, bootstrap_seed)
                 if bootstrap_replicates else None)
    return {"schema": "three-pass-selection-evaluation/v1", "planned_cases": len(by_case),
            "planned_candidates_per_pass": len(roster), "reference_type": reference.get("reference_type"),
            "source_receipts": sealed["source_receipts"], "by_arm_pass": scored,
            "repeatability": repeatability, "group_bootstrap": bootstrap, "reference_caveat":
            "Agreement with AI-adjudicated and targeted-human-reviewed relevance reference; not independent human gold."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-choices", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {"sealed_choices": args.sealed_choices, "candidates": args.candidates,
             "reference": args.reference}
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    result = evaluate_selection(values["sealed_choices"], values["candidates"], values["reference"],
                                expected_grade_counts={0: 247, 1: 514, 2: 230, "U": 9},
                                bootstrap_replicates=2000)
    result["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in paths.items()}
    body = (json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": result["schema"], "output_sha256": hashlib.sha256(body).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
