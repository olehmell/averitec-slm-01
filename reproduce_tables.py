"""Recalculate CSIT article Tables 2–5 from the public, text-free result files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTICLE = ROOT / "results/article"
FINAL = ROOT / "experiments/orchestration/averitec-controller-reliability-2026/results/final"
ARMS = ("jev", "gemini", "qwen", "lfm26", "jeff", "laya_typed", "lfm")
LABEL = {"jev": "Jev", "gemini": "Gemini", "qwen": "Qwen-4B", "lfm26": "LFM-2.6B",
         "jeff": "Jeff", "laya_typed": "Laya typed", "lfm": "LFM-1.2B",
         "include_all": "All snippets", "include_none": "No snippets"}
VERDICT_LABELS = {"Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verified_article_rows() -> tuple[list[dict[str, str]], ...]:
    manifest = json.loads((ARTICLE / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for name in ("reference.csv", "selection-decisions.csv", "verdict-outcomes.csv", "workflow-decisions.csv"):
        path = ARTICLE / name
        expected = manifest["files"][name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
        data = rows(path)
        assert len(data) == expected["rows"], name
        result.append(data)
    return tuple(result)


def macro_f1(counts: Counter[tuple[int, str]]) -> float:
    positive = sum(n for (grade, _), n in counts.items() if grade == 2)
    negative = sum(n for (grade, _), n in counts.items() if grade != 2)
    included = sum(n for (_, action), n in counts.items() if action == "include")
    excluded = sum(n for (_, action), n in counts.items() if action == "exclude")
    true_positive = counts[(2, "include")]
    true_negative = counts[(0, "exclude")] + counts[(1, "exclude")]
    f1_positive = 2 * true_positive / (positive + included) if positive + included else 0.0
    f1_negative = 2 * true_negative / (negative + excluded) if negative + excluded else 0.0
    return (f1_positive + f1_negative) / 2


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def selection_tables(reference: list[dict[str, str]], selection: list[dict[str, str]]) -> tuple[list[list[str]], list[list[str]]]:
    grade = {(r["case_id"], r["candidate_id"]): int(r["final_grade"]) for r in reference}
    group = {(r["case_id"], r["candidate_id"]): r["source_group_id"] for r in reference}
    assert len(grade) == 1000 and Counter(grade.values()) == {0: 251, 1: 516, 2: 233}
    assert len({r["case_id"] for r in reference}) == len(set(group.values())) == 100
    assert len(selection) == 21000
    counts = defaultdict(Counter)
    per_group = defaultdict(Counter)
    seen = set()
    for row in selection:
        arm, number = row["deployment"], int(row["comparison_pass"])
        key = (row["case_id"], row["candidate_id"])
        assert arm in ARMS and number in (1, 2, 3) and key in grade
        assert int(row["raw_pass"]) == (4 if arm == "jev" and number == 3 else number)
        unique = (arm, number, *key)
        assert unique not in seen, unique
        seen.add(unique)
        action = row["decision"] if row["outcome"] == "ok" and row["decision"] in ("include", "exclude") else "failure"
        item = (grade[key], action)
        counts[(arm, number)][item] += 1
        per_group[(arm, number, group[key])][item] += 1
    assert len(seen) == 21000 and all(sum(c.values()) == 1000 for c in counts.values())

    groups = sorted(set(group.values()))
    def score(arm: str, sampled: list[str]) -> float:
        return sum(macro_f1(sum((per_group[(arm, number, g)] for g in sampled), Counter()))
                   for number in (1, 2, 3)) / 3

    point = {arm: sum(macro_f1(counts[(arm, number)]) for number in (1, 2, 3)) / 3 for arm in ARMS}
    bootstrap = {arm: [] for arm in ARMS}
    generator = random.Random(20260923)
    for _ in range(2000):
        sampled = generator.choices(groups, k=len(groups))
        for arm in ARMS:
            bootstrap[arm].append(score(arm, sampled))

    table3, table4 = [], []
    for arm in ARMS:
        interval = (percentile(bootstrap[arm], .025), percentile(bootstrap[arm], .975))
        selected = [sum(n for (g, action), n in counts[(arm, number)].items() if action == "include")
                    for number in (1, 2, 3)]
        table3.append([LABEL[arm], f"{point[arm]:.3f} ({interval[0]:.3f}-{interval[1]:.3f})",
                       " / ".join(map(str, selected))])
        rates = []
        for grade_value in (2, 1, 0):
            rates.append(100 * sum(counts[(arm, n)][(grade_value, "include")] /
                                   sum(v for (g, _), v in counts[(arm, n)].items() if g == grade_value)
                                   for n in (1, 2, 3)) / 3)
        graded_scores = []
        for number in (1, 2, 3):
            c = counts[(arm, number)]
            gain = c[(2, "include")] + .5 * c[(1, "include")]
            selected_n = sum(v for (_, action), v in c.items() if action == "include")
            available = sum(v for (g, _), v in c.items() if g == 2) + .5 * sum(v for (g, _), v in c.items() if g == 1)
            precision = gain / selected_n if selected_n else 0.0
            recall = gain / available if available else 0.0
            graded_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        table4.append([LABEL[arm], *(f"{value:.1f}" for value in rates), f"{sum(graded_scores)/3:.3f}"])
    return table3, table4


def workflow_table(decisions: list[dict[str, str]]) -> list[list[str]]:
    published = {(row["controller"], int(row["measurement"])): row for row in rows(FINAL / "workflow.csv")}
    by_run = defaultdict(list)
    seen = set()
    for row in decisions:
        arm, number, case_id = row["deployment"], int(row["measurement"]), row["case_id"]
        assert arm in ARMS and case_id.startswith("case_")
        key = (arm, number, case_id, row["scenario"], int(row["step"]))
        assert key not in seen
        seen.add(key)
        assert row["compliant"] in ("0", "1")
        assert (row["compliant"] == "1") == (row["provider_outcome"] == "ok" and row["selected_action"] == row["expected_action"])
        by_run[(arm, number)].append(row)
    assert len(decisions) == len(seen) == 45980
    assert set(by_run) == set(published)
    for key, source in published.items():
        events = by_run[key]
        assert len(events) == int(source["planned"]) == 2299
        assert len({(r["case_id"], r["scenario"]) for r in events}) == 300
        assert sum(r["compliant"] == "1" for r in events) == int(source["correct"])
        for action in ("abort", "finish"):
            subset = [r for r in events if r["expected_action"] == action]
            assert len(subset) == int(source[f"{action}_planned"])
            assert sum(r["compliant"] == "1" for r in subset) == int(source[f"{action}_correct"])
    result = []
    for arm in ARMS:
        source = [published[(arm, number)] for number in (1, 2, 3) if (arm, number) in published]
        assert len(source) == (2 if arm == "jev" else 3)
        assert len({(r["correct"], r["planned"], r["abort_correct"], r["abort_planned"],
                     r["finish_correct"], r["finish_planned"]) for r in source}) == 1
        first = source[0]
        assert int(first["planned"]) == 2299
        result.append([LABEL[arm], f"{int(first['correct']):,}/2,299", str(len(source)),
                       f"{first['abort_correct']}/{first['abort_planned']}",
                       f"{first['finish_correct']}/{first['finish_planned']}"])
    return result


def verdict_table(verdicts: list[dict[str, str]]) -> list[list[str]]:
    observed = defaultdict(list)
    seen = set()
    case_gold = {}
    for row in verdicts:
        arm, number, case_id = row["evidence_set"], int(row["comparison_pass"]), row["case_id"]
        assert arm in set(ARMS) | {"include_all", "include_none"} and number in (1, 2, 3)
        assert int(row["raw_pass"]) == (4 if arm == "jev" and number == 3 else number)
        key = (arm, number, case_id)
        assert key not in seen
        seen.add(key)
        assert row["valid"] in ("0", "1") and row["correct"] in ("0", "1")
        assert row["gold_label"] in VERDICT_LABELS
        if case_id in case_gold:
            assert case_gold[case_id] == row["gold_label"]
        case_gold[case_id] = row["gold_label"]
        assert (row["valid"] == "1") == (row["predicted_label"] in VERDICT_LABELS)
        assert int(row["correct"]) == int(row["valid"] == "1" and row["gold_label"] == row["predicted_label"])
        observed[(arm, number)].append(int(row["correct"]))
    assert len(verdicts) == len(seen) == 2700 and len(case_gold) == 100
    result = []
    for arm in ("include_all", "include_none", *ARMS):
        scores = []
        for number in (1, 2, 3):
            values = observed[(arm, number)]
            assert len(values) == 100
            scores.append(str(sum(values)))
        result.append([LABEL[arm], " / ".join(scores)])
    return result


def show(title: str, heading: list[str], data: list[list[str]]) -> None:
    print(f"\n{title}")
    print("| " + " | ".join(heading) + " |")
    print("| " + " | ".join("---" for _ in heading) + " |")
    for row in data:
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare calculated article values with the recorded article tables")
    args = parser.parse_args()
    reference, selection, verdicts, workflow = verified_article_rows()
    tables = {2: workflow_table(workflow), 5: verdict_table(verdicts)}
    tables[3], tables[4] = selection_tables(reference, selection)
    if args.check:
        expected = json.loads((ARTICLE / "expected-tables.json").read_text(encoding="utf-8"))
        for number in (2, 3, 4, 5):
            assert tables[number] == expected[str(number)], f"Table {number} differs from the manuscript"
        print("Tables 2–5 match the final manuscript.")
    else:
        show("Table 2. Workflow compliance across runs", ["Deployment", "Correct per run", "Runs", "Abort", "Finish"], tables[2])
        show("Table 3. Direct-evidence selection", ["Deployment", "Mean macro-F1 (95% CI)", "Selected per pass"], tables[3])
        show("Table 4. Inclusion by grade", ["Deployment", "Direct %", "Context %", "Irrelevant %", "Graded F1"], tables[4])
        show("Table 5. HerO verdicts", ["Evidence set", "Correct / 100, passes 1 / 2 / 3"], tables[5])


if __name__ == "__main__":
    main()
