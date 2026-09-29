#!/usr/bin/env python3
"""Verify public workflow accounting and frozen-state identity without private logs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent
STAGES = {"decompose", "queries", "retrieve", "qa", "coverage", "select", "verdict"}
SCENARIOS = {"nominal", "retrieval_timeout_once", "retrieval_timeout_persistent"}


def rows(path: Path, expected: dict) -> list[dict[str, str]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    if len(result) != expected["rows"]:
        raise ValueError(f"Row-count mismatch: {path}")
    return result


def key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["case_id"], row["scenario"], int(row["step"])


def main() -> None:
    original = json.loads((ARTICLE / "manifest.json").read_text())["files"]
    derived = json.loads((HERE / "analysis-manifest.json").read_text())["files"]
    decisions = rows(ARTICLE / "workflow-decisions.csv", original["workflow-decisions.csv"])
    observations = rows(HERE / "workflow-observations.csv", derived["workflow-observations.csv"])
    hashes = rows(HERE / "workflow-state-hashes.csv", derived["workflow-state-hashes.csv"])

    baseline = {}
    counts = defaultdict(Counter)
    for row in observations:
        checkpoint = key(row)
        if checkpoint in baseline or row["scenario"] not in SCENARIOS:
            raise ValueError("Duplicate or invalid baseline checkpoint")
        baseline[checkpoint] = (row["expected_action"], row["observation_sha256"])
        c = counts[row["scenario"]]
        c["checkpoints"] += 1
        action = row["expected_action"]
        attempts = int(row["attempts_on_stage"])
        if action in STAGES:
            c["tool_dispatches"] += 1
            if attempts == 1:
                if action != row["stage"] or row["tool_status"] not in {"timeout", "invalid"}:
                    raise ValueError("Invalid retry observation")
                c[f"retry_after_{row['tool_status']}"] += 1
        elif action == "finish":
            c["finish"] += 1
        elif action == "abort":
            c["abort"] += 1
            if int(row["calls_remaining"]) < 8 - int(row["completed_stages_count"]):
                c["abort_budget_exhaustion"] += 1
            if attempts >= 2:
                c[f"abort_second_{row['tool_status']}"] += 1
        else:
            raise ValueError(f"Unknown action: {action}")
    if len(baseline) != 2299 or len({case for case, _, _ in baseline}) != 100:
        raise ValueError("Incomplete baseline grid")

    per_run = defaultdict(dict)
    for row in hashes:
        run = row["deployment"], int(row["measurement"])
        checkpoint = key(row)
        if checkpoint in per_run[run]:
            raise ValueError(f"Duplicate run-state row: {run}, {checkpoint}")
        per_run[run][checkpoint] = (row["expected_action"], row["observation_sha256"])
    if len(per_run) != 20 or any(states != baseline for states in per_run.values()):
        raise ValueError("Frozen states differ across complete runs")

    measured = defaultdict(dict)
    for row in decisions:
        run = row["deployment"], int(row["measurement"])
        checkpoint = key(row)
        if checkpoint in measured[run]:
            raise ValueError(f"Duplicate decision row: {run}, {checkpoint}")
        measured[run][checkpoint] = row["expected_action"]
    if set(measured) != set(per_run):
        raise ValueError("Measured run set differs from state-hash run set")
    for run, actions in measured.items():
        if actions != {checkpoint: value[0] for checkpoint, value in per_run[run].items()}:
            raise ValueError(f"Measured reference actions differ for {run}")

    totals = sum(counts.values(), Counter())
    expected = {"checkpoints": 2299, "tool_dispatches": 1999, "finish": 104,
                "abort": 196, "retry_after_timeout": 203, "retry_after_invalid": 96,
                "abort_second_timeout": 100, "abort_second_invalid": 96}
    if any(totals[name] != value for name, value in expected.items()) or totals["abort_budget_exhaustion"]:
        raise ValueError(f"Workflow accounting differs: {dict(totals)}")
    if totals["tool_dispatches"] - totals["retry_after_timeout"] - totals["retry_after_invalid"] != 1700:
        raise ValueError("Other dispatch count differs")
    scenario_expected = {"nominal": (849, 48), "retrieval_timeout_once": (949, 48),
                         "retrieval_timeout_persistent": (501, 100)}
    if any((counts[name]["checkpoints"], counts[name]["abort"]) != value
           for name, value in scenario_expected.items()):
        raise ValueError("Scenario counts differ")
    print("PASS: 20 complete runs share 2,299 public state hashes; 299 retries, 1,700 other dispatches, and 196 second-failure aborts per run.")


if __name__ == "__main__":
    main()
