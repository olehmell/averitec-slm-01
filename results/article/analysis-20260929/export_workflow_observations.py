#!/usr/bin/env python3
"""Export text-free frozen workflow states from the original private journals.

Run only with the original PhD workspace present. The public verification
command uses the generated CSVs and does not need private journals.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from frozen_workflow_audit import read_run, source_runs

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent
SCENARIOS = ("nominal", "retrieval_timeout_once", "retrieval_timeout_persistent")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(source_root: Path) -> None:
    runs = source_runs(source_root)
    baseline = None
    baseline_observations = None
    raw_runs = []
    for arm, number, kind, file, plan in runs:
        states, observations = read_run(source_root, kind, file, plan)
        if baseline is None:
            baseline, baseline_observations = states, observations
        if states != baseline:
            raise ValueError(f"Frozen states differ for {arm} run {number}")
        raw_runs.append((arm, number, states))

    raw_cases = sorted({case for case, _, _ in baseline})
    if len(raw_cases) != 100:
        raise ValueError("Expected 100 source cases")
    pseudonym = {case: f"case_{index:03d}" for index, case in enumerate(raw_cases, 1)}

    # The published IDs were assigned in sorted source-case order. Check the
    # complete public grid before reusing that assignment for the new export.
    public = defaultdict(dict)
    with (ARTICLE / "workflow-decisions.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["case_id"], row["scenario"], int(row["step"]))
            public[(row["deployment"], int(row["measurement"]))][key] = row["expected_action"]
    if len(public) != len(runs):
        raise ValueError("Published workflow run count differs")
    for arm, number, states in raw_runs:
        expected = {(pseudonym[case], scenario, step): action
                    for (case, scenario, step), (action, _) in states.items()}
        if public[(arm, number)] != expected:
            raise ValueError(f"Pseudonym or action mismatch for {arm} run {number}")

    observations = []
    for (case, scenario, step), (action, state_hash) in sorted(
        baseline.items(), key=lambda item: (pseudonym[item[0][0]],
                                            SCENARIOS.index(item[0][1]), item[0][2])
    ):
        state = baseline_observations[(case, scenario, step)]
        observations.append({
            "case_id": pseudonym[case], "scenario": scenario, "step": step,
            "expected_action": action, "observation_sha256": state_hash,
            "stage": state["stage"], "tool_status": state["tool_status"] or "",
            "attempts_on_stage": state["attempts_on_stage"],
            "calls_remaining": state["calls_remaining"],
            "completed_stages_count": len(state["completed_stages"]),
        })
    state_hashes = []
    for arm, number, states in sorted(raw_runs):
        for (case, scenario, step), (action, state_hash) in sorted(
            states.items(), key=lambda item: (pseudonym[item[0][0]],
                                             SCENARIOS.index(item[0][1]), item[0][2])
        ):
            state_hashes.append({"deployment": arm, "measurement": number,
                                 "case_id": pseudonym[case], "scenario": scenario,
                                 "step": step, "expected_action": action,
                                 "observation_sha256": state_hash})
    write_csv(HERE / "workflow-observations.csv", list(observations[0]), observations)
    write_csv(HERE / "workflow-state-hashes.csv", list(state_hashes[0]), state_hashes)
    print(f"Exported {len(observations)} observation rows and {len(state_hashes)} run-state rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root.resolve())
