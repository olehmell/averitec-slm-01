#!/usr/bin/env python3
"""Check that every complete workflow run used the same frozen observations.

Requires the original repository, including its private frozen result files:
python frozen_workflow_audit.py --root /path/to/phd --out workflow-observation-audit.json
No model is invoked and no claim or snippet text is written to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

STAGES = {"decompose", "queries", "retrieve", "qa", "coverage", "select", "verdict"}
EXPERIMENT = Path("experiments/orchestration/averitec-controller-reliability-2026")
REGISTRY = EXPERIMENT / "results/workflow-three-full-measurements-20260923.json"
SLOTS = Path(".private-artifacts/averitec-controller-reliability-2026/three-pass-verdict-2026/evaluation-inputs-v1/evaluation-inputs.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation_digest(observation: dict) -> str:
    canonical = json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verified(root: Path, record: dict) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = root / path
    if digest(path) != record["sha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    return path


def source_runs(root: Path) -> list[tuple[str, int, str, dict, dict | None]]:
    registry = json.loads((root / REGISTRY).read_text())
    slots = json.loads((root / SLOTS).read_text())["workflow_slots"]
    runs = []
    for arm in ("jev", "gemini"):
        for number, entry in enumerate(registry["arms"][arm]["complete_measurements"], 1):
            runs.append((arm, number, "trial_results", entry["results"], None))
    for slot in slots:
        if slot["arm"] in ("jev", "gemini") or slot["kind"] not in ("trial_results", "native_ledger"):
            continue
        runs.append((slot["arm"], int(slot["pass"]), slot["kind"], slot["file"], slot.get("plan")))
    if len(runs) != 20:
        raise ValueError(f"Expected 20 complete runs, got {len(runs)}")
    return runs


def read_run(root: Path, kind: str, file: dict, plan: dict | None):
    source = verified(root, file)
    state = {}
    observations = {}
    if kind == "trial_results":
        for line in source.open(encoding="utf-8"):
            trial = json.loads(line)
            for event in trial["events"]:
                key = (trial["case_id"], trial["scenario"], int(event["step"]))
                if key in state:
                    raise ValueError(f"Duplicate checkpoint {key}")
                observation = event["observation"]
                if observation_digest(observation) != event["observation_sha256"]:
                    raise ValueError(f"Observation hash mismatch at {key}")
                state[key] = (event["expected_action"], event["observation_sha256"])
                observations[key] = observation
    elif kind == "native_ledger":
        if plan is None:
            raise ValueError("Native ledger requires a verified plan")
        document = json.loads(verified(root, plan).read_text())
        trials = {trial["trial_id"]: trial for trial in document["trials"]}
        planned_ids = set()
        for point in document["checkpoints"]:
            trial = trials[point["trial_id"]]
            key = (trial["case_id"], trial["scenario"], int(point["step"]))
            if key in state or point["checkpoint_id"] in planned_ids:
                raise ValueError(f"Duplicate native checkpoint {key}")
            state[key] = (point["expected_action"], point["observation_sha256"])
            planned_ids.add(point["checkpoint_id"])
        measured_ids = []
        for line in source.open(encoding="utf-8"):
            event = json.loads(line)
            if event["kind"] == "response":
                measured_ids.append(event["checkpoint_id"])
        if len(measured_ids) != len(planned_ids) or set(measured_ids) != planned_ids:
            raise ValueError("Native responses do not cover the complete plan")
    else:
        raise ValueError(kind)
    if len(state) != 2299:
        raise ValueError(f"Expected 2,299 checkpoints, got {len(state)}")
    return state, observations


def accounting(states: dict, observations: dict) -> dict:
    by_scenario = defaultdict(Counter)
    for (case, scenario, step), (action, _) in states.items():
        observation = observations[(case, scenario, step)]
        counts = by_scenario[scenario]
        counts["checkpoints"] += 1
        if action in STAGES:
            counts["tool_dispatches"] += 1
            if observation["attempts_on_stage"] == 1:
                counts[f"retry_after_{observation['tool_status']}"] += 1
        elif action == "finish":
            counts["finish"] += 1
        elif action == "abort":
            counts["abort"] += 1
            remaining = 7 - len(observation["completed_stages"]) + 1
            if observation["calls_remaining"] < remaining:
                counts["abort_budget_exhaustion"] += 1
            if observation["attempts_on_stage"] >= 2:
                counts[f"abort_second_{observation['tool_status']}"] += 1
        else:
            raise ValueError(action)
    totals = sum(by_scenario.values(), Counter())
    expected = {"checkpoints": 2299, "tool_dispatches": 1999, "finish": 104,
                "abort": 196, "retry_after_timeout": 203, "retry_after_invalid": 96,
                "abort_second_timeout": 100, "abort_second_invalid": 96}
    if any(totals[key] != value for key, value in expected.items()) or totals["abort_budget_exhaustion"]:
        raise ValueError(f"Unexpected workflow accounting: {dict(totals)}")
    return {"per_scenario": {name: dict(counts) for name, counts in sorted(by_scenario.items())},
            "per_run": dict(totals), "other_tool_dispatches": totals["tool_dispatches"] - 299}


def run(root: Path, out: Path) -> None:
    baseline = None
    reference_observations = None
    report = []
    for arm, number, kind, file, plan in source_runs(root):
        states, observations = read_run(root, kind, file, plan)
        if baseline is None:
            baseline, reference_observations = states, observations
        if states != baseline:
            raise ValueError(f"Frozen states differ for {arm} run {number}")
        report.append({"deployment": arm, "measurement": number, "kind": kind,
                       "checkpoints": len(states), "same_expected_actions_and_observation_hashes": True,
                       "source_sha256": file["sha256"],
                       "plan_sha256": plan["sha256"] if plan else None})
    result = {"schema": "csit-frozen-workflow-audit/v1", "complete_runs": len(report),
              "reference": {"deployment": report[0]["deployment"], "measurement": report[0]["measurement"]},
              "state_identity": "same case, scenario, step, expected action and observation SHA-256 in every run",
              "accounting": accounting(baseline, reference_observations), "runs": report,
              "limits": "Observation hashes verify replay identity. The report does not claim that each timeout was injected rather than returned by a frozen tool."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {len(report)} runs share 2,299 frozen states; 203 timeout and 96 invalid retries; 196 second-failure aborts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Original repository root with private frozen results")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.out)
