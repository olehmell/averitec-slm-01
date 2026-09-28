#!/usr/bin/env python3
"""Bounded, append-only Jeff workflow extension; no outcome resume or retry."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from importlib import metadata
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

EXT = Path(__file__).resolve().parent
EXP = EXT.parents[1]
sys.path.insert(0, str(EXP))

import yaml
from data import ROOT, check_freeze, code_binding, file_hash, verify_preparation_traces, write_new
from engine import ACTIONS, INSTRUCTIONS, FaultTools, OracleController, ReplayTools, digest, replay_checkpoints, run_episode
from run import action_order, _append
from tracing import TraceRecorder
from warmup import WARMUP_OBSERVATION, premeasure_warmup
from adapter import (MODEL, MODEL_REVISION, PROBABILITY_SUM_TOLERANCE, RETURNED_MODEL,
                     NativeJeffController)


class BoundedController:
    """Stop the phase after any infrastructure failure or wall-budget breach."""

    def __init__(self, delegate, deadline):
        self.delegate, self.deadline, self.model = delegate, deadline, delegate.model

    def choose(self, observation, instructions, actions):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("jeff_workflow_phase_wall_budget")
        result = self.delegate.choose(observation, instructions, actions)
        if result.outcome not in ("ok", "invalid_output"):
            raise RuntimeError("jeff_workflow_infrastructure_failure")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("jeff_workflow_phase_wall_budget")
        return result


def settings():
    return yaml.safe_load((EXT / "config.yaml").read_text())


def extension_binding():
    paths = [path for path in EXT.rglob("*") if path.is_file()
             and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts
             and path.suffix in (".py", ".json", ".yaml", ".txt", ".sh", ".md")]
    return {str(path.relative_to(EXT)): file_hash(path) for path in sorted(paths)}


def require_committed():
    for relative in extension_binding():
        path = EXT / relative
        result = subprocess.run(["git", "show", "HEAD:" + str(path.relative_to(ROOT))],
                                cwd=ROOT, capture_output=True, check=False)
        if result.returncode or result.stdout != path.read_bytes():
            raise ValueError("jeff_workflow_source_must_be_committed:" + relative)


def validate_asset_manifest(path, cfg):
    if file_hash(path) != cfg["asset_manifest_sha256"]:
        raise ValueError("jeff_workflow_asset_manifest_hash")
    value = json.loads(path.read_text())
    if (value.get("schema") != "averitec-jeff-native-assets/v1"
            or value.get("model") != MODEL
            or value.get("model_revision") != MODEL_REVISION
            or value.get("checkpoint_sha256") != cfg["checkpoint_sha256"]):
        raise ValueError("jeff_workflow_asset_identity")
    return value


def validate_freeze(freeze, path, phase, cfg):
    check_freeze(freeze, freeze["selection"])
    verify_preparation_traces(freeze, path.parent)
    if freeze["selection_sha256"] != cfg["selection_sha256"]:
        raise ValueError("jeff_workflow_selection_mismatch")
    role = "evaluation" if phase == "evaluation" else "development"
    expected = ([row["case_id"] for row in freeze["selection"]["cases"] if row["role"] == role]
                if phase == "evaluation" else cfg["development_case_ids"])
    if ([row["case_id"] for row in freeze["cases"]] != expected
            or len(freeze["cases"]) != cfg["phases"][phase]["cases"]
            or any(row["role"] != role for row in freeze["cases"])):
        raise ValueError("jeff_workflow_freeze_case_set_mismatch")


def plan(freeze, phase, cfg):
    rows = []
    policy = cfg["phases"][phase]
    for case in freeze["cases"]:
        for scenario in cfg["scenarios"]:
            for order in policy["orders"]:
                repeats = policy["canonical_repetitions"] if order == "canonical" else 1
                for repetition in range(repeats):
                    row = dict(controller="jeff", model=MODEL, case_id=case["case_id"],
                               group_id=case["group_id"], role=case["role"], scenario=scenario,
                               order=order, repetition=repetition, phase=phase, mode="checkpoints",
                               prompt_variant="jeff_native_workflow_v1", tools_mode="frozen")
                    row["trial_id"] = digest(row)
                    rows.append(row)
    return rows


def reference(case, scenario):
    return run_episode(OracleController(), FaultTools(ReplayTools(case["records"]), scenario))


def runtime_identity():
    versions = {}
    for name in ("torch", "transformers", "gliformer", "gliner", "scipy", "numpy"):
        versions[name] = metadata.version(name)
    return {"distributions": versions, "python": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "image_sha256": os.environ.get("JEFF_WORKFLOW_IMAGE_SHA256")}


def score(rows):
    events = [event for row in rows for event in row["events"]]
    if not events:
        raise ValueError("jeff_workflow_empty_results")
    compliant = sum(event["compliant"] for event in events)
    return {"decisions": len(events), "compliant": compliant,
            "decision_compliance": compliant / len(events),
            "valid_action_rate": sum(event["provider_outcome"] == "ok" for event in events) / len(events),
            "median_native_inference_latency_ms": statistics.median(event["latency_ms"] for event in events)}


def evidence_files(output):
    return {str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "receipt.json"}


def audit_condition(output, freeze, phase, cfg):
    meta = json.loads((output / "metadata.json").read_text())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    tasks = meta["planned_tasks"]
    if tasks != plan(freeze, phase, cfg) or meta["freeze_sha256"] != digest(freeze):
        raise ValueError("jeff_workflow_plan_mismatch")
    if [row["trial_id"] for row in rows] != [task["trial_id"] for task in tasks]:
        raise ValueError("jeff_workflow_trial_completeness")
    intents = [json.loads(line) for line in (output / "intents.jsonl").read_text().splitlines()]
    if intents != tasks:
        raise ValueError("jeff_workflow_intent_completeness")
    traces = [json.loads(line) for line in (output / "traces.jsonl").read_text().splitlines()]
    starts = [row["span_id"] for row in traces if row["event"] == "span_start"]
    ends = [row["span_id"] for row in traces if row["event"] == "span_end"]
    if len(set(starts)) != len(starts) or sorted(starts) != sorted(ends):
        raise ValueError("jeff_workflow_trace_completeness")
    roots = [row for row in traces if row["event"] == "span_start" and row.get("parent_span_id") is None]
    if [row["input"] for row in roots] != tasks or any(row["name"] != "controller.trial" for row in roots):
        raise ValueError("jeff_workflow_trace_trial_mismatch")
    decisions = [row for row in traces if row["event"] == "span_start" and row.get("name") == "controller.decision"]
    requests = [row for row in traces if row["event"] == "span_start" and row.get("name") == "controller.request"]
    if len(decisions) != sum(len(row["events"]) for row in rows) or len(requests) != len(decisions):
        raise ValueError("jeff_workflow_trace_decision_count")
    by_id = {case["case_id"]: case for case in freeze["cases"]}
    for row, task in zip(rows, tasks):
        if row["row_sha256"] != digest({key: value for key, value in row.items() if key != "row_sha256"}):
            raise ValueError("jeff_workflow_row_digest")
        if any(row[key] != value for key, value in task.items()) or row["freeze_sha256"] != meta["freeze_sha256"]:
            raise ValueError("jeff_workflow_row_identity")
        if len(row["events"]) != row["planned_decisions"] or not row["events"]:
            raise ValueError("jeff_workflow_decision_completeness")
        expected_events = reference(by_id[task["case_id"]], task["scenario"])["events"]
        if len(expected_events) != row["planned_decisions"]:
            raise ValueError("jeff_workflow_reference_length")
        for event, expected in zip(row["events"], expected_events):
            if any(event[key] != expected[key] for key in
                   ("step", "observation", "observation_sha256", "expected_action")):
                raise ValueError("jeff_workflow_reference_mismatch")
            if event["provider_outcome"] not in ("ok", "invalid_output"):
                raise ValueError("jeff_workflow_infrastructure_failure")
            if event["model"] != MODEL:
                raise ValueError("jeff_workflow_model_identity")
            if (event["provider_outcome"] == "ok"
                    and event["returned_model"] != RETURNED_MODEL):
                raise ValueError("jeff_workflow_returned_model_identity")
            if (event["provider_outcome"] == "invalid_output"
                    and event["returned_model"] not in (None, RETURNED_MODEL)):
                raise ValueError("jeff_workflow_invalid_returned_model_identity")
            if event["compliant"] != (event["provider_outcome"] == "ok"
                                       and event["action"] == event["expected_action"]):
                raise ValueError("jeff_workflow_compliance_mismatch")
    return rows


def verify_receipt(path, phase, cfg):
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "jeff-workflow-extension-receipt/v1"
            or receipt.get("phase") != phase or receipt.get("status") != "complete"
            or receipt.get("extension_binding") != extension_binding()
            or receipt.get("core_binding") != code_binding()
            or receipt.get("selection_sha256") != cfg["selection_sha256"]):
        raise ValueError("jeff_workflow_receipt_identity")
    if receipt.get("evidence") != evidence_files(path.parent):
        raise ValueError("jeff_workflow_receipt_evidence_drift")
    if receipt.get("receipt_sha256") != digest({key: value for key, value in receipt.items()
                                                if key != "receipt_sha256"}):
        raise ValueError("jeff_workflow_receipt_digest")
    freeze_path = path.parent / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    validate_freeze(freeze, freeze_path, phase, cfg)
    if receipt["freeze_sha256"] != digest(freeze):
        raise ValueError("jeff_workflow_receipt_freeze_mismatch")
    rows = audit_condition(path.parent / "jeff", freeze, phase, cfg)
    if receipt["condition"]["status"] != "complete" or receipt["condition"]["score"] != score(rows):
        raise ValueError("jeff_workflow_receipt_score_mismatch")
    return receipt


def run_condition(output, tasks, freeze, checkpoint, asset_manifest, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("jeff_workflow_phase_wall_budget")
    output.mkdir()
    controller = NativeJeffController(checkpoint, device="cuda")
    by_id = {row["case_id"]: row for row in freeze["cases"]}
    references = {(task["case_id"], task["scenario"]):
                  reference(by_id[task["case_id"]], task["scenario"]) for task in tasks}
    try:
        preflights, seen = [], set()
        for task in tasks:
            actions = action_order(task["order"], settings()["seed"])
            for event in references[task["case_id"], task["scenario"]]["events"]:
                key = digest([event["observation"], actions])
                if key not in seen:
                    preflights.append({"input_sha256": key,
                                       **controller.preflight(event["observation"], actions)})
                    seen.add(key)
        preflights.append({"warmup": True,
                           **controller.preflight(WARMUP_OBSERVATION, list(ACTIONS))})
        write_new(output / "preflight.json", {"checks": preflights})
        meta = {"planned_tasks": tasks, "freeze_sha256": digest(freeze),
                "identity": controller.identity(), "asset_manifest": asset_manifest,
                "runtime": runtime_identity(),
                "latency_boundary": "synchronized_native_forward_no_network"}
        write_new(output / "metadata.json", meta)
        run_id = digest(meta)
        warm_tracer = TraceRecorder(output / "warmup.traces.jsonl", run_id=run_id, trial_id="warmup")
        with warm_tracer.span("controller.warmup", input={"controller": "jeff"}) as span:
            warmup = premeasure_warmup(controller, instructions=INSTRUCTIONS,
                                       actions=list(ACTIONS), tracer=warm_tracer)
            span.update(output=warmup)
        write_new(output / "warmup.json", warmup)
        bounded = BoundedController(controller, deadline)
        with (output / "intents.jsonl").open("x") as intents, (output / "results.jsonl").open("x") as results:
            for task in tasks:
                if time.monotonic() >= deadline:
                    raise TimeoutError("jeff_workflow_phase_wall_budget")
                _append(intents, task)
                tracer = TraceRecorder(output / "traces.jsonl", run_id=run_id,
                                       trial_id=task["trial_id"], metadata=task)
                controller.tracer = tracer
                ref = references[task["case_id"], task["scenario"]]
                with tracer.span("controller.trial", input=task) as span:
                    row = replay_checkpoints(bounded, ref,
                                             actions=action_order(task["order"], settings()["seed"]),
                                             tracer=tracer)
                    span.update(output=row)
                row.update(task)
                row.update(freeze_sha256=digest(freeze), planned_decisions=len(ref["events"]),
                           preparation_trace_id=by_id[task["case_id"]]["preparation_trace_id"])
                row["row_sha256"] = digest(row)
                _append(results, row)
        rows = audit_condition(output, freeze, tasks[0]["phase"], settings())
        return {"status": "complete", "score": score(rows), "trials": len(rows)}
    finally:
        controller.close()


def parse_native_response(payload, ordered_actions):
    """Validate a persisted Jeff response without executing the model again."""
    if not isinstance(payload, dict) or payload.get("model") != RETURNED_MODEL:
        raise ValueError("jeff_workflow_recovery_model_mismatch")
    answers = payload.get("answers")
    answer = answers.get("next_action") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("jeff_workflow_recovery_answer_invalid")
    probabilities = answer.get("probabilities")
    choice = answer.get("choice")
    confidence = answer.get("confidence")
    number = lambda value: (type(value) in (int, float) and math.isfinite(value)
                            and 0 <= value <= 1)
    if (not isinstance(probabilities, dict)
            or set(probabilities) != set(ordered_actions)
            or not all(number(value) for value in probabilities.values())
            or abs(sum(probabilities.values()) - 1) > PROBABILITY_SUM_TOLERANCE
            or not isinstance(choice, str) or choice not in ordered_actions
            or probabilities[choice] != max(probabilities.values())
            or not number(confidence)):
        raise ValueError("jeff_workflow_recovery_choice_invalid")
    usage = payload.get("usage")
    if (not isinstance(usage, dict) or type(usage.get("input_tokens")) is not int
            or usage["input_tokens"] < 0):
        raise ValueError("jeff_workflow_recovery_usage_invalid")
    return {"action": choice, "probabilities": dict(probabilities),
            "confidence": confidence, "input_tokens": usage["input_tokens"]}


def recover_qualification(destination, source, freeze, cfg, asset_manifest, asset_manifest_path):
    """Re-audit completed qualification responses after fixing local validation."""
    if destination.exists():
        raise FileExistsError(destination)
    required_root = ("plan.json", "freeze.json", freeze["preparation_trace_journal"])
    required_condition = ("metadata.json", "preflight.json", "warmup.json",
                          "warmup.traces.jsonl", "intents.jsonl", "results.jsonl",
                          "traces.jsonl")
    for relative in required_root:
        if not (source / relative).is_file():
            raise ValueError("jeff_workflow_recovery_source_incomplete:" + relative)
    for relative in required_condition:
        if not (source / "jeff" / relative).is_file():
            raise ValueError("jeff_workflow_recovery_source_incomplete:jeff/" + relative)
    if (source / "receipt.json").exists():
        raise ValueError("jeff_workflow_recovery_source_already_complete")
    source_freeze = json.loads((source / "freeze.json").read_text())
    if source_freeze != freeze:
        raise ValueError("jeff_workflow_recovery_freeze_mismatch")
    expected_plan = {"phase": "qualification",
                     "condition": plan(freeze, "qualification", cfg),
                     "freeze_sha256": digest(freeze),
                     "gpu_wall_seconds": cfg["phases"]["qualification"]["gpu_wall_seconds"]}
    if json.loads((source / "plan.json").read_text()) != expected_plan:
        raise ValueError("jeff_workflow_recovery_plan_mismatch")

    source_rows = [json.loads(line) for line in
                   (source / "jeff" / "results.jsonl").read_text().splitlines()]
    traces = [json.loads(line) for line in
              (source / "jeff" / "traces.jsonl").read_text().splitlines()]
    request_starts = [row for row in traces
                      if row["event"] == "span_start" and row.get("name") == "controller.request"]
    request_ends = {row["span_id"]: row for row in traces if row["event"] == "span_end"}
    event_count = sum(len(row["events"]) for row in source_rows)
    if len(request_starts) != event_count:
        raise ValueError("jeff_workflow_recovery_request_count")

    recovered_rows = deepcopy(source_rows)
    recovered_count = 0
    request_index = 0
    for row in recovered_rows:
        for event in row["events"]:
            start = request_starts[request_index]
            request_index += 1
            end = request_ends.get(start["span_id"])
            if end is None or not isinstance(end.get("output"), dict):
                raise ValueError("jeff_workflow_recovery_response_missing")
            criteria = start.get("input", {}).get("questions", {}).get(
                "next_action", {}).get("criteria")
            if not isinstance(criteria, dict) or set(criteria) != set(ACTIONS):
                raise ValueError("jeff_workflow_recovery_request_invalid")
            parsed = parse_native_response(end["output"], list(criteria))
            if event["provider_outcome"] != "ok":
                recovered_count += 1
            event.update(parsed)
            event.update(provider_outcome="ok", error_code=None, output_tokens=None,
                         model=MODEL, returned_model=RETURNED_MODEL)
            event["compliant"] = event["action"] == event["expected_action"]
        row["row_sha256"] = digest({key: value for key, value in row.items()
                                    if key != "row_sha256"})

    destination.mkdir(parents=True)
    for relative in required_root:
        shutil.copyfile(source / relative, destination / relative)
    condition_output = destination / "jeff"
    condition_output.mkdir()
    for relative in required_condition:
        if relative != "results.jsonl":
            shutil.copyfile(source / "jeff" / relative, condition_output / relative)
    with (condition_output / "results.jsonl").open("x") as handle:
        for row in recovered_rows:
            _append(handle, row)
    source_evidence = evidence_files(source)
    recovery = {
        "schema": "jeff-workflow-qualification-recovery/v1",
        "source": str(source),
        "source_job_id": "45542750",
        "execution_source_commit": cfg["qualification_execution_commit"],
        "reason": "nine four-decimal probabilities require a 0.0005 sum tolerance",
        "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
        "responses": event_count,
        "recovered_false_invalid_outputs": recovered_count,
        "model_calls": 0,
        "source_evidence": source_evidence,
    }
    write_new(destination / "recovery.json", recovery)
    rows = audit_condition(condition_output, freeze, "qualification", cfg)
    condition = {"status": "complete", "score": score(rows), "trials": len(rows)}
    receipt = {"schema": "jeff-workflow-extension-receipt/v1", "status": "complete",
               "phase": "qualification", "created_utc": datetime.now(timezone.utc).isoformat(),
               "freeze_sha256": digest(freeze), "selection_sha256": cfg["selection_sha256"],
               "extension_binding": extension_binding(), "core_binding": code_binding(),
               "asset_manifest_sha256": file_hash(asset_manifest_path),
               "condition": condition, "qualification_receipt_sha256": None,
               "recovery": recovery, "evidence": evidence_files(destination)}
    receipt["receipt_sha256"] = digest(receipt)
    write_new(destination / "receipt.json", receipt)
    verify_receipt(destination / "receipt.json", "qualification", cfg)
    return condition


def execute(args):
    cfg = settings()
    freeze = json.loads(args.freeze.read_text())
    validate_freeze(freeze, args.freeze, args.phase, cfg)
    qualification = None
    if args.phase == "evaluation":
        if args.qualification_receipt is None:
            raise ValueError("jeff_workflow_qualification_receipt_required")
        qualification = verify_receipt(args.qualification_receipt, "qualification", cfg)
    tasks = plan(freeze, args.phase, cfg)
    planned = {"phase": args.phase, "condition": tasks, "freeze_sha256": digest(freeze),
               "gpu_wall_seconds": cfg["phases"][args.phase]["gpu_wall_seconds"]}
    if args.plan:
        print(json.dumps(planned, indent=2))
        return
    require_committed()
    asset_manifest = validate_asset_manifest(args.asset_manifest, cfg)
    if args.recover_qualification_from is not None:
        if args.phase != "qualification":
            raise ValueError("jeff_workflow_recovery_qualification_only")
        condition = recover_qualification(args.output, args.recover_qualification_from,
                                          freeze, cfg, asset_manifest,
                                          args.asset_manifest)
        print(json.dumps({"phase": args.phase, "condition": condition,
                          "recovered_from": str(args.recover_qualification_from)}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "plan.json", planned)
    write_new(args.output / "freeze.json", freeze)
    shutil.copyfile(args.freeze.parent / freeze["preparation_trace_journal"],
                    args.output / freeze["preparation_trace_journal"])
    deadline = time.monotonic() + planned["gpu_wall_seconds"]
    condition = run_condition(args.output / "jeff", tasks, freeze, args.checkpoint,
                              asset_manifest, deadline)
    receipt = {"schema": "jeff-workflow-extension-receipt/v1", "status": "complete",
               "phase": args.phase, "created_utc": datetime.now(timezone.utc).isoformat(),
               "freeze_sha256": digest(freeze), "selection_sha256": cfg["selection_sha256"],
               "extension_binding": extension_binding(), "core_binding": code_binding(),
               "asset_manifest_sha256": file_hash(args.asset_manifest), "condition": condition,
               "qualification_receipt_sha256": file_hash(args.qualification_receipt)
               if qualification else None, "evidence": evidence_files(args.output)}
    receipt["receipt_sha256"] = digest(receipt)
    write_new(args.output / "receipt.json", receipt)
    print(json.dumps({"phase": args.phase, "condition": condition}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("qualification", "evaluation"))
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--recover-qualification-from", type=Path)
    parser.add_argument("--plan", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
