"""Gated countTokens preflight for all 1,001 frozen Gemini selector requests.

Every attempted count has a durable intent. An uncertain API outcome stops the
run, and the output namespace cannot be reused. No generation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
from time import monotonic, sleep
from typing import Any

import gemini_count_preflight as smoke
import selector_passes as runner


SCHEMA = "averitec-gemini-count-all/v1"
CODE = (
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/gemini_count_all.py",
    *smoke.CODE,
)


def _hash(value: Any) -> str:
    return hashlib.sha256(runner.canonical(value).encode()).hexdigest()


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("ab") as target:
        target.write((runner.canonical(value) + "\n").encode())
        target.flush()
        os.fsync(target.fileno())


def _empty(path: Path) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)


def _plan(candidates: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observations, requests = smoke._requests(candidates)
    records = []
    for observation, request in zip(observations, requests):
        serialized = runner.wire_bytes(request)
        records.append({"observation_sha256": _hash(observation),
                        "request_sha256": hashlib.sha256(serialized).hexdigest(),
                        "serialized_bytes": len(serialized)})
    if len(records) != 1001:
        raise ValueError("gemini_count_all_plan_count")
    return observations, requests, records


def make_manifest(candidates: Path, smoke_receipt: Path, output: Path) -> dict[str, Any]:
    _observations, _requests, records = _plan(candidates)
    receipt = json.loads(smoke_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != "averitec-gemini-count-smoke-receipt/v1" or receipt.get("complete") is not True:
        raise ValueError("gemini_count_all_smoke_incomplete")
    result = smoke_receipt.parent / "result.json"
    if (runner.digest(result) != receipt.get("result_sha256")
            or type(json.loads(result.read_text()).get("total_tokens")) is not int):
        raise ValueError("gemini_count_all_smoke_result")
    return {
        "schema": SCHEMA, "source_commit": runner._head(),
        "candidates_sha256": runner.digest(candidates),
        "code_sha256": {name: runner.digest(runner.REPO / name) for name in CODE},
        "smoke_receipt_sha256": runner.digest(smoke_receipt),
        "model": smoke.MODEL, "endpoint": smoke.URL,
        "request_plan_sha256": _hash(records), "request_count": len(records),
        "maximum_count_requests": 1001, "maximum_generation_requests": 0,
        "timeout_seconds": 30, "maximum_wall_seconds": 7200,
        "inter_call_delay_seconds": 1.0, "retries": 0,
        "context_limit_tokens": 32768, "output_request_limit_tokens": 2048,
        "input_margin_tokens": 128,
        "output_path": str(output.resolve()),
    }


def check(manifest: Path, candidates: Path, smoke_receipt: Path, output: Path,
          gate: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    policy = json.loads(manifest.read_text(encoding="utf-8"))
    if policy != make_manifest(candidates, smoke_receipt, output):
        raise ValueError("gemini_count_all_manifest_drift")
    if gate is not None and json.loads(gate.read_text(encoding="utf-8")) != {
        "schema": "averitec-astra-launch-gate/v1",
        "manifest_sha256": runner.digest(manifest),
        "decision": "approved", "reviewer": "gpt-6-astra",
    }:
        raise ValueError("gemini_count_all_astra_gate")
    _observations, requests, records = _plan(candidates)
    return policy, requests, records


def execute(manifest: Path, candidates: Path, smoke_receipt: Path, output: Path, gate: Path,
            *, key: str) -> dict[str, Any]:
    policy, requests, records = check(manifest, candidates, smoke_receipt, output, gate)
    if output.exists() or not key or not output.parent.is_dir():
        raise ValueError("gemini_count_all_output_or_credential")
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError("gemini_count_all_alarm_unavailable")
    output.mkdir(mode=0o700)
    intent_path, result_path = output / "intents.jsonl", output / "results.jsonl"
    _empty(intent_path)
    _empty(result_path)
    smoke._write_once(output / "metadata.json", {"manifest_sha256": runner.digest(manifest),
                                                "smoke_receipt_sha256": policy["smoke_receipt_sha256"],
                                                "request_count": len(requests)})
    previous_handler = signal.getsignal(signal.SIGALRM)
    def deadline(_signum: int, _frame: Any) -> None:
        raise TimeoutError("gemini_count_all_wall_deadline")
    begun = monotonic()
    stop_reason = None
    counts: list[int] = []
    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, policy["maximum_wall_seconds"])
        for index, (request, item) in enumerate(zip(requests, records), 1):
            if monotonic() - begun >= policy["maximum_wall_seconds"]:
                stop_reason = "wall_cap_before_call"
                break
            _append(intent_path, {"ordinal": index, **item})
            try:
                counted = smoke._count(request, key, policy["timeout_seconds"])
                outcome = {"ordinal": index, "status": "ok", "total_tokens": counted}
            except BaseException as error:
                outcome = {"ordinal": index, "status": "unknown_submission_outcome",
                           "error_type": type(error).__name__}
                stop_reason = "uncertain_count_outcome"
            _append(result_path, outcome)
            if stop_reason:
                break
            counts.append(counted)
            if counted + policy["input_margin_tokens"] + policy["output_request_limit_tokens"] > policy["context_limit_tokens"]:
                stop_reason = "context_cap_exceeded"
                break
            if index < len(requests):
                sleep(policy["inter_call_delay_seconds"])
    except BaseException as error:
        stop_reason = "interrupted_" + type(error).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    complete = stop_reason is None and len(counts) == policy["request_count"]
    if complete:
        report_records = [{**item, "input_tokens": count} for item, count in zip(records, counts)]
        method = {"model": policy["model"], "endpoint": policy["endpoint"],
                  "count_schema": "generateContentRequest", "source_commit": policy["source_commit"],
                  "code_sha256": policy["code_sha256"],
                  "smoke_receipt_sha256": policy["smoke_receipt_sha256"]}
        report = {"schema": "averitec-selector-token-preflight/v2", "arm": "gemini",
                  "model": policy["model"], "profile": runner.ARMS["gemini"][1],
                  "method": "provider_count_tokens", "method_identity_sha256": _hash(method),
                  "context_limit_tokens": policy["context_limit_tokens"],
                  "output_request_limit_tokens": policy["output_request_limit_tokens"],
                  "output_limit_enforced": True, "input_margin_tokens": policy["input_margin_tokens"],
                  "records": report_records}
        smoke._write_once(output / "token-preflight.json", report)
    receipt = {"schema": "averitec-gemini-count-all-receipt/v1",
               "manifest_sha256": runner.digest(manifest), "complete": complete,
               "requests_attempted": sum(1 for _ in intent_path.open("rb")),
               "successful_counts": len(counts), "stop_reason": stop_reason,
               "max_count": max(counts) if counts else None,
               "total_count": sum(counts),
               "metadata_sha256": runner.digest(output / "metadata.json"),
               "intents_sha256": runner.digest(intent_path),
               "results_sha256": runner.digest(result_path),
               "token_preflight_sha256": runner.digest(output / "token-preflight.json") if complete else None}
    smoke._write_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--astra-gate", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--make-manifest", action="store_true")
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.make_manifest:
        smoke._write_once(args.manifest, make_manifest(args.candidates, args.smoke_receipt, args.output))
        print(json.dumps({"manifest_sha256": runner.digest(args.manifest), "count_calls": 0}, sort_keys=True))
    elif args.plan:
        policy, _requests, _records = check(args.manifest, args.candidates, args.smoke_receipt, args.output)
        print(json.dumps({"planned_count_requests": policy["request_count"], "count_calls": 0}, sort_keys=True))
    else:
        if args.astra_gate is None:
            raise ValueError("gemini_count_all_astra_gate_required")
        key = os.environ.get("GEMINI_API_KEY", "")
        print(json.dumps(execute(args.manifest, args.candidates, args.smoke_receipt,
                                 args.output, args.astra_gate, key=key), sort_keys=True))


if __name__ == "__main__":
    main()
