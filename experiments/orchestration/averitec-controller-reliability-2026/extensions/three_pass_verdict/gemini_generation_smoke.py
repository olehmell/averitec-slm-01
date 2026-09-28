"""One gated Gemini selector generation using the previously counted request.

The unchanged v2 selector adapter builds the request. The transport boundary
checks its exact hash before a single no-redirect HTTP dispatch.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
from time import monotonic
from typing import Any
from urllib.request import HTTPRedirectHandler, Request, build_opener

import gemini_count_preflight as counter
import selector_passes as runner
from tracing import TraceRecorder


SCHEMA = "averitec-gemini-selector-generation-smoke/v1"
CODE = (
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/gemini_generation_smoke.py",
    *counter.CODE,
    "experiments/orchestration/averitec-controller-reliability-2026/tracing.py",
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _write_once(path: Path, value: dict[str, Any]) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write((runner.canonical(value) + "\n").encode())
        target.flush()
        os.fsync(target.fileno())


def _count_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    result_path = path.parent / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "averitec-gemini-count-smoke-receipt/v1"
            or receipt.get("complete") is not True or receipt.get("requests_attempted") != 1
            or receipt.get("result_sha256") != runner.digest(result_path)
            or result.get("status") != "ok"
            or type(result.get("total_tokens")) is not int or result["total_tokens"] < 1):
        raise ValueError("gemini_generation_count_smoke_invalid")
    return receipt, result


def make_manifest(candidates: Path, count_receipt: Path, output: Path) -> dict[str, Any]:
    receipt, result = _count_receipt(count_receipt)
    ordinal, observation, body, size, request_hash = counter._selected(candidates)
    smoke_manifest = count_receipt.parent.parent.parent / "inputs" / "gemini-count-smoke-manifest.json"
    count_policy = json.loads(smoke_manifest.read_text(encoding="utf-8"))
    if (receipt["manifest_sha256"] != runner.digest(smoke_manifest)
            or count_policy.get("selected_ordinal") != ordinal
            or count_policy.get("generate_request_sha256") != request_hash
            or count_policy.get("generate_request_bytes") != size):
        raise ValueError("gemini_generation_count_request_drift")
    if body.get("generationConfig", {}).get("maxOutputTokens") != 2048:
        raise ValueError("gemini_generation_output_limit_drift")
    planned = runner.previous.plan_observations(candidates)
    row = planned[ordinal - 2] if ordinal > 1 else None
    if row is None or row["observation"] != observation:
        raise ValueError("gemini_generation_selected_warmup")
    return {"schema": SCHEMA, "source_commit": runner._head(),
            "candidates_sha256": runner.digest(candidates),
            "code_sha256": {name: runner.digest(runner.REPO / name) for name in CODE},
            "count_receipt_sha256": runner.digest(count_receipt),
            "count_manifest_sha256": runner.digest(smoke_manifest),
            "model": counter.MODEL, "profile": runner.ARMS["gemini"][1],
            "returned_model": counter.MODEL,
            "selected_ordinal": ordinal, "case_id": row["case_id"], "candidate_id": row["id"],
            "observation_sha256": hashlib.sha256(runner.canonical(observation).encode()).hexdigest(),
            "request_sha256": request_hash, "request_bytes": size,
            "count_tokens": result["total_tokens"], "input_margin_tokens": 128,
            "output_limit_tokens": 2048, "context_limit_tokens": 32768,
            "maximum_generation_requests": 1, "request_timeout_seconds": 120,
            "maximum_wall_seconds": 150, "retries": 0,
            "output_path": str(output.resolve())}


def check(manifest: Path, candidates: Path, count_receipt: Path, output: Path,
          gate: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = json.loads(manifest.read_text(encoding="utf-8"))
    if policy != make_manifest(candidates, count_receipt, output):
        raise ValueError("gemini_generation_manifest_drift")
    if gate is not None and json.loads(gate.read_text(encoding="utf-8")) != {
        "schema": "averitec-astra-launch-gate/v1", "manifest_sha256": runner.digest(manifest),
        "decision": "approved", "reviewer": "gpt-6-astra",
    }:
        raise ValueError("gemini_generation_astra_gate")
    return policy, counter._selected(candidates)[1]


@contextmanager
def _single_request(policy: dict[str, Any]):
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError("gemini_generation_alarm_unavailable")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_post = runner.recovery_module._post
    dispatched = 0

    def dispatch(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        nonlocal dispatched
        dispatched += 1
        if dispatched != 1 or url != counter.URL.replace(":countTokens", ":generateContent"):
            raise ValueError("gemini_generation_endpoint_or_call_drift")
        if hashlib.sha256(runner.wire_bytes(body)).hexdigest() != policy["request_sha256"]:
            raise ValueError("gemini_generation_request_drift")
        request = Request(url, data=runner.wire_bytes(body), method="POST",
                          headers={"Content-Type": "application/json", **headers})
        with build_opener(_NoRedirect()).open(request, timeout=min(timeout, policy["request_timeout_seconds"])) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("gemini_generation_response_not_object")
        return decoded

    def deadline(_signum: int, _frame: Any) -> None:
        raise TimeoutError("gemini_generation_wall_deadline")

    try:
        runner.recovery_module._post = dispatch
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, policy["maximum_wall_seconds"])
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        runner.recovery_module._post = previous_post


def execute(manifest: Path, candidates: Path, count_receipt: Path, output: Path,
            gate: Path, *, key: str) -> dict[str, Any]:
    policy, observation = check(manifest, candidates, count_receipt, output, gate)
    if output.exists() or not key:
        raise ValueError("gemini_generation_output_or_credential")
    selector = runner._make_selector("gemini", endpoint=None, laya_checkpoint=None, keys={"gemini": key})
    runner._local_identity("gemini", selector, policy["model"], policy["returned_model"], policy["profile"])
    output.mkdir(parents=True, mode=0o700)
    trace = output / "traces.jsonl"
    trace.touch(exist_ok=False)
    _write_once(output / "intent.json", {"manifest_sha256": runner.digest(manifest),
                                         "case_id": policy["case_id"], "candidate_id": policy["candidate_id"],
                                         "request_sha256": policy["request_sha256"]})
    recorder = TraceRecorder(trace, "gemini-selector-generation-smoke", "longest-development-request", {"phase": "smoke"})
    runner.previous._set_tracer(selector, recorder)
    instructions = (runner.EVIDENCE / "selector_instructions.txt").read_text(encoding="utf-8")
    started = monotonic()
    try:
        with _single_request(policy):
            result = selector.choose(observation, instructions, list(runner.BINARY_ACTIONS))
        elapsed = monotonic() - started
        trace_complete, trace_model = runner._trace_response(trace, recorder.trace_id, "gemini")
        usage = {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
        status = ("ok" if result.outcome == "ok" and result.action in runner.BINARY_ACTIONS
                  and result.model == policy["model"] and result.returned_model == policy["returned_model"]
                  and trace_complete and trace_model == policy["returned_model"]
                  and type(usage["input_tokens"]) is int
                  and 0 < usage["input_tokens"] <= policy["count_tokens"] + policy["input_margin_tokens"]
                  and type(usage["output_tokens"]) is int
                  and 0 <= usage["output_tokens"] <= policy["output_limit_tokens"]
                  and elapsed <= policy["maximum_wall_seconds"] else
                  "unknown_submission_outcome" if result.outcome == "transport_error" else "qualification_failed")
        measured = {"status": status, "outcome": result.outcome, "action": result.action,
                    "model": result.model, "returned_model": result.returned_model,
                    "trace_returned_model": trace_model, "trace_complete": trace_complete,
                    "usage": usage, "count_tokens": policy["count_tokens"],
                    "elapsed_seconds": elapsed, "error_code": result.error_code}
    except BaseException as error:
        measured = {"status": "unknown_submission_outcome", "error_type": type(error).__name__,
                    "usage": None, "elapsed_seconds": monotonic() - started}
    _write_once(output / "result.json", measured)
    receipt = {"schema": "averitec-gemini-selector-generation-smoke-receipt/v1",
               "manifest_sha256": runner.digest(manifest), "requests_attempted": 1,
               "complete": measured["status"] == "ok",
               "intent_sha256": runner.digest(output / "intent.json"),
               "result_sha256": runner.digest(output / "result.json"),
               "trace_sha256": runner.digest(trace)}
    _write_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--count-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--astra-gate", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--make-manifest", action="store_true")
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.make_manifest:
        _write_once(args.manifest, make_manifest(args.candidates, args.count_receipt, args.output))
        print(json.dumps({"manifest_sha256": runner.digest(args.manifest), "generation_calls": 0}, sort_keys=True))
    elif args.plan:
        policy, _ = check(args.manifest, args.candidates, args.count_receipt, args.output)
        print(json.dumps({"ordinal": policy["selected_ordinal"], "count_tokens": policy["count_tokens"],
                          "generation_calls": 0}, sort_keys=True))
    else:
        if args.astra_gate is None:
            raise ValueError("gemini_generation_astra_gate_required")
        key = os.environ.get("GEMINI_API_KEY", "")
        print(json.dumps(execute(args.manifest, args.candidates, args.count_receipt,
                                 args.output, args.astra_gate, key=key), sort_keys=True))


if __name__ == "__main__":
    main()
