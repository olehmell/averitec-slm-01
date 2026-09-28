"""One gated Gemini countTokens smoke on the largest frozen selector request.

The count request contains the exact GenerateContent body from the unchanged
selector adapter. It does not generate an action or spend a generation call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
from time import monotonic
from typing import Any
from urllib.request import HTTPRedirectHandler, Request, build_opener

import selector_passes as runner


SCHEMA = "averitec-gemini-count-smoke/v1"
MODEL = "gemini-3.1-flash-lite"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:countTokens"
CODE = (
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/gemini_count_preflight.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/selector_passes.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/run_selector.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/recovery_runtime.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_runtime.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_instructions.txt",
)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write((runner.canonical(value) + "\n").encode("utf-8"))
        target.flush()
        os.fsync(target.fileno())


def _requests(candidates: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    planned = runner.previous.plan_observations(candidates)
    if len(planned) != 1000:
        raise ValueError("gemini_count_candidate_count")
    observations = [runner.WARMUP, *(row["observation"] for row in planned)]
    instructions = (runner.EVIDENCE / "selector_instructions.txt").read_text(encoding="utf-8")
    selector = runner._make_selector("gemini", endpoint=None, laya_checkpoint=None, keys={"gemini": "offline-capture"})
    captured = runner._capture_http_requests("gemini", selector, observations, instructions)
    if len(captured) != 1001:
        raise ValueError("gemini_count_request_count")
    return observations, captured


def _selected(candidates: Path) -> tuple[int, dict[str, Any], dict[str, Any], int, str]:
    observations, requests = _requests(candidates)
    index = max(range(len(requests)), key=lambda number: len(runner.wire_bytes(requests[number])))
    body = requests[index]
    serialized = runner.wire_bytes(body)
    return index + 1, observations[index], body, len(serialized), hashlib.sha256(serialized).hexdigest()


def make_manifest(candidates: Path, output: Path) -> dict[str, Any]:
    ordinal, observation, _body, size, request_hash = _selected(candidates)
    return {
        "schema": SCHEMA, "source_commit": runner._head(),
        "candidates_sha256": runner.digest(candidates),
        "code_sha256": {name: runner.digest(runner.REPO / name) for name in CODE},
        "model": MODEL, "endpoint": URL, "selected_ordinal": ordinal,
        "observation_sha256": hashlib.sha256(runner.canonical(observation).encode()).hexdigest(),
        "generate_request_sha256": request_hash, "generate_request_bytes": size,
        "maximum_count_requests": 1, "maximum_generation_requests": 0,
        "timeout_seconds": 30, "maximum_wall_seconds": 120, "retries": 0,
        "output_path": str(output.resolve()),
    }


def check(manifest: Path, candidates: Path, output: Path, gate: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = json.loads(manifest.read_text(encoding="utf-8"))
    if policy != make_manifest(candidates, output):
        raise ValueError("gemini_count_manifest_drift")
    if gate is not None and json.loads(gate.read_text(encoding="utf-8")) != {
        "schema": "averitec-astra-launch-gate/v1",
        "manifest_sha256": runner.digest(manifest),
        "decision": "approved", "reviewer": "gpt-6-astra",
    }:
        raise ValueError("gemini_count_astra_gate")
    return policy, _selected(candidates)[2]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _count(body: dict[str, Any], key: str, timeout: int) -> int:
    payload = {"generateContentRequest": {"model": f"models/{MODEL}", **body}}
    request = Request(URL, data=runner.wire_bytes(payload), method="POST",
                      headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    count = decoded.get("totalTokens") if isinstance(decoded, dict) else None
    if type(count) is not int or count < 1:
        raise ValueError("gemini_count_response_invalid")
    return count


def execute(manifest: Path, candidates: Path, output: Path, gate: Path, *, key: str) -> dict[str, Any]:
    policy, body = check(manifest, candidates, output, gate)
    if output.exists() or not key:
        raise ValueError("gemini_count_output_or_credential")
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError("gemini_count_alarm_unavailable")
    output.mkdir(parents=True, mode=0o700)
    _write_once(output / "intent.json", {
        "manifest_sha256": runner.digest(manifest), "selected_ordinal": policy["selected_ordinal"],
        "generate_request_sha256": policy["generate_request_sha256"],
    })
    previous_handler = signal.getsignal(signal.SIGALRM)
    def deadline(_signum: int, _frame: Any) -> None:
        raise TimeoutError("gemini_count_wall_deadline")
    started = monotonic()
    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, policy["maximum_wall_seconds"])
        count = _count(body, key, policy["timeout_seconds"])
        result = {"status": "ok", "total_tokens": count, "elapsed_seconds": monotonic() - started}
    except BaseException as error:
        result = {"status": "unknown_submission_outcome", "error_type": type(error).__name__,
                  "elapsed_seconds": monotonic() - started}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    _write_once(output / "result.json", result)
    receipt = {"schema": "averitec-gemini-count-smoke-receipt/v1",
               "manifest_sha256": runner.digest(manifest), "requests_attempted": 1,
               "complete": result["status"] == "ok",
               "intent_sha256": runner.digest(output / "intent.json"),
               "result_sha256": runner.digest(output / "result.json")}
    _write_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--astra-gate", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--make-manifest", action="store_true")
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.make_manifest:
        _write_once(args.manifest, make_manifest(args.candidates, args.output))
        print(json.dumps({"manifest_sha256": runner.digest(args.manifest), "count_calls": 0}, sort_keys=True))
    elif args.plan:
        policy, _ = check(args.manifest, args.candidates, args.output)
        print(json.dumps({"ordinal": policy["selected_ordinal"], "request_bytes": policy["generate_request_bytes"],
                          "count_calls": 0}, sort_keys=True))
    else:
        if args.astra_gate is None:
            raise ValueError("gemini_count_astra_gate_required")
        key = os.environ.get("GEMINI_API_KEY", "")
        print(json.dumps(execute(args.manifest, args.candidates, args.output, args.astra_gate, key=key), sort_keys=True))


if __name__ == "__main__":
    main()
