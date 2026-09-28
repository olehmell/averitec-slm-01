"""One hash-bound Jev selector request on the longest frozen development input.

The smoke has exactly one measured request and no extra warmup. It checks the
estimated resource policy; it cannot establish an exact provider token bound.
Plan mode has no model call. Execute requires an Astra gate for this manifest.
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

import selector_passes as runner
from tracing import TraceRecorder


SCHEMA = "averitec-jev-selector-smoke/v1"
CODE = (
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/jev_smoke.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/selector_passes.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/jev_preflight.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/run_selector.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_runtime.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_instructions.txt",
    "experiments/orchestration/averitec-controller-reliability-2026/providers.py",
    "experiments/orchestration/averitec-controller-reliability-2026/tracing.py",
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _one_post_no_redirect(url: str, body: dict, headers: dict, timeout: float) -> dict:
    if url != runner.provider_module.TYPESAFE_SYSTEM_ONE_URL:
        raise ValueError("jev_smoke_endpoint_drift")
    request = Request(url, data=runner.wire_bytes(body),
                      headers={"Content-Type": "application/json", **headers}, method="POST")
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("jev_smoke_response_not_object")
    return payload


@contextmanager
def _single_request_deadline(seconds: int):
    """Bind one HTTP dispatch, disallow redirects, and interrupt slow reads."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("jev_smoke_main_thread_required")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] or previous_timer[1]:
        raise ValueError("jev_smoke_existing_alarm")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_post = runner.provider_module._http_post

    def deadline(_signum: int, _frame: Any) -> None:
        raise TimeoutError("jev_smoke_wall_deadline")

    try:
        runner.provider_module._http_post = _one_post_no_redirect
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        runner.provider_module._http_post = previous_post


def _write_once(path: Path, value: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write((runner.canonical(value) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _selection(candidates: Path, preflight: Path) -> tuple[dict, dict, int, dict]:
    report = json.loads(preflight.read_text(encoding="utf-8"))
    planned = runner.previous.plan_observations(candidates)
    if (report.get("schema") != "averitec-selector-token-preflight/v2"
            or report.get("method") != "estimated_wire_bytes_plus_1024"
            or not isinstance(report.get("records"), list) or len(report["records"]) != len(planned) + 1):
        raise ValueError("jev_smoke_preflight_shape")
    ordinal = max(range(1, len(report["records"])), key=lambda index: report["records"][index]["serialized_bytes"])
    row = planned[ordinal - 1]
    record = report["records"][ordinal]
    selector = runner._make_selector("jev", endpoint=None, laya_checkpoint=None, keys={})
    instructions = (runner.EVIDENCE / "selector_instructions.txt").read_text(encoding="utf-8")
    request = runner._capture_http_requests("jev", selector, [row["observation"]], instructions)[0]
    wire = runner.wire_bytes(request)
    if (record["request_sha256"] != hashlib.sha256(wire).hexdigest()
            or record["serialized_bytes"] != len(wire)
            or record["input_tokens"] != len(wire) + 1024
            or len(wire) > 2048):
        raise ValueError("jev_smoke_request_drift")
    return row, record, ordinal, request


def make_manifest(candidates: Path, preflight: Path, provider_contract: Path, output: Path) -> dict:
    row, record, ordinal, _ = _selection(candidates, preflight)
    return {"schema": SCHEMA, "source_commit": runner._head(),
            "candidates_sha256": runner.digest(candidates), "preflight_sha256": runner.digest(preflight),
            "provider_contract_sha256": runner.digest(provider_contract),
            "code_sha256": {name: runner.digest(runner.REPO / name) for name in CODE},
            "model": "jev-1.13.0", "profile": "baseline", "returned_model": "jev-1.13.0",
            "ordinal_in_full_pass": ordinal + 1, "case_id": row["case_id"], "candidate_id": row["id"],
            "request_sha256": record["request_sha256"], "serialized_bytes": record["serialized_bytes"],
            "maximum_serialized_bytes": 2048, "assumed_context_tokens": 32768,
            "input_reserve_tokens": record["input_tokens"], "output_reserve_tokens": 1024,
            "maximum_requests": 1, "request_timeout_seconds": 30, "maximum_wall_seconds": 120,
            "retries": 0, "lossless_context_verified": False, "output_limit_enforced": False,
            "output_path": str(output.resolve())}


def check(manifest: Path, candidates: Path, preflight: Path, provider_contract: Path,
          output: Path, gate: Path | None = None) -> tuple[dict, dict, dict]:
    given = json.loads(manifest.read_text(encoding="utf-8"))
    expected = make_manifest(candidates, preflight, provider_contract, output)
    report = json.loads(preflight.read_text(encoding="utf-8"))
    if report.get("method_identity_sha256") != runner.digest(provider_contract):
        raise ValueError("jev_smoke_provider_contract_drift")
    if given != expected:
        raise ValueError("jev_smoke_manifest_drift")
    row, record, _, request = _selection(candidates, preflight)
    if gate is not None:
        review = json.loads(gate.read_text(encoding="utf-8"))
        if review != {"schema": "averitec-astra-launch-gate/v1", "manifest_sha256": runner.digest(manifest),
                       "decision": "approved", "reviewer": "gpt-6-astra"}:
            raise ValueError("jev_smoke_astra_gate")
    return given, row, {"record": record, "request": request}


def execute(manifest: Path, candidates: Path, preflight: Path, provider_contract: Path,
            output: Path, gate: Path) -> dict:
    policy, row, _ = check(manifest, candidates, preflight, provider_contract, output, gate)
    if output.exists():
        raise ValueError("jev_smoke_output_exists")
    selector = runner._make_selector("jev", endpoint=None, laya_checkpoint=None, keys={})
    runner._local_identity("jev", selector, policy["model"], policy["returned_model"], policy["profile"])
    if not runner.previous._provider(selector).key:
        raise ValueError("jev_smoke_credential_unavailable")
    output.mkdir(parents=True, mode=0o700)
    trace_path = output / "traces.jsonl"
    trace_path.touch(exist_ok=False)
    intent = {"schema": SCHEMA, "manifest_sha256": runner.digest(manifest),
              "case_id": row["case_id"], "id": row["id"],
              "request_sha256": policy["request_sha256"], "serialized_bytes": policy["serialized_bytes"],
              "input_reserve_tokens": policy["input_reserve_tokens"],
              "output_reserve_tokens": policy["output_reserve_tokens"]}
    _write_once(output / "intent.json", intent)
    tracer = TraceRecorder(trace_path, "jev-selector-smoke", "longest-development-request", {"phase": "smoke"})
    runner.previous._set_tracer(selector, tracer)
    instructions = (runner.EVIDENCE / "selector_instructions.txt").read_text(encoding="utf-8")
    begun = monotonic()
    try:
        with _single_request_deadline(policy["maximum_wall_seconds"]):
            result = selector.choose(row["observation"], instructions, list(runner.BINARY_ACTIONS))
        elapsed = monotonic() - begun
        trace_complete, trace_model = runner._trace_response(trace_path, tracer.trace_id, "jev")
        usage = {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
        status = ("ok" if result.outcome == "ok" and result.action in runner.BINARY_ACTIONS
                  and result.model == policy["model"] and result.returned_model == policy["returned_model"]
                  and trace_complete and trace_model == policy["returned_model"]
                  and all(type(usage[name]) is int and 0 <= usage[name] <= policy[f"{name.split('_')[0]}_reserve_tokens"]
                          for name in ("input_tokens", "output_tokens"))
                  and elapsed <= policy["maximum_wall_seconds"] else
                  "unknown_submission_outcome" if result.outcome == "transport_error" else "qualification_failed")
        measured = {"status": status, "outcome": result.outcome, "action": result.action,
                    "model": result.model, "returned_model": result.returned_model,
                    "trace_returned_model": trace_model, "trace_complete": trace_complete,
                    "usage": usage, "elapsed_seconds": elapsed, "error_code": result.error_code}
    except BaseException as error:
        measured = {"status": "unknown_submission_outcome", "error_type": type(error).__name__,
                    "usage": None, "elapsed_seconds": monotonic() - begun}
    _write_once(output / "result.json", measured)
    receipt = {"schema": "averitec-jev-selector-smoke-receipt/v1", "manifest_sha256": runner.digest(manifest),
               "requests_attempted": 1, "complete": measured["status"] == "ok",
               "result_sha256": runner.digest(output / "result.json"),
               "trace_sha256": runner.digest(trace_path), "intent_sha256": runner.digest(output / "intent.json")}
    _write_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--provider-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--astra-gate", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-manifest", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.make_manifest:
        if args.manifest is None:
            raise ValueError("jev_smoke_manifest_path_required")
        value = make_manifest(args.candidates, args.preflight, args.provider_contract, args.output)
        _write_once(args.manifest, value)
        print(json.dumps({"manifest_sha256": runner.digest(args.manifest), "model_calls": 0}, sort_keys=True))
    elif args.plan:
        if args.manifest is None:
            raise ValueError("jev_smoke_manifest_path_required")
        value, _, _ = check(args.manifest, args.candidates, args.preflight, args.provider_contract, args.output)
        print(json.dumps({"request_bytes": value["serialized_bytes"], "input_reserve_tokens": value["input_reserve_tokens"],
                          "output_reserve_tokens": value["output_reserve_tokens"], "model_calls": 0}, sort_keys=True))
    else:
        if args.manifest is None or args.astra_gate is None:
            raise ValueError("jev_smoke_exact_manifest_and_astra_gate_required")
        print(json.dumps(execute(args.manifest, args.candidates, args.preflight,
                                 args.provider_contract, args.output, args.astra_gate), sort_keys=True))


if __name__ == "__main__":
    main()
