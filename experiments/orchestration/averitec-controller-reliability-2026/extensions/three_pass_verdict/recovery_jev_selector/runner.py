"""Separate, one-shot Jev selector recovery. No HTTP in plan/check modes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
THREE = HERE.parent
ROOT = HERE.parents[5]
sys.path.insert(0, str(THREE))
import selector_passes as old  # noqa: E402

SCHEMA = "averitec-jev-selector-recovery/v1"
GATE_SCHEMA = "averitec-jev-selector-recovery-astra-gate/v1"
RECEIPT_SCHEMA = "averitec-jev-selector-recovery-receipt/v1"
COHORT = "jev-selector-recovery-20260923-1"
MODEL = "jev-1.13.0"
RETRY_DELAYS = (5, 10, 20)
MAX_PASS_ATTEMPTS = 1101
MAX_PASS_EXTRA = 100
MAX_PER_LOGICAL = 4
TIMEOUT = 30
WALL_CAP = 14400


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_sha(value: Any) -> str:
    return hashlib.sha256(old.wire_bytes(value)).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def file_spec(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def verified(spec: dict) -> Path:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("file_spec_required")
    path = Path(spec["path"])
    if not path.is_absolute() or not path.is_file() or sha(path) != spec["sha256"]:
        raise ValueError("file_hash_drift")
    return path


def budget_remaining(path: Path, phase: str) -> dict[str, int]:
    audit = read(path)
    if audit.get("schema") != "averitec-jev-selector-cumulative-budget/v1":
        raise ValueError("cumulative_budget_schema")
    cap = {"requests": 9910, "input_tokens": 9_000_000, "output_tokens": 1_000_000}
    fields = {"requests": "prior_requests", "input_tokens": "prior_input_tokens",
              "output_tokens": "prior_output_tokens"}
    if any(audit.get("maximum_" + k) != v for k, v in
           (("generation_requests", cap["requests"]), ("input_tokens", cap["input_tokens"]),
            ("output_tokens", cap["output_tokens"]))):
        raise ValueError("cumulative_budget_cap_drift")
    remaining = {}
    for key, field in fields.items():
        used = audit.get(field)
        if type(used) is not int or used < 0 or used > cap[key]:
            raise ValueError("cumulative_budget_used_invalid")
        remaining[key] = cap[key] - used
    baseline = {"prior_requests": 4655, "prior_input_tokens": 3_201_246,
                "prior_output_tokens": 285_652}
    if any(audit[field] < value for field, value in baseline.items()):
        raise ValueError("historical_jev_usage_omitted")
    if phase == "pass" and audit["prior_requests"] < baseline["prior_requests"] + 1:
        raise ValueError("smoke_usage_omitted_from_pass_budget")
    if (remaining["requests"] < (1 if phase == "smoke" else 1001)
            or remaining["input_tokens"] < 2459 or remaining["output_tokens"] < 1024):
        raise ValueError("cumulative_budget_insufficient")
    return remaining


def new_json(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write((old.canonical(value) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def append(path: Path, value: dict) -> None:
    with path.open("ab") as handle:
        handle.write((old.canonical(value) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def plans(candidates: Path) -> list[tuple[str, str | None, str | None, dict]]:
    measured = old.previous.plan_observations(candidates)
    if len(measured) != 1000:
        raise ValueError("measured_count_drift")
    return [("warmup", None, None, old.WARMUP)] + [
        ("measured", row["case_id"], row["id"], row["observation"]) for row in measured]


def preflight(candidates: Path, original_preflight: Path) -> tuple[list[tuple], list[dict]]:
    calls = plans(candidates)
    selector = old._make_selector("jev", endpoint=None, laya_checkpoint=None, keys={})
    requests = old._capture_http_requests("jev", selector, [row[3] for row in calls],
                                          (old.EVIDENCE / "selector_instructions.txt").read_text())
    saved = read(original_preflight)
    records = saved.get("records")
    if len(requests) != 1001 or not isinstance(records, list) or len(records) != 1001:
        raise ValueError("original_preflight_count_drift")
    for (phase, case_id, candidate_id, observation), request, record in zip(calls, requests, records):
        if (record.get("request_sha256") != body_sha(request)
                or record.get("observation_sha256") != hashlib.sha256(old.canonical(observation).encode()).hexdigest()
                or record.get("serialized_bytes") != len(old.wire_bytes(request))
                or record.get("input_tokens") != len(old.wire_bytes(request)) + 1024
                or len(old.wire_bytes(request)) > 2048):
            raise ValueError("original_preflight_request_drift")
    return calls, records


def code_binding() -> dict[str, str]:
    paths = [Path(__file__), HERE / "config.yaml", THREE / "selector_passes.py", old.EVIDENCE / "run_selector.py",
             old.EVIDENCE / "selector_runtime.py", old.EVIDENCE / "selector_instructions.txt",
             old.EXPERIMENT / "providers.py", old.EXPERIMENT / "tracing.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def prepare(*, phase: str, candidates: Path, original_preflight: Path,
            original_failed_receipt: Path, original_ledger: Path, output: Path,
            manifest: Path, smoke_receipt: Path | None = None,
            prior_budget: Path | None = None) -> dict:
    if phase not in {"smoke", "pass"} or not output.is_absolute() or output.exists():
        raise ValueError("phase_or_output_invalid")
    calls, records = preflight(candidates, original_preflight)
    failed = read(original_failed_receipt)
    if (failed.get("arm") != "jev" or failed.get("pass") != 3
            or failed.get("complete") is not False or failed.get("stop_reason") != "unknown_transport_outcome_no_reissue"):
        raise ValueError("original_failure_provenance")
    if phase == "pass" and smoke_receipt is None:
        raise ValueError("smoke_receipt_required")
    if prior_budget is None:
        raise ValueError("cumulative_budget_audit_required")
    remaining = budget_remaining(prior_budget, phase)
    result = {"schema": SCHEMA, "cohort": COHORT, "phase": phase, "arm": "jev",
              "model": MODEL, "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "code_sha256": code_binding(), "candidates": file_spec(candidates),
              "original_preflight": file_spec(original_preflight),
              "original_failed_receipt": file_spec(original_failed_receipt),
              "original_hosted_ledger": file_spec(original_ledger),
              "prior_budget_audit": file_spec(prior_budget),
              "smoke_receipt": file_spec(smoke_receipt) if smoke_receipt else None,
              "output": str(output), "planned_ids_sha256": hashlib.sha256(old.canonical(
                  [(r[1], r[2]) for r in calls[1:]]).encode()).hexdigest(),
              "request_hashes_sha256": hashlib.sha256(old.canonical(
                  [r["request_sha256"] for r in records]).encode()).hexdigest(),
              "limits": {"logical_calls": 1 if phase == "smoke" else 1001,
                         "http_attempts": 1 if phase == "smoke" else MAX_PASS_ATTEMPTS,
                         "extra_http_attempts": 0 if phase == "smoke" else MAX_PASS_EXTRA,
                         "input_tokens": 2459 if phase == "smoke" else min(3_000_000, remaining["input_tokens"]),
                         "output_tokens": 1024 if phase == "smoke" else remaining["output_tokens"],
                         "wall_seconds": 120 if phase == "smoke" else WALL_CAP,
                         "minimum_request_spacing_seconds": 0 if phase == "smoke" else 1},
              "retry": {"http_statuses": [] if phase == "smoke" else [529],
                        "maximum_retries_per_logical_request": 0 if phase == "smoke" else 3,
                        "backoff_seconds": [] if phase == "smoke" else list(RETRY_DELAYS),
                        "retry_after": "honor_if_valid", "unknown_transport": "stop_no_retry"}}
    # The provisional caps above are upper bounds. Astra must approve the exact
    # manifest and a current cumulative budget audit before any provider call.
    new_json(manifest, result)
    return result


class StopRun(BaseException):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def retry_after_seconds(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers else None
    if value is None:
        return None
    value = str(value).strip()
    if value.isdigit():
        return float(value)
    date = parsedate_to_datetime(value)
    if date.tzinfo is None:
        raise ValueError("retry_after_timezone")
    return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())


def validate(manifest: Path, gate: Path | None = None) -> tuple[dict, list[tuple], list[dict]]:
    m = read(manifest)
    if m.get("schema") != SCHEMA or m.get("cohort") != COHORT or m.get("arm") != "jev" or m.get("model") != MODEL:
        raise ValueError("manifest_identity")
    phase = m.get("phase")
    if phase not in {"smoke", "pass"} or not Path(m.get("output", "")).is_absolute():
        raise ValueError("manifest_phase_or_output")
    if m.get("source_commit") != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip():
        raise ValueError("source_commit_drift")
    if m.get("code_sha256") != code_binding():
        raise ValueError("code_binding_drift")
    candidates = verified(m["candidates"])
    original_preflight = verified(m["original_preflight"])
    failed = read(verified(m["original_failed_receipt"]))
    verified(m["original_hosted_ledger"])
    if failed.get("arm") != "jev" or failed.get("pass") != 3 or failed.get("complete") is not False:
        raise ValueError("original_failure_provenance")
    if m.get("prior_budget_audit") is None:
        raise ValueError("cumulative_budget_audit_required")
    remaining = budget_remaining(verified(m["prior_budget_audit"]), phase)
    if phase == "pass":
        smoke = read(verified(m["smoke_receipt"]))
        if (smoke.get("schema") != RECEIPT_SCHEMA or smoke.get("phase") != "smoke"
                or smoke.get("complete") is not True or smoke.get("measured_ok") != 1):
            raise ValueError("successful_smoke_required")
    calls, records = preflight(candidates, original_preflight)
    if m.get("planned_ids_sha256") != hashlib.sha256(old.canonical([(r[1], r[2]) for r in calls[1:]]).encode()).hexdigest():
        raise ValueError("planned_ids_drift")
    if m.get("request_hashes_sha256") != hashlib.sha256(old.canonical([r["request_sha256"] for r in records]).encode()).hexdigest():
        raise ValueError("request_hashes_drift")
    limits = m.get("limits", {})
    if (limits.get("logical_calls") != (1 if phase == "smoke" else 1001)
            or limits.get("http_attempts") != (1 if phase == "smoke" else MAX_PASS_ATTEMPTS)
            or limits.get("extra_http_attempts") != (0 if phase == "smoke" else MAX_PASS_EXTRA)
            or limits.get("wall_seconds") != (120 if phase == "smoke" else WALL_CAP)
            or limits.get("minimum_request_spacing_seconds") != (0 if phase == "smoke" else 1)
            or type(limits.get("input_tokens")) is not int or limits["input_tokens"] < 2459
            or type(limits.get("output_tokens")) is not int or limits["output_tokens"] < 1024
            or limits["input_tokens"] > remaining["input_tokens"]
            or limits["output_tokens"] > remaining["output_tokens"]
            or limits["http_attempts"] > remaining["requests"]):
        raise ValueError("limits_drift")
    if m.get("retry") != {"http_statuses": [] if phase == "smoke" else [529],
                          "maximum_retries_per_logical_request": 0 if phase == "smoke" else 3,
                          "backoff_seconds": [] if phase == "smoke" else list(RETRY_DELAYS),
                          "retry_after": "honor_if_valid", "unknown_transport": "stop_no_retry"}:
        raise ValueError("retry_policy_drift")
    if gate is not None:
        g = read(gate)
        if (g.get("schema") != GATE_SCHEMA or g.get("decision") != "approved"
                or g.get("reviewer") != "gpt-6-astra" or g.get("phase") != phase
                or g.get("manifest_sha256") != sha(manifest) or g.get("cohort") != COHORT):
            raise ValueError("exact_astra_gate_required")
    return m, calls, records


@contextmanager
def metered_http(*, m: dict, ledger: Path, expected_sha: str, ordinal: int,
                 counters: dict, sleep=time.sleep, clock=time.monotonic):
    """One logical request with durable intent/terminal events per HTTP attempt."""
    import providers
    original = providers._http_post
    phase = m["phase"]
    limits = m["limits"]
    calls = 0

    def post(url: str, body: dict, headers: dict, timeout: float) -> dict:
        nonlocal calls
        calls += 1
        if (calls != 1 or url != providers.TYPESAFE_SYSTEM_ONE_URL
                or body_sha(body) != expected_sha or not 0 < timeout <= TIMEOUT):
            raise StopRun("exact_request_drift")
        for attempt in range(1, MAX_PER_LOGICAL + 1 if phase == "pass" else 2):
            if (counters["http_attempts"] >= limits["http_attempts"]
                    or (attempt > 1 and counters["extra_attempts"] >= limits["extra_http_attempts"])):
                raise StopRun("attempt_cap")
            input_reserve = len(old.wire_bytes(body)) + 1024
            output_reserve = 1024
            if (counters["input_charged"] + input_reserve > limits["input_tokens"]
                    or counters["output_charged"] + output_reserve > limits["output_tokens"]
                    or clock() - counters["started"] + TIMEOUT > limits["wall_seconds"]):
                raise StopRun("resource_reservation_cap")
            if counters["last_attempt_start"] is not None:
                wait = limits["minimum_request_spacing_seconds"] - (clock() - counters["last_attempt_start"])
                if wait > 0:
                    sleep(wait)
            attempt_id = f"{phase}-{ordinal:04d}-{attempt}"
            append(ledger, {"kind": "intent", "attempt_id": attempt_id, "ordinal": ordinal,
                            "attempt": attempt, "phase": "warmup" if ordinal == 1 else "measured",
                            "request_sha256": expected_sha, "reserved_input_tokens": input_reserve,
                            "reserved_output_tokens": output_reserve, "timeout_seconds": TIMEOUT})
            counters["http_attempts"] += 1
            counters["extra_attempts"] += int(attempt > 1)
            counters["last_attempt_start"] = clock()
            started = clock()
            try:
                request = Request(url, data=old.wire_bytes(body),
                                  headers={"Content-Type": "application/json", **headers}, method="POST")
                with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("response_not_object")
            except HTTPError as error:
                counters["input_charged"] += input_reserve
                counters["output_charged"] += output_reserve
                append(ledger, {"kind": "http_error", "attempt_id": attempt_id, "status": error.code,
                                "charged_input_tokens": input_reserve, "charged_output_tokens": output_reserve,
                                "elapsed_seconds": clock() - started})
                if error.code != 529 or phase != "pass" or attempt == MAX_PER_LOGICAL:
                    raise StopRun(f"http_{error.code}") from None
                counters["first_attempt_529"] += int(attempt == 1 and ordinal > 1)
                try:
                    jitter = int(hashlib.sha256((expected_sha + ":" + str(attempt)).encode()).hexdigest()[:8], 16) / 0xffffffff
                    delay = max(RETRY_DELAYS[attempt - 1] + jitter, retry_after_seconds(error.headers) or 0)
                except (ValueError, TypeError, OverflowError):
                    raise StopRun("invalid_retry_after") from None
                if clock() - counters["started"] + delay + TIMEOUT > limits["wall_seconds"]:
                    raise StopRun("wall_cap_before_retry")
                append(ledger, {"kind": "retry_wait", "attempt_id": attempt_id, "seconds": delay})
                sleep(delay)
                continue
            except BaseException:
                counters["input_charged"] += input_reserve
                counters["output_charged"] += output_reserve
                append(ledger, {"kind": "uncertain", "attempt_id": attempt_id,
                                "charged_input_tokens": input_reserve, "charged_output_tokens": output_reserve,
                                "elapsed_seconds": clock() - started})
                raise StopRun("unknown_transport_or_decode") from None
            usage = payload.get("usage")
            input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
            output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
            valid = all(type(x) is int and x >= 0 for x in (input_tokens, output_tokens))
            counters["input_charged"] += input_tokens if valid else input_reserve
            counters["output_charged"] += output_tokens if valid else output_reserve
            append(ledger, {"kind": "response", "attempt_id": attempt_id,
                            "response_sha256": body_sha(payload), "returned_model": payload.get("model"),
                            "input_tokens": input_tokens, "output_tokens": output_tokens,
                            "charged_input_tokens": input_tokens if valid else input_reserve,
                            "charged_output_tokens": output_tokens if valid else output_reserve,
                            "elapsed_seconds": clock() - started})
            if not valid or input_tokens > input_reserve or output_tokens > output_reserve:
                raise StopRun("usage_unknown_or_reserve_overrun")
            if payload.get("model") != MODEL:
                raise StopRun("returned_model_drift")
            return payload
        raise StopRun("retry_limit")

    providers._http_post = post
    try:
        yield lambda: calls
    finally:
        providers._http_post = original


def execute(manifest: Path, gate: Path) -> dict:
    m, calls, records = validate(manifest, gate)
    output = Path(m["output"])
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output_must_be_new_directory")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ValueError("TYPESAFE_API_KEY_missing")
    os.mkdir(output, 0o700)
    new_json(output / "run-metadata.json", {"schema": SCHEMA, "cohort": COHORT,
                                           "phase": m["phase"], "manifest_sha256": sha(manifest),
                                           "astra_gate_sha256": sha(gate), "source_commit": m["source_commit"]})
    for name in ("http-ledger.jsonl", "results.jsonl", "choices.jsonl", "traces.jsonl"):
        (output / name).touch(mode=0o600, exist_ok=False)
    counters = {"logical_calls": 0, "http_attempts": 0, "extra_attempts": 0, "input_charged": 0,
                "output_charged": 0, "first_attempt_529": 0,
                "started": time.monotonic(), "last_attempt_start": None}
    stopped = False
    stop_reason = None
    selector = old._make_selector("jev", endpoint=None, laya_checkpoint=None, keys={"jev": key})
    instructions = (old.EVIDENCE / "selector_instructions.txt").read_text()
    run_id = hashlib.sha256((sha(manifest) + COHORT).encode()).hexdigest()[:32]
    original_signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def mark_stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    for s in original_signals:
        signal.signal(s, mark_stop)
    try:
        selected = [(2, calls[1])] if m["phase"] == "smoke" else list(enumerate(calls, 1))
        for ordinal, (phase, case_id, candidate_id, observation) in selected:
            if stopped:
                stop_reason = "signal_before_call"
                break
            recorder = old.TraceRecorder(output / "traces.jsonl", run_id,
                                         f"recovery:{ordinal}:{case_id or 'warmup'}",
                                         {"arm": "jev", "phase": phase, "cohort": COHORT})
            old.previous._set_tracer(selector, recorder)
            trace_offset = (output / "traces.jsonl").stat().st_size
            append(output / "results.jsonl", {"kind": "intent", "ordinal": ordinal, "phase": phase,
                                                  "case_id": case_id, "id": candidate_id,
                                                  "observation_sha256": hashlib.sha256(old.canonical(observation).encode()).hexdigest()})
            counters["logical_calls"] += 1
            try:
                attempts_before = counters["http_attempts"]
                with metered_http(m=m, ledger=output / "http-ledger.jsonl",
                                  expected_sha=records[ordinal - 1]["request_sha256"],
                                  ordinal=ordinal, counters=counters) as call_count:
                    result = old.previous._bounded_choose(selector, observation, instructions,
                                                           m["limits"]["wall_seconds"] - (time.monotonic() - counters["started"]))
                    if call_count() != 1:
                        raise StopRun("dispatch_count_drift")
                trace_complete, trace_model = old._trace_response(output / "traces.jsonl", recorder.trace_id, "jev", trace_offset)
                entry = {"kind": "result", "ordinal": ordinal, "phase": phase, "case_id": case_id,
                         "id": candidate_id, "action": result.action, "outcome": result.outcome,
                         "error_code": result.error_code, "model": result.model,
                         "returned_model": result.returned_model, "trace_returned_model": trace_model,
                         "trace_complete": trace_complete,
                         "http_attempts_for_logical": counters["http_attempts"] - attempts_before,
                         "usage": {"input_tokens": result.input_tokens,
                                                                       "output_tokens": result.output_tokens}}
                append(output / "results.jsonl", entry)
                append(output / "choices.jsonl", {key: entry[key] for key in
                                                ("phase", "case_id", "id", "action", "outcome", "error_code",
                                                 "model", "returned_model", "trace_returned_model", "trace_complete",
                                                 "http_attempts_for_logical", "usage")})
                if (result.outcome != "ok" or result.action not in old.BINARY_ACTIONS
                        or result.returned_model != MODEL or trace_model != MODEL or not trace_complete):
                    stop_reason = "result_or_identity_invalid"
                    break
            except StopRun as exc:
                stop_reason = str(exc)
                append(output / "results.jsonl", {"kind": "stopped", "ordinal": ordinal, "reason": stop_reason})
                break
            except Exception as exc:
                stop_reason = "runner_error_" + type(exc).__name__
                append(output / "results.jsonl", {"kind": "stopped", "ordinal": ordinal, "reason": stop_reason})
                break
    finally:
        for s, handler in original_signals.items():
            signal.signal(s, handler)
    measured = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
                if line.strip() and json.loads(line).get("kind") == "result" and json.loads(line).get("phase") == "measured"]
    complete = stop_reason is None and counters["logical_calls"] == m["limits"]["logical_calls"]
    receipt = {"schema": RECEIPT_SCHEMA, "cohort": COHORT, "phase": m["phase"],
               "complete": complete, "stop_reason": stop_reason, "measured_results": len(measured),
               "measured_ok": sum(row["outcome"] == "ok" for row in measured),
               "first_attempt_measured_http_529": counters["first_attempt_529"],
               "first_attempt_no_retry_measured_ok": sum(row["outcome"] == "ok" and row["http_attempts_for_logical"] == 1 for row in measured),
               "after_retry_measured_ok": sum(row["outcome"] == "ok" for row in measured),
               "logical_calls": counters["logical_calls"], "http_attempts": counters["http_attempts"],
               "charged_input_tokens": counters["input_charged"],
               "charged_output_tokens": counters["output_charged"],
               "manifest_sha256": sha(manifest), "gate_sha256": sha(gate),
               "journal_sha256": {name: sha(output / name) for name in
                                  ("run-metadata.json", "http-ledger.jsonl", "results.jsonl", "choices.jsonl", "traces.jsonl")}}
    new_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare", help="offline immutable manifest")
    prep.add_argument("--phase", choices=("smoke", "pass"), required=True)
    for p in ("candidates", "original_preflight", "original_failed_receipt", "original_ledger", "output", "manifest"):
        prep.add_argument("--" + p.replace("_", "-"), type=Path, required=True)
    prep.add_argument("--smoke-receipt", type=Path)
    prep.add_argument("--prior-budget", type=Path)
    check = sub.add_parser("check", help="offline manifest validation")
    check.add_argument("--manifest", type=Path, required=True)
    run = sub.add_parser("execute", help="live one-shot; exact Astra gate required")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--astra-gate", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(phase=args.phase, candidates=args.candidates,
                         original_preflight=args.original_preflight,
                         original_failed_receipt=args.original_failed_receipt,
                         original_ledger=args.original_ledger, output=args.output,
                         manifest=args.manifest, smoke_receipt=args.smoke_receipt,
                         prior_budget=args.prior_budget)
    elif args.mode == "check":
        result = {"valid": True, "phase": validate(args.manifest)[0]["phase"]}
    else:
        result = execute(args.manifest, args.astra_gate)
    print(old.canonical(result))


if __name__ == "__main__":
    main()
