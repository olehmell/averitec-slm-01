"""Bounded, append-only runner for the sealed evidence-selection supplement.

This module deliberately has no access path to reference labels, annotations,
or retrieval.  It writes an intent before every possible forward/request and
never resumes an output directory, so an interrupted request is not reissued.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from time import monotonic, time
from typing import Any, Callable, Iterable, Mapping

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from annotation import validate_candidates  # Safe: standard-library-only validation.
from selector_runtime import BINARY_ACTIONS, LAYA_TYPED_MODEL, create_selector
from tracing import TraceRecorder
from recovery_runtime import RECOVERY_PROFILE_IDENTITIES, create_recovery_selector


SCHEMA = "averitec-selector-launch/v1"
SCHEMA_V2 = "averitec-selector-launch/v2"
CONTROLLERS = {
    "jev": ("jev-1.13.0", "baseline", 14400),
    "gemini": ("gemini-3.1-flash-lite", "gemini_native", 14400),
    "qwen": ("Qwen/Qwen3.5-4B", "baseline", 900),
    "lfm": ("LiquidAI/LFM2.5-1.2B-Instruct", "lfm_native", 900),
    "lfm26": ("LiquidAI/LFM2.5-2.6B", "lfm26_native", 8100),
    "laya_typed": (LAYA_TYPED_MODEL, "native_binary_choice", 900),
}
EXPECTED_RETURNED_MODELS = {name: values[0] for name, values in CONTROLLERS.items()}
GPU_CONTROLLERS = {"qwen", "lfm", "lfm26", "laya_typed"}
CONTROLLERS_V2 = {
    "gemini": ("gemini-3.1-flash-lite", "gemini_native_timeout120_v2", 14400),
    "qwen": ("Qwen/Qwen3.5-4B", "qwen_baseline_warmup180_v2", 1800),
    "lfm26": ("LiquidAI/LFM2.5-2.6B", "lfm26_native_4096_v2", 9000),
}
EXPECTED_RETURNED_MODELS_V2 = {name: values[0] for name, values in CONTROLLERS_V2.items()}
REQUIRED_CODE = {
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/annotation.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/run_selector.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_runtime.py",
    "experiments/orchestration/averitec-controller-reliability-2026/tracing.py",
    "experiments/orchestration/averitec-controller-reliability-2026/providers.py",
    "experiments/orchestration/averitec-controller-reliability-2026/gemini_provider.py",
    "experiments/orchestration/averitec-controller-reliability-2026/generation_profiles.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_instructions.txt",
}
REQUIRED_CODE_V2 = REQUIRED_CODE | {
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/recovery_runtime.py",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("selector_invalid_json") from None


def _safe_relpath(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("selector_manifest_code_path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("selector_manifest_code_path")
    return path


def _head_commit() -> str:
    """Bind a checkout to HEAD, or a verified source archive to its marker."""
    if (REPO_ROOT / ".git").exists():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            raise ValueError("selector_git_head_unavailable") from None
    marker = REPO_ROOT / "SOURCE_COMMIT"
    try:
        commit = marker.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        raise ValueError("selector_source_commit_marker_missing") from None
    if not HEX40.fullmatch(commit):
        raise ValueError("selector_source_commit_marker_invalid")
    return commit


def _validate_manifest(value: Any, candidates: Path, instructions: Path) -> dict[str, Any]:
    keys = {"schema", "source_commit", "candidates_sha256", "reference_sha256", "instructions_sha256", "code_sha256", "controllers", "max_gpu_seconds", "retries", "approved_date"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != SCHEMA:
        raise ValueError("selector_manifest_shape")
    if value.get("approved_date") != "2026-09-20" or value.get("retries") != 0 or value.get("max_gpu_seconds") != 10800:
        raise ValueError("selector_manifest_policy")
    if not isinstance(value.get("source_commit"), str) or not HEX40.fullmatch(value["source_commit"]):
        raise ValueError("selector_manifest_source_commit")
    if value["source_commit"] != _head_commit():
        raise ValueError("selector_manifest_source_commit_drift")
    for name in ("candidates_sha256", "reference_sha256", "instructions_sha256"):
        if not isinstance(value.get(name), str) or not HEX64.fullmatch(value[name]):
            raise ValueError("selector_manifest_hash")
    if value["candidates_sha256"] != _sha256_path(candidates) or value["instructions_sha256"] != _sha256_path(instructions):
        raise ValueError("selector_manifest_input_drift")
    # ``reference_sha256`` is intentionally validated only as a hash-shaped
    # binding; no reference path is accepted or opened by this runner.
    code = value.get("code_sha256")
    if not isinstance(code, dict) or not REQUIRED_CODE.issubset(code):
        raise ValueError("selector_manifest_code_missing")
    for name, expected in code.items():
        relative = _safe_relpath(name)
        if not isinstance(expected, str) or not HEX64.fullmatch(expected):
            raise ValueError("selector_manifest_code_hash")
        path = REPO_ROOT / relative
        if not path.is_file() or _sha256_path(path) != expected:
            raise ValueError("selector_manifest_code_drift")
    controllers = value.get("controllers")
    if not isinstance(controllers, dict) or set(controllers) != set(CONTROLLERS):
        raise ValueError("selector_manifest_controllers")
    for name, (model, profile, wall) in CONTROLLERS.items():
        row = controllers[name]
        if not isinstance(row, dict) or set(row) != {"model", "profile", "measured_calls", "warmup_calls", "maximum_calls", "wall_seconds"}:
            raise ValueError("selector_manifest_controller_shape")
        if row != {"model": model, "profile": profile, "measured_calls": 1000, "warmup_calls": 1, "maximum_calls": 1001, "wall_seconds": wall}:
            raise ValueError("selector_manifest_controller_policy")
    return value


def _validate_manifest_v2(value: Any, candidates: Path, instructions: Path) -> dict[str, Any]:
    """Validate the independent rerun binding without adding manifest fields."""
    keys = {"schema", "source_commit", "candidates_sha256", "reference_sha256", "instructions_sha256", "code_sha256", "controllers", "max_gpu_seconds", "retries", "approved_date"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != SCHEMA_V2:
        raise ValueError("selector_v2_manifest_shape")
    if value.get("approved_date") != "2026-09-20" or value.get("retries") != 0 or value.get("max_gpu_seconds") != 10800:
        raise ValueError("selector_v2_manifest_policy")
    if not isinstance(value.get("source_commit"), str) or not HEX40.fullmatch(value["source_commit"]) or value["source_commit"] != _head_commit():
        raise ValueError("selector_v2_manifest_source_commit")
    for name in ("candidates_sha256", "reference_sha256", "instructions_sha256"):
        if not isinstance(value.get(name), str) or not HEX64.fullmatch(value[name]):
            raise ValueError("selector_v2_manifest_hash")
    if value["candidates_sha256"] != _sha256_path(candidates) or value["instructions_sha256"] != _sha256_path(instructions):
        raise ValueError("selector_v2_manifest_input_drift")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or not REQUIRED_CODE_V2.issubset(code):
        raise ValueError("selector_v2_manifest_code_missing")
    for name, expected in code.items():
        relative = _safe_relpath(name)
        if not isinstance(expected, str) or not HEX64.fullmatch(expected) or not (REPO_ROOT / relative).is_file() or _sha256_path(REPO_ROOT / relative) != expected:
            raise ValueError("selector_v2_manifest_code_drift")
    controllers = value.get("controllers")
    if not isinstance(controllers, dict) or set(controllers) != set(CONTROLLERS_V2):
        raise ValueError("selector_v2_manifest_controllers")
    for name, (model, profile, wall) in CONTROLLERS_V2.items():
        expected = {"model": model, "profile": profile, "measured_calls": 1000, "warmup_calls": 1, "maximum_calls": 1001, "wall_seconds": wall}
        if controllers[name] != expected:
            raise ValueError("selector_v2_manifest_controller_policy")
    return value


def load_launch(manifest: Path, candidates: Path, instructions: Path | None = None) -> dict[str, Any]:
    """Load the sealed launch binding without opening any reference material."""
    if not candidates.is_file():
        raise ValueError("selector_candidates_missing")
    value = _load_json(manifest)
    instruction_path = instructions or HERE / "selector_instructions.txt"
    if isinstance(value, dict) and value.get("schema") == SCHEMA_V2:
        return _validate_manifest_v2(value, candidates, instruction_path)
    return _validate_manifest(value, candidates, instruction_path)


def plan_observations(candidates: Path, *, limit_for_test: int | None = None) -> list[dict[str, Any]]:
    """Produce the canonical 1,000 claim/candidate inputs and nothing else."""
    raw = _load_json(candidates)
    validate_candidates(raw)
    planned: list[dict[str, Any]] = []
    for case in raw["cases"]:
        for candidate in case["candidates"]:
            planned.append({"case_id": case["case_id"], "id": candidate["id"], "observation": {
                "claim": case["claim"],
                "candidate": {key: candidate[key] for key in ("id", "text", "url")},
            }})
    if len(planned) != 1000 or len({(row["case_id"], row["id"]) for row in planned}) != 1000:
        raise ValueError("selector_plan_not_1000_unique")
    if limit_for_test is not None:
        if not isinstance(limit_for_test, int) or isinstance(limit_for_test, bool) or not 1 <= limit_for_test <= 1000:
            raise ValueError("selector_test_limit")
        return planned[:limit_for_test]
    return planned


def _write_new(path: Path, value: Any) -> None:
    encoded = (_canonical(value) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _append(path: Path, value: Any) -> None:
    encoded = (_canonical(value) + "\n").encode("utf-8")
    with path.open("ab") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())


def _create_empty(path: Path) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.flush()
        os.fsync(target.fileno())


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("selector_journal_missing")
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError("selector_journal_invalid") from None
            if not isinstance(row, dict):
                raise ValueError("selector_journal_invalid")
            output.append(row)
    return output


def _provider(selector: Any) -> Any:
    return getattr(selector, "provider", selector)


def _set_tracer(selector: Any, tracer: TraceRecorder | None) -> None:
    provider = _provider(selector)
    if not hasattr(provider, "tracer"):
        raise ValueError("selector_trace_unsupported")
    provider.tracer = tracer


def _identity(selector: Any) -> tuple[str, str, dict[str, Any] | None]:
    provider = _provider(selector)
    model = getattr(provider, "model", None)
    returned = getattr(provider, "expected_returned_model", model)
    profile = getattr(selector, "resolved_profile", None)
    if profile is None:
        profile = getattr(provider, "resolved_profile", None)
    if not isinstance(model, str) or not model or not isinstance(returned, str) or not returned:
        raise ValueError("selector_identity_missing")
    return model, returned, profile if isinstance(profile, dict) else None


def _bounded_choose(selector: Any, observation: dict[str, Any], instructions: str, timeout: float) -> Any:
    provider = _provider(selector)
    previous = getattr(provider, "timeout_seconds", None)
    if not isinstance(previous, (int, float)) or isinstance(previous, bool) or previous <= 0:
        raise ValueError("selector_timeout_missing")
    provider.timeout_seconds = min(float(previous), max(0.001, timeout))
    try:
        return selector.choose(observation, instructions, list(BINARY_ACTIONS))
    finally:
        provider.timeout_seconds = previous


def _safe_result(result: Any, *, phase: str, trace_id: str, case_id: str | None, candidate_id: str | None) -> dict[str, Any]:
    action = getattr(result, "action", None)
    outcome = getattr(result, "outcome", None)
    if outcome == "ok" and action not in BINARY_ACTIONS:
        outcome, action = "invalid_output", None
    return {
        "phase": phase, "case_id": case_id, "id": candidate_id, "action": action if action in BINARY_ACTIONS else None,
        "outcome": outcome if outcome in {"ok", "invalid_output", "transport_error", "version_mismatch"} else "invalid_output",
        "usage": {"input_tokens": getattr(result, "input_tokens", None), "output_tokens": getattr(result, "output_tokens", None)},
        "latency_ms": getattr(result, "latency_ms", None), "model": getattr(result, "model", None),
        "returned_model": getattr(result, "returned_model", None), "error_code": getattr(result, "error_code", None),
        "trace_id": trace_id,
    }


def _trace_has_lfm26_warmup_evidence(path: Path, trace_id: str) -> bool:
    for row in _rows(path):
        if row.get("event") != "span_end" or row.get("trace_id") != trace_id:
            continue
        output = row.get("output")
        choices = output.get("choices") if isinstance(output, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            message = choice.get("message")
            if (choice.get("finish_reason") == "stop" and isinstance(message, dict)
                    and isinstance(message.get("reasoning"), str) and message["reasoning"].strip()):
                return True
    return False


def _trace_complete(path: Path, trace_id: str) -> bool:
    starts: set[str] = set()
    ends: set[str] = set()
    request_starts: dict[str, Any] = {}
    request_ends: dict[str, Any] = {}
    for row in _rows(path):
        if row.get("trace_id") != trace_id:
            continue
        if row.get("event") == "span_start" and isinstance(row.get("span_id"), str):
            starts.add(row["span_id"])
            if row.get("name") == "controller.request":
                request_starts[row["span_id"]] = row.get("input")
        elif row.get("event") == "span_end" and isinstance(row.get("span_id"), str):
            ends.add(row["span_id"])
            request_ends[row["span_id"]] = row.get("output")
    raw_request_response = any(isinstance(value, dict) and isinstance(request_ends.get(span_id), dict)
                               for span_id, value in request_starts.items())
    return bool(starts) and starts == ends and raw_request_response


def _create_output(path: Path) -> None:
    if path.exists() or not path.parent.is_dir():
        raise ValueError("selector_output_must_be_new_directory")
    os.mkdir(path, 0o700)
    os.chmod(path, 0o700)


def _receipt(output: Path, metadata: dict[str, Any], *, complete: bool, stop_reason: str | None) -> dict[str, Any]:
    journals = {name: _sha256_path(output / name) for name in ("run-metadata.json", "intents.jsonl", "results.jsonl", "traces.jsonl")}
    receipt = {"schema": "averitec-selector-run-receipt/v1", "complete": complete, "stop_reason": stop_reason,
               "run_id": metadata["run_id"], "controller": metadata["controller"], "planned_ids_sha256": metadata["planned_ids_sha256"],
               "journal_sha256": journals}
    _write_new(output / "receipt.json", receipt)
    return receipt


def _install_signal_stop() -> tuple[dict[str, bool], dict[int, Any]]:
    stopped = {"value": False}
    prior: dict[int, Any] = {}
    def handler(_signum: int, _frame: Any) -> None:
        stopped["value"] = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        prior[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return stopped, prior


def _restore_signals(prior: Mapping[int, Any]) -> None:
    for signum, handler in prior.items():
        signal.signal(signum, handler)


def execute(
    controller: str, candidates: Path, launch_manifest: Path, output: Path, *, endpoint: str | None = None,
    laya_checkpoint: str | Path | None = None, keys: Mapping[str, str] | None = None,
    selector_factory: Callable[..., Any] = create_selector, limit_for_test: int | None = None,
) -> dict[str, Any]:
    """Run one arm exactly once.  ``limit_for_test`` is deliberately not a CLI option."""
    if controller not in CONTROLLERS:
        raise ValueError("selector_unknown_controller")
    instructions_path = HERE / "selector_instructions.txt"
    manifest = load_launch(launch_manifest, candidates, instructions_path)
    if manifest["schema"] == SCHEMA_V2:
        v2_factory = create_recovery_selector if selector_factory is create_selector else selector_factory
        return execute_v2(controller, candidates, launch_manifest, output, endpoint=endpoint, keys=keys,
                          selector_factory=v2_factory, limit_for_test=limit_for_test)
    planned = plan_observations(candidates, limit_for_test=limit_for_test)
    if limit_for_test is None and len(planned) != manifest["controllers"][controller]["measured_calls"]:
        raise ValueError("selector_manifest_plan_count")
    _create_output(output)
    instructions = instructions_path.read_text(encoding="utf-8")
    model, profile, wall = CONTROLLERS[controller]
    run_id = hashlib.sha256((_sha256_path(launch_manifest) + controller + _sha256_path(candidates)).encode()).hexdigest()[:32]
    metadata = {"schema": "averitec-selector-run/v1", "run_id": run_id, "controller": controller, "model": model, "profile": profile,
                "source_commit": manifest["source_commit"], "launch_manifest_sha256": _sha256_path(launch_manifest),
                "candidates_sha256": _sha256_path(candidates), "reference_sha256": manifest["reference_sha256"],
                "instructions_sha256": _sha256_path(instructions_path), "planned_calls": len(planned),
                "planned_ids_sha256": hashlib.sha256(_canonical([(r["case_id"], r["id"]) for r in planned]).encode()).hexdigest(),
                "maximum_calls": len(planned) + 1, "wall_seconds": wall, "retries": 0, "started_unix": int(time())}
    _write_new(output / "run-metadata.json", metadata)
    # Create journals before any call, then make the directory fs-visible.
    for name in ("intents.jsonl", "results.jsonl", "traces.jsonl"):
        _create_empty(output / name)
    try:
        selector = selector_factory(controller, endpoint=endpoint, keys=keys, laya_checkpoint=laya_checkpoint)
        actual_model, expected_returned, identity = _identity(selector)
        if actual_model != model:
            raise ValueError("selector_factory_model_mismatch")
    except Exception:
        _append(output / "results.jsonl", {"phase": "startup", "outcome": "runner_error", "error_code": "selector_startup_failed", "trace_id": None})
        return _receipt(output, metadata, complete=False, stop_reason="startup_failure")
    if controller == "laya_typed":
        # Lossless preflight is required for all 1,000 real inputs before the
        # first native forward.  It performs no predict/inference.
        try:
            for row in planned:
                selector.preflight(row["observation"], instructions, list(BINARY_ACTIONS))
        except Exception:
            _append(output / "results.jsonl", {"phase": "preflight", "outcome": "runner_error", "error_code": "laya_preflight_failed", "trace_id": None})
            return _receipt(output, metadata, complete=False, stop_reason="preflight_failure")
    stopped, prior_signals = _install_signal_stop()
    started = monotonic()
    total_calls, stop_reason = 0, None
    try:
        call_plan: Iterable[tuple[str, str | None, str | None, dict[str, Any]]] = [
            ("warmup", None, None, {"claim": "Synthetic selector deployment check.", "candidate": {"id": "synthetic", "text": "This is a synthetic plumbing check.", "url": "https://example.invalid/synthetic"}}),
            *(("measured", row["case_id"], row["id"], row["observation"]) for row in planned),
        ]
        for phase, case_id, candidate_id, observation in call_plan:
            if stopped["value"]:
                stop_reason = "signal_before_call"
                break
            remaining = wall - (monotonic() - started)
            if remaining <= 0:
                stop_reason = "wall_cap"
                break
            trial = phase if phase == "warmup" else case_id + ":" + candidate_id  # type: ignore[operator]
            recorder = TraceRecorder(output / "traces.jsonl", run_id, trial, {"controller": controller, "phase": phase})
            intent = {"phase": phase, "case_id": case_id, "id": candidate_id, "trace_id": recorder.trace_id, "ordinal": total_calls + 1}
            _append(output / "intents.jsonl", intent)  # Durable before the request/forward.
            total_calls += 1
            _set_tracer(selector, recorder)
            try:
                with recorder.span("selector." + phase, input={"case_id": case_id, "id": candidate_id}, model=model) as span:
                    result = _bounded_choose(selector, observation, instructions, remaining)
                    row = _safe_result(result, phase=phase, trace_id=recorder.trace_id, case_id=case_id, candidate_id=candidate_id)
                    span.update(output={key: row[key] for key in ("action", "outcome", "model", "returned_model", "error_code")}, usage=row["usage"])
            except KeyboardInterrupt:
                row = {"phase": phase, "case_id": case_id, "id": candidate_id, "action": None, "outcome": "interrupted_unknown_outcome", "usage": {"input_tokens": None, "output_tokens": None}, "latency_ms": None, "model": model, "returned_model": None, "error_code": "signal_during_call", "trace_id": recorder.trace_id}
                stop_reason = "signal_during_call"
            except Exception:
                row = {"phase": phase, "case_id": case_id, "id": candidate_id, "action": None, "outcome": "runner_error", "usage": {"input_tokens": None, "output_tokens": None}, "latency_ms": None, "model": model, "returned_model": None, "error_code": "unexpected_exception", "trace_id": recorder.trace_id}
                stop_reason = "unexpected_exception"
            _append(output / "results.jsonl", row)
            if stop_reason:
                break
            if row["model"] != model or row["returned_model"] != expected_returned or row["outcome"] in {"transport_error", "version_mismatch"}:
                stop_reason = "identity_or_transport_failure"
                break
            if phase == "warmup":
                valid_warmup = row["outcome"] == "ok" and row["action"] in BINARY_ACTIONS and all(isinstance(row["usage"].get(key), int) and not isinstance(row["usage"][key], bool) and row["usage"][key] >= 0 for key in ("input_tokens", "output_tokens"))
                if controller == "lfm26":
                    valid_warmup = valid_warmup and _trace_has_lfm26_warmup_evidence(output / "traces.jsonl", recorder.trace_id)
                if not valid_warmup:
                    stop_reason = "warmup_failure"
                    break
    finally:
        _restore_signals(prior_signals)
    complete = stop_reason is None and total_calls == len(planned) + 1
    return _receipt(output, metadata, complete=complete, stop_reason=stop_reason)


def _trace_model_identity(path: Path, trace_id: str, controller: str) -> str | None:
    """Read only the raw response model recorded in the matching request span."""
    request_spans: set[str] = set()
    for row in _rows(path):
        if row.get("trace_id") == trace_id and row.get("event") == "span_start" and row.get("name") == "controller.request" and isinstance(row.get("span_id"), str):
            request_spans.add(row["span_id"])
    for row in _rows(path):
        if row.get("trace_id") == trace_id and row.get("event") == "span_end" and row.get("span_id") in request_spans:
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            value = output.get("modelVersion") if controller == "gemini" else output.get("model")
            if isinstance(value, str):
                return value
    return None


def _v2_receipt(output: Path, metadata: dict[str, Any], *, complete: bool, all_valid: bool,
                outcome_counts: dict[str, int], stop_reason: str | None) -> dict[str, Any]:
    journals = {name: _sha256_path(output / name) for name in ("run-metadata.json", "intents.jsonl", "results.jsonl", "traces.jsonl")}
    receipt = {"schema": "averitec-selector-run-receipt/v2", "complete": complete, "all_valid": all_valid,
               "outcome_counts": dict(sorted(outcome_counts.items())), "stop_reason": stop_reason,
               "run_id": metadata["run_id"], "controller": metadata["controller"],
               "planned_ids_sha256": metadata["planned_ids_sha256"], "journal_sha256": journals}
    _write_new(output / "receipt.json", receipt)
    return receipt


def execute_v2(
    controller: str, candidates: Path, launch_manifest: Path, output: Path, *, endpoint: str | None = None,
    keys: Mapping[str, str] | None = None, selector_factory: Callable[..., Any] = create_recovery_selector,
    limit_for_test: int | None = None,
) -> dict[str, Any]:
    """Independent v2 rerun: exactly one call per planned item, even failures."""
    if controller not in CONTROLLERS_V2:
        raise ValueError("selector_v2_unknown_controller")
    instructions_path = HERE / "selector_instructions.txt"
    manifest = load_launch(launch_manifest, candidates, instructions_path)
    if manifest["schema"] != SCHEMA_V2:
        raise ValueError("selector_v2_manifest_required")
    planned = plan_observations(candidates, limit_for_test=limit_for_test)
    if limit_for_test is None and len(planned) != 1000:
        raise ValueError("selector_v2_plan_count")
    model, profile_name, wall = CONTROLLERS_V2[controller]
    try:
        selector = selector_factory(controller, endpoint=endpoint, keys=keys)
        actual_model, expected_returned, profile = _identity(selector)
        if actual_model != model or expected_returned != EXPECTED_RETURNED_MODELS_V2[controller] or not isinstance(profile, dict) or profile.get("name") != profile_name:
            raise ValueError("selector_v2_factory_identity")
    except Exception:
        # No directory is claimed if the adapter cannot be constructed; a
        # caller can resolve a local configuration error without consuming an
        # arm or creating ambiguous journals.
        raise ValueError("selector_v2_startup_failed") from None
    _create_output(output)
    instructions = instructions_path.read_text(encoding="utf-8")
    run_id = hashlib.sha256((_sha256_path(launch_manifest) + controller + _sha256_path(candidates) + "v2").encode()).hexdigest()[:32]
    metadata = {"schema": "averitec-selector-run/v2", "run_id": run_id, "controller": controller, "model": model,
                "profile": profile_name, "profile_identity": profile, "source_commit": manifest["source_commit"],
                "launch_manifest_sha256": _sha256_path(launch_manifest), "candidates_sha256": _sha256_path(candidates),
                "reference_sha256": manifest["reference_sha256"], "instructions_sha256": _sha256_path(instructions_path),
                "planned_calls": len(planned), "planned_ids_sha256": hashlib.sha256(_canonical([(r["case_id"], r["id"]) for r in planned]).encode()).hexdigest(),
                "maximum_calls": len(planned) + 1, "wall_seconds": wall, "retries": 0, "started_unix": int(time())}
    _write_new(output / "run-metadata.json", metadata)
    for name in ("intents.jsonl", "results.jsonl", "traces.jsonl"):
        _create_empty(output / name)
    stopped, prior_signals = _install_signal_stop()
    started, total_calls, stop_reason = monotonic(), 0, None
    outcome_counts: dict[str, int] = {}
    try:
        call_plan: Iterable[tuple[str, str | None, str | None, dict[str, Any]]] = [
            ("warmup", None, None, {"claim": "Synthetic selector deployment check.", "candidate": {"id": "synthetic", "text": "This is a synthetic plumbing check.", "url": "https://example.invalid/synthetic"}}),
            *(("measured", row["case_id"], row["id"], row["observation"]) for row in planned),
        ]
        for phase, case_id, candidate_id, observation in call_plan:
            if stopped["value"]:
                stop_reason = "signal_before_call"
                break
            remaining = wall - (monotonic() - started)
            if remaining <= 0:
                stop_reason = "wall_cap"
                break
            trial = phase if phase == "warmup" else case_id + ":" + candidate_id  # type: ignore[operator]
            recorder = TraceRecorder(output / "traces.jsonl", run_id, trial, {"controller": controller, "phase": phase, "protocol": "v2"})
            _append(output / "intents.jsonl", {"phase": phase, "case_id": case_id, "id": candidate_id, "trace_id": recorder.trace_id, "ordinal": total_calls + 1})
            total_calls += 1
            _set_tracer(selector, recorder)
            try:
                if hasattr(selector, "set_phase"):
                    selector.set_phase(phase)
                with recorder.span("selector." + phase, input={"case_id": case_id, "id": candidate_id}, model=model) as span:
                    result = _bounded_choose(selector, observation, instructions, remaining)
                    row = _safe_result(result, phase=phase, trace_id=recorder.trace_id, case_id=case_id, candidate_id=candidate_id)
                    span.update(output={key: row[key] for key in ("action", "outcome", "model", "returned_model", "error_code")}, usage=row["usage"])
            except KeyboardInterrupt:
                row = {"phase": phase, "case_id": case_id, "id": candidate_id, "action": None, "outcome": "runner_error", "usage": {"input_tokens": None, "output_tokens": None}, "latency_ms": None, "model": model, "returned_model": None, "error_code": "signal_during_call", "trace_id": recorder.trace_id}
                stop_reason = "signal_during_call"
            except Exception:
                row = {"phase": phase, "case_id": case_id, "id": candidate_id, "action": None, "outcome": "runner_error", "usage": {"input_tokens": None, "output_tokens": None}, "latency_ms": None, "model": model, "returned_model": None, "error_code": "unexpected_exception", "trace_id": recorder.trace_id}
                stop_reason = "unexpected_exception"
            raw_model = _trace_model_identity(output / "traces.jsonl", recorder.trace_id, controller)
            row["trace_returned_model"] = raw_model
            if raw_model is None:
                row["identity_status"] = "unknown_response"
            elif raw_model == expected_returned:
                row["identity_status"] = "confirmed_trace"
            else:
                row["identity_status"] = "mismatch_trace"
            _append(output / "results.jsonl", row)
            outcome_counts[row["outcome"]] = outcome_counts.get(row["outcome"], 0) + 1
            if stop_reason:
                break
            if row["model"] != model or row["outcome"] == "version_mismatch" or row["identity_status"] == "mismatch_trace":
                stop_reason = "identity_failure"
                break
            valid_usage = all(isinstance(row["usage"].get(key), int) and not isinstance(row["usage"][key], bool) and row["usage"][key] >= 0 for key in ("input_tokens", "output_tokens"))
            if phase == "warmup":
                if (row["outcome"] != "ok" or row["action"] not in BINARY_ACTIONS or not valid_usage
                        or row["returned_model"] != expected_returned
                        or (controller == "lfm26" and not _trace_has_lfm26_warmup_evidence(output / "traces.jsonl", recorder.trace_id))):
                    stop_reason = "warmup_failure"
                    break
            # Measured invalid/transport outcomes are explicitly retained and
            # continue to the next distinct planned item; no reissue occurs.
    finally:
        _restore_signals(prior_signals)
    complete = stop_reason is None and total_calls == len(planned) + 1
    all_valid = complete and outcome_counts == {"ok": len(planned) + 1}
    return _v2_receipt(output, metadata, complete=complete, all_valid=all_valid,
                       outcome_counts=outcome_counts, stop_reason=stop_reason)


def verify_receipt_v2(output: Path, candidates: Path, launch_manifest: Path) -> dict[str, Any]:
    """Fail closed on journal/identity drift while allowing recorded model failures."""
    manifest = load_launch(launch_manifest, candidates)
    if manifest["schema"] != SCHEMA_V2:
        raise ValueError("selector_v2_manifest_required")
    metadata, receipt = _load_json(output / "run-metadata.json"), _load_json(output / "receipt.json")
    metadata_keys = {"schema", "run_id", "controller", "model", "profile", "profile_identity", "source_commit", "launch_manifest_sha256", "candidates_sha256", "reference_sha256", "instructions_sha256", "planned_calls", "planned_ids_sha256", "maximum_calls", "wall_seconds", "retries", "started_unix"}
    receipt_keys = {"schema", "complete", "all_valid", "outcome_counts", "stop_reason", "run_id", "controller", "planned_ids_sha256", "journal_sha256"}
    if (not isinstance(metadata, dict) or set(metadata) != metadata_keys or metadata.get("schema") != "averitec-selector-run/v2"
            or not isinstance(receipt, dict) or set(receipt) != receipt_keys or receipt.get("schema") != "averitec-selector-run-receipt/v2"
            or receipt.get("complete") is not True or receipt.get("stop_reason") is not None):
        raise ValueError("selector_v2_receipt_incomplete")
    controller = metadata.get("controller")
    if controller not in CONTROLLERS_V2 or metadata.get("planned_calls") != 1000 or metadata.get("maximum_calls") != 1001:
        raise ValueError("selector_v2_metadata")
    model, profile, wall = CONTROLLERS_V2[controller]
    if (metadata.get("model") != model or metadata.get("profile") != profile or metadata.get("wall_seconds") != wall
            or metadata.get("profile_identity") != RECOVERY_PROFILE_IDENTITIES[controller]):
        raise ValueError("selector_v2_profile")
    if (metadata.get("source_commit") != manifest["source_commit"] or metadata.get("reference_sha256") != manifest["reference_sha256"]
            or metadata.get("instructions_sha256") != manifest["instructions_sha256"] or metadata.get("retries") != 0
            or metadata.get("launch_manifest_sha256") != _sha256_path(launch_manifest) or metadata.get("candidates_sha256") != _sha256_path(candidates)):
        raise ValueError("selector_v2_input_binding")
    hashes = receipt.get("journal_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {"run-metadata.json", "intents.jsonl", "results.jsonl", "traces.jsonl"}:
        raise ValueError("selector_v2_journal_hashes")
    for name, digest in hashes.items():
        if digest != _sha256_path(output / name):
            raise ValueError("selector_v2_journal_drift")
    planned = plan_observations(candidates)
    expected_pairs = [(row["case_id"], row["id"]) for row in planned]
    expected_hash = hashlib.sha256(_canonical(expected_pairs).encode()).hexdigest()
    if metadata.get("planned_ids_sha256") != expected_hash or receipt.get("planned_ids_sha256") != expected_hash or receipt.get("run_id") != metadata.get("run_id") or receipt.get("controller") != controller:
        raise ValueError("selector_v2_plan_binding")
    intents, results = _rows(output / "intents.jsonl"), _rows(output / "results.jsonl")
    if len(intents) != 1001 or len(results) != 1001 or intents[0].get("phase") != "warmup" or results[0].get("phase") != "warmup":
        raise ValueError("selector_v2_call_count")
    if [(row.get("case_id"), row.get("id")) for row in intents[1:]] != expected_pairs:
        raise ValueError("selector_v2_planned_ids")
    trace_ids = [row.get("trace_id") for row in intents]
    if len(set(trace_ids)) != 1001 or any(not isinstance(value, str) for value in trace_ids):
        raise ValueError("selector_v2_trace_ids")
    valid_outcomes = {"ok", "invalid_output", "transport_error"}
    counts: dict[str, int] = {}
    expected_returned = EXPECTED_RETURNED_MODELS_V2[controller]
    for ordinal, (intent, result) in enumerate(zip(intents, results), 1):
        if intent.get("ordinal") != ordinal or (intent.get("phase"), intent.get("case_id"), intent.get("id"), intent.get("trace_id")) != (result.get("phase"), result.get("case_id"), result.get("id"), result.get("trace_id")):
            raise ValueError("selector_v2_intent_result_join")
        if result.get("model") != model or result.get("outcome") not in valid_outcomes or not _trace_complete_v2(output / "traces.jsonl", result.get("trace_id")):
            raise ValueError("selector_v2_trace_or_outcome")
        raw_model = _trace_model_identity(output / "traces.jsonl", result["trace_id"], controller)
        if raw_model is not None and raw_model != expected_returned:
            raise ValueError("selector_v2_trace_identity")
        if result.get("trace_returned_model") != raw_model:
            raise ValueError("selector_v2_trace_binding")
        if result["outcome"] == "ok" and (result.get("action") not in BINARY_ACTIONS or result.get("returned_model") != expected_returned or raw_model != expected_returned):
            raise ValueError("selector_v2_valid_identity")
        if result["outcome"] == "invalid_output" and result.get("error_code") == "truncated_or_invalid" and raw_model != expected_returned:
            raise ValueError("selector_v2_truncation_identity")
        counts[result["outcome"]] = counts.get(result["outcome"], 0) + 1
    warmup = results[0]
    usage = warmup.get("usage")
    if (warmup.get("outcome") != "ok" or warmup.get("returned_model") != expected_returned
            or not isinstance(usage, dict)
            or any(not isinstance(usage.get(key), int) or isinstance(usage[key], bool) or usage[key] < 0 for key in ("input_tokens", "output_tokens"))
            or _trace_model_identity(output / "traces.jsonl", warmup["trace_id"], controller) != expected_returned
            or (controller == "lfm26" and not _trace_has_lfm26_warmup_evidence(output / "traces.jsonl", warmup["trace_id"]))):
        raise ValueError("selector_v2_warmup")
    if receipt.get("outcome_counts") != dict(sorted(counts.items())) or receipt.get("all_valid") is not (counts == {"ok": 1001}):
        raise ValueError("selector_v2_outcome_counts")
    return receipt


def _trace_complete_v2(path: Path, trace_id: Any) -> bool:
    if not isinstance(trace_id, str):
        return False
    starts: set[str] = set()
    ends: set[str] = set()
    request = False
    for row in _rows(path):
        if row.get("trace_id") != trace_id:
            continue
        if row.get("event") == "span_start" and isinstance(row.get("span_id"), str):
            starts.add(row["span_id"])
            request = request or (row.get("name") == "controller.request" and isinstance(row.get("input"), dict))
        elif row.get("event") == "span_end" and isinstance(row.get("span_id"), str):
            ends.add(row["span_id"])
    return bool(starts) and starts == ends and request


def verify_receipt(output: Path, candidates: Path, launch_manifest: Path) -> dict[str, Any]:
    """Offline, fail-closed verification of a completed run; never runs inference."""
    manifest = load_launch(launch_manifest, candidates)
    if manifest["schema"] == SCHEMA_V2:
        return verify_receipt_v2(output, candidates, launch_manifest)
    metadata = _load_json(output / "run-metadata.json")
    receipt = _load_json(output / "receipt.json")
    metadata_keys = {"schema", "run_id", "controller", "model", "profile", "source_commit", "launch_manifest_sha256", "candidates_sha256", "reference_sha256", "instructions_sha256", "planned_calls", "planned_ids_sha256", "maximum_calls", "wall_seconds", "retries", "started_unix"}
    receipt_keys = {"schema", "complete", "stop_reason", "run_id", "controller", "planned_ids_sha256", "journal_sha256"}
    if (not isinstance(metadata, dict) or set(metadata) != metadata_keys
            or not isinstance(receipt, dict) or set(receipt) != receipt_keys
            or receipt.get("schema") != "averitec-selector-run-receipt/v1" or receipt.get("complete") is not True
            or receipt.get("stop_reason") is not None):
        raise ValueError("selector_receipt_incomplete")
    controller = metadata.get("controller")
    if controller not in CONTROLLERS or metadata.get("planned_calls") != 1000 or metadata.get("maximum_calls") != 1001:
        raise ValueError("selector_receipt_metadata")
    model, profile, wall = CONTROLLERS[controller]
    expected_returned = EXPECTED_RETURNED_MODELS[controller]
    if metadata.get("model") != model or metadata.get("profile") != profile or metadata.get("wall_seconds") != wall:
        raise ValueError("selector_receipt_identity")
    if (metadata.get("source_commit") != manifest["source_commit"] or metadata.get("reference_sha256") != manifest["reference_sha256"]
            or metadata.get("instructions_sha256") != manifest["instructions_sha256"] or metadata.get("retries") != 0
            or metadata.get("launch_manifest_sha256") != _sha256_path(launch_manifest)
            or metadata.get("candidates_sha256") != _sha256_path(candidates)):
        raise ValueError("selector_receipt_input_hash")
    planned = plan_observations(candidates)
    expected_ids = [(row["case_id"], row["id"]) for row in planned]
    if metadata.get("planned_ids_sha256") != hashlib.sha256(_canonical(expected_ids).encode()).hexdigest():
        raise ValueError("selector_receipt_plan_hash")
    if receipt.get("run_id") != metadata["run_id"] or receipt.get("controller") != controller or receipt.get("planned_ids_sha256") != metadata["planned_ids_sha256"]:
        raise ValueError("selector_receipt_binding")
    hashes = receipt.get("journal_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {"run-metadata.json", "intents.jsonl", "results.jsonl", "traces.jsonl"}:
        raise ValueError("selector_receipt_journal_hashes")
    for name, expected in hashes.items():
        if expected != _sha256_path(output / name):
            raise ValueError("selector_receipt_journal_drift")
    intents, results = _rows(output / "intents.jsonl"), _rows(output / "results.jsonl")
    if len(intents) != 1001 or len(results) != 1001:
        raise ValueError("selector_receipt_call_count")
    if intents[0].get("phase") != "warmup" or results[0].get("phase") != "warmup":
        raise ValueError("selector_receipt_warmup")
    pairs = [(row.get("case_id"), row.get("id")) for row in intents[1:]]
    if pairs != expected_ids or len(set(pairs)) != 1000:
        raise ValueError("selector_receipt_planned_ids")
    intent_trace_ids = [row.get("trace_id") for row in intents]
    if len(set(intent_trace_ids)) != 1001 or any(not isinstance(item, str) for item in intent_trace_ids):
        raise ValueError("selector_receipt_intents")
    for ordinal, (intent, result) in enumerate(zip(intents, results), 1):
        if intent.get("ordinal") != ordinal:
            raise ValueError("selector_receipt_intent_order")
        if (intent.get("phase"), intent.get("case_id"), intent.get("id"), intent.get("trace_id")) != (result.get("phase"), result.get("case_id"), result.get("id"), result.get("trace_id")):
            raise ValueError("selector_receipt_intent_result_join")
        if result.get("model") != model or result.get("returned_model") != expected_returned or result.get("outcome") != "ok" or result.get("action") not in BINARY_ACTIONS:
            raise ValueError("selector_receipt_model_outcome")
        if not _trace_complete(output / "traces.jsonl", result["trace_id"]):
            raise ValueError("selector_receipt_trace_join")
    warmup = results[0]
    usage = warmup.get("usage")
    if (not isinstance(usage, dict)
            or any(not isinstance(usage.get(key), int) or isinstance(usage[key], bool) or usage[key] < 0
                   for key in ("input_tokens", "output_tokens"))):
        raise ValueError("selector_receipt_warmup_usage")
    if controller == "lfm26" and not _trace_has_lfm26_warmup_evidence(output / "traces.jsonl", warmup["trace_id"]):
        raise ValueError("selector_receipt_lfm26_warmup")
    if manifest["controllers"][controller]["maximum_calls"] != 1001:
        raise ValueError("selector_receipt_cap")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--controller", choices=tuple(CONTROLLERS), required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--laya-checkpoint", type=Path)
    args = parser.parse_args()
    if args.plan:
        load_launch(args.launch_manifest, args.candidates)
        plan = plan_observations(args.candidates)
        print(json.dumps({"controller": args.controller, "planned_calls": len(plan), "planned_ids_sha256": hashlib.sha256(_canonical([(r["case_id"], r["id"]) for r in plan]).encode()).hexdigest()}, sort_keys=True))
    elif args.verify:
        print(json.dumps(verify_receipt(args.output, args.candidates, args.launch_manifest), sort_keys=True))
    else:
        receipt = execute(args.controller, args.candidates, args.launch_manifest, args.output, endpoint=args.endpoint, laya_checkpoint=args.laya_checkpoint)
        print(json.dumps(receipt, sort_keys=True))
        if receipt["complete"] is not True:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
