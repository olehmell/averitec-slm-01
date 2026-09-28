"""Standalone, bounded Jeff evidence-selection runner.

This is deliberately separate from ``run_selector.py``: it only accepts the
sealed, gold-free candidate file and has no recovery or resume path.  A launch
always consists of one synthetic warm-up followed by the fixed 1,000 measured
candidate decisions (at most 1,001 native forwards total).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic, time
from collections.abc import Mapping
from typing import Any, Callable, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
EXPERIMENT_ROOT = HERE.parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
SCHEMA = "averitec-jeff-launch/v1"
ASSET_SCHEMA = "averitec-jeff-native-assets/v1"
RUN_SCHEMA = "averitec-jeff-run/v1"
RECEIPT_SCHEMA = "averitec-jeff-run-receipt/v1"
MODEL = "knowledgator/gliformer-large-v1"
RETURNED_MODEL = "gliformer-large-v1"
UPSTREAM_COMMIT = "34b32f99a727c47b679adde33f4702a001e02979"
MODEL_REVISION = "d0a4e53d09cebe6bc963dd9be319d4279084bb2d"
WARMUP_CALLS, MEASURED_CALLS, MAXIMUM_CALLS = 1, 1000, 1001
GPU_ALLOCATION_SECONDS, WALL_SECONDS, RETRIES = 1740, 1640, 0
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_CODE = frozenset({
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/jeff_runner.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/jeff_adapter.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/jeff_native.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_runtime.py",
    "experiments/orchestration/averitec-controller-reliability-2026/tracing.py",
    "experiments/orchestration/averitec-controller-reliability-2026/providers.py",
    "experiments/orchestration/averitec-controller-reliability-2026/gemini_provider.py",
    "experiments/orchestration/averitec-controller-reliability-2026/generation_profiles.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/gpu/jeff.slurm",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/jeff-plan.json",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, error: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(error) from None


def _head_commit() -> str:
    marker = REPO_ROOT / "SOURCE_COMMIT"
    if marker.is_file():
        try:
            value = marker.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            raise ValueError("jeff_source_commit_unavailable") from None
        if not HEX40.fullmatch(value):
            raise ValueError("jeff_source_commit_invalid")
        return value
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("jeff_source_commit_unavailable") from None
    if not HEX40.fullmatch(value):
        raise ValueError("jeff_source_commit_invalid")
    return value


def _relative_code_path(name: Any) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("jeff_manifest_code_path")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("jeff_manifest_code_path")
    return path


def _validate_candidates(value: Any) -> list[dict[str, Any]]:
    """Standard-library-only validation; no label/reference file is accepted."""
    root = {"schema", "source_freeze_sha256", "source_traces_sha256", "gold_included", "cases"}
    if not isinstance(value, dict) or set(value) != root or value.get("schema") != "averitec-frozen-evidence-candidates/v1" or value.get("gold_included") is not False:
        raise ValueError("jeff_candidates_not_gold_free")
    if not all(isinstance(value.get(k), str) and HEX64.fullmatch(value[k]) for k in ("source_freeze_sha256", "source_traces_sha256")):
        raise ValueError("jeff_candidates_source_hash")
    planned: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for case in value.get("cases", []):
        if not isinstance(case, dict) or set(case) != {"case_id", "group_id", "claim", "candidates"}:
            raise ValueError("jeff_candidates_schema")
        case_id, claim, candidates = case.get("case_id"), case.get("claim"), case.get("candidates")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids or not isinstance(claim, str) or not claim.strip() or not isinstance(candidates, list) or len(candidates) != 10:
            raise ValueError("jeff_candidates_cases")
        case_ids.add(case_id)
        passage_ids: set[str] = set()
        for ordinal, candidate in enumerate(candidates, 1):
            keys = {"id", "passage_id", "text", "url", "source_start", "source_text_length", "source_text_sha256"}
            if not isinstance(candidate, dict) or set(candidate) != keys or candidate.get("id") != f"C{ordinal:02d}":
                raise ValueError("jeff_candidates_candidate")
            if (not isinstance(candidate.get("passage_id"), str) or candidate["passage_id"] in passage_ids
                    or not isinstance(candidate.get("text"), str) or not candidate["text"].strip()
                    or not isinstance(candidate.get("url"), str) or not candidate["url"]
                    or type(candidate.get("source_start")) is not int or candidate["source_start"] < 0
                    or type(candidate.get("source_text_length")) is not int or candidate["source_text_length"] < 0
                    or not isinstance(candidate.get("source_text_sha256"), str) or not HEX64.fullmatch(candidate["source_text_sha256"])):
                raise ValueError("jeff_candidates_candidate")
            passage_ids.add(candidate["passage_id"])
            planned.append({"case_id": case_id, "id": candidate["id"], "observation": {
                "claim": claim, "candidate": {k: candidate[k] for k in ("id", "text", "url")},
            }})
    if len(case_ids) != 100 or len(planned) != MEASURED_CALLS or len({(r["case_id"], r["id"]) for r in planned}) != MEASURED_CALLS:
        raise ValueError("jeff_candidates_not_1000_unique")
    return planned


def plan_observations(candidates: Path) -> list[dict[str, Any]]:
    if not candidates.is_file():
        raise ValueError("jeff_candidates_missing")
    return _validate_candidates(_load_json(candidates, "jeff_candidates_json"))


def _validate_assets(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    value = _load_json(path, "jeff_assets_json")
    required = {"schema", "upstream_commit", "model", "model_revision", "checkpoint_sha256", "base_image_sha256", "files"}
    if not isinstance(value, dict) or set(value) != required or value.get("schema") != ASSET_SCHEMA:
        raise ValueError("jeff_assets_shape")
    if (value.get("upstream_commit") != manifest["upstream_commit"] or value.get("model") != manifest["model"]
            or value.get("model_revision") != manifest["model_revision"] or value.get("checkpoint_sha256") != manifest["checkpoint_sha256"]):
        raise ValueError("jeff_assets_identity_drift")
    if not isinstance(value["checkpoint_sha256"], str) or not HEX64.fullmatch(value["checkpoint_sha256"]):
        raise ValueError("jeff_assets_checkpoint_hash")
    if not isinstance(value["base_image_sha256"], str) or not HEX64.fullmatch(value["base_image_sha256"]):
        raise ValueError("jeff_assets_image_hash")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("jeff_assets_files")
    checkpoint: dict[str, str] = {}
    for name, digest in files.items():
        relative = _relative_code_path(name)
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ValueError("jeff_assets_file_hash")
        if str(relative).startswith("checkpoint/"):
            checkpoint[str(relative)] = digest
    if not checkpoint or hashlib.sha256(_canonical(checkpoint).encode()).hexdigest() != value["checkpoint_sha256"]:
        raise ValueError("jeff_assets_checkpoint_binding")
    return value


def load_launch(manifest_path: Path, candidates: Path, instructions: Path, assets: Path) -> dict[str, Any]:
    value = _load_json(manifest_path, "jeff_manifest_json")
    keys = {"schema", "source_commit", "candidates_sha256", "instructions_sha256", "asset_manifest_sha256", "code_sha256",
            "upstream_commit", "model", "returned_model", "model_revision", "checkpoint_sha256", "warmup_calls",
            "measured_calls", "maximum_calls", "retries", "gpu_allocation_seconds", "wall_seconds"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != SCHEMA:
        raise ValueError("jeff_manifest_shape")
    if (value.get("source_commit") != _head_commit() or value.get("upstream_commit") != UPSTREAM_COMMIT
            or value.get("model") != MODEL or value.get("returned_model") != RETURNED_MODEL or value.get("model_revision") != MODEL_REVISION
            or value.get("warmup_calls") != WARMUP_CALLS or value.get("measured_calls") != MEASURED_CALLS
            or value.get("maximum_calls") != MAXIMUM_CALLS or value.get("retries") != RETRIES
            or value.get("gpu_allocation_seconds") != GPU_ALLOCATION_SECONDS or value.get("wall_seconds") != WALL_SECONDS):
        raise ValueError("jeff_manifest_policy")
    for name in ("candidates_sha256", "instructions_sha256", "asset_manifest_sha256", "checkpoint_sha256"):
        if not isinstance(value.get(name), str) or not HEX64.fullmatch(value[name]):
            raise ValueError("jeff_manifest_hash")
    if _sha256(candidates) != value["candidates_sha256"] or _sha256(instructions) != value["instructions_sha256"] or _sha256(assets) != value["asset_manifest_sha256"]:
        raise ValueError("jeff_manifest_input_drift")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != REQUIRED_CODE:
        raise ValueError("jeff_manifest_code_shape")
    for name, expected in code.items():
        path = REPO_ROOT / _relative_code_path(name)
        if not isinstance(expected, str) or not HEX64.fullmatch(expected) or not path.is_file() or _sha256(path) != expected:
            raise ValueError("jeff_manifest_code_drift")
    _validate_assets(assets, value)
    plan_observations(candidates)
    return value


def _request(observation: dict[str, Any], instructions: str) -> dict[str, Any]:
    return {"model": RETURNED_MODEL, "state": observation, "questions": {"next_action": {
        "type": "choice", "instructions": instructions, "criteria": {"include": None, "exclude": None},
    }}}


def _write_new(path: Path, value: Any) -> None:
    encoded = (_canonical(value) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())


def _append(path: Path, value: Any) -> None:
    with path.open("ab") as target:
        target.write((_canonical(value) + "\n").encode("utf-8"))
        target.flush()
        os.fsync(target.fileno())


def _empty_new(path: Path) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.flush()
        os.fsync(target.fileno())


def _replace_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ValueError("jeff_atomic_temporary_exists")
    _write_new(temporary, value)
    os.replace(temporary, path)


def _create_output(output: Path) -> None:
    if output.exists() or not output.parent.is_dir():
        raise ValueError("jeff_output_must_be_new_directory")
    os.mkdir(output, 0o700)
    os.chmod(output, 0o700)


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("jeff_journal_missing")
    result: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError
            result.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("jeff_journal_invalid") from None
    return result


def _normalise_runtime_identity(identity: Any) -> dict[str, str]:
    """Accept the native evidence envelope while comparing only sealed facts."""
    if not isinstance(identity, Mapping):
        raise ValueError("jeff_runtime_identity_missing")
    upstream = identity.get("upstream")
    checkpoint = identity.get("checkpoint")
    commit = identity.get("upstream_commit")
    if commit is None and isinstance(upstream, Mapping):
        commit = upstream.get("commit")
    checkpoint_hash = identity.get("checkpoint_sha256")
    if checkpoint_hash is None and isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("files_sha256"), Mapping):
        files = {"checkpoint/" + str(name): digest for name, digest in checkpoint["files_sha256"].items()}
        checkpoint_hash = hashlib.sha256(_canonical(files).encode()).hexdigest()
    normalized = {"upstream_commit": commit, "model": identity.get("model"), "returned_model": identity.get("returned_model"),
                  "model_revision": identity.get("model_revision"), "checkpoint_sha256": checkpoint_hash}
    if any(not isinstance(value, str) for value in normalized.values()):
        raise ValueError("jeff_runtime_identity_missing")
    return normalized  # type: ignore[return-value]


def _runtime_identity(runtime: Any, manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    identity = getattr(runtime, "identity", None)
    if callable(identity):
        identity = identity()
    normalized = _normalise_runtime_identity(identity)
    if any(normalized[key] != manifest[key] for key in normalized):
        raise ValueError("jeff_runtime_identity_drift")
    return normalized, dict(identity)


def _load_runtime(checkpoint: Path, runtime_factory: Callable[..., Any] | None) -> Any:
    if runtime_factory is not None:
        return runtime_factory(checkpoint=checkpoint, device="cuda")
    module = importlib.import_module("jeff_native")
    factory = getattr(module, "create_runtime", None)
    if not callable(factory):
        raise ValueError("jeff_native_create_runtime_missing")
    return factory(checkpoint=checkpoint, device="cuda")


def _validate_checkpoint_tree(checkpoint: Path, assets: dict[str, Any]) -> None:
    """Hash every staged asset before imports/runtime construction."""
    if not checkpoint.is_dir():
        raise ValueError("jeff_checkpoint_missing")
    root = checkpoint.parent
    expected = dict(assets["files"])
    actual: dict[str, str] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("jeff_asset_symlink_forbidden")
        if path.is_file():
            actual[str(relative).replace(os.sep, "/")] = _sha256(path)
    if actual != expected:
        raise ValueError("jeff_asset_tree_drift")


def _load_adapter_factory(adapter_factory: Callable[..., Any] | None) -> Callable[..., Any]:
    if adapter_factory is not None:
        return adapter_factory
    return importlib.import_module("jeff_adapter").JeffEvidenceSelector


def _trace_events(path: Path, trace_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    starts = [r for r in _rows(path) if r.get("event") == "span_start" and r.get("trace_id") == trace_id and r.get("name") == "controller.request"]
    ends = [r for r in _rows(path) if r.get("event") == "span_end" and r.get("trace_id") == trace_id]
    if len(starts) != 1 or len(ends) != 1 or starts[0].get("span_id") != ends[0].get("span_id"):
        return None
    return starts[0], ends[0]


def _result_from_adapter(result: Any, *, phase: str, case_id: str | None, candidate_id: str | None, trace_id: str, request: dict[str, Any]) -> dict[str, Any]:
    action, outcome = getattr(result, "action", None), getattr(result, "outcome", None)
    if outcome not in {"ok", "invalid_output", "transport_error", "version_mismatch"}:
        outcome = "invalid_output"
    if outcome != "ok" or action not in {"include", "exclude"}:
        action = None
    return {"phase": phase, "case_id": case_id, "id": candidate_id, "action": action, "outcome": outcome,
            "error_code": getattr(result, "error_code", None), "model": getattr(result, "model", None),
            "returned_model": getattr(result, "returned_model", None), "usage": {
                "input_tokens": getattr(result, "input_tokens", None), "output_tokens": getattr(result, "output_tokens", None)},
            "latency_ms": getattr(result, "latency_ms", None), "trace_id": trace_id,
            "request_sha256": hashlib.sha256(_canonical(request).encode()).hexdigest()}


def _receipt(output: Path, metadata: dict[str, Any], *, complete: bool, all_valid: bool, stop_reason: str | None) -> dict[str, Any]:
    names = ("run-metadata.json", "preflight.json", "intents.jsonl", "results.jsonl", "traces.jsonl", "scored-results.jsonl")
    journals = {name: _sha256(output / name) for name in names if (output / name).is_file()}
    value = {"schema": RECEIPT_SCHEMA, "run_id": metadata["run_id"], "complete": complete, "all_valid": all_valid,
             "stop_reason": stop_reason, "planned_ids_sha256": metadata["planned_ids_sha256"], "journal_sha256": journals}
    _write_new(output / "receipt.json", value)
    return value


def execute(candidates: Path, launch_manifest: Path, output: Path, assets: Path, checkpoint: Path, *,
            runtime_factory: Callable[..., Any] | None = None, adapter_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Execute one non-resumable run.  Any non-``ok`` outcome stops the arm."""
    instructions_path = HERE / "selector_instructions.txt"
    manifest = load_launch(launch_manifest, candidates, instructions_path, assets)
    asset_manifest = _validate_assets(assets, manifest)
    planned = plan_observations(candidates)
    instructions = instructions_path.read_text(encoding="utf-8")
    synthetic = {"claim": "Synthetic warm-up only; do not use external knowledge.", "candidate": {
        "id": "WARMUP", "text": "Synthetic candidate used solely to initialize the native runtime.", "url": "https://example.invalid/warmup"}}
    call_plan = [("warmup", None, None, synthetic), *(("measured", r["case_id"], r["id"], r["observation"]) for r in planned)]
    run_id = hashlib.sha256((_sha256(launch_manifest) + _sha256(candidates) + manifest["checkpoint_sha256"]).encode()).hexdigest()[:32]
    started = monotonic()
    _create_output(output)
    metadata = {"schema": RUN_SCHEMA, "run_id": run_id, "source_commit": manifest["source_commit"],
                "launch_manifest_sha256": _sha256(launch_manifest), "candidates_sha256": _sha256(candidates),
                "instructions_sha256": _sha256(instructions_path), "asset_manifest_sha256": _sha256(assets),
                "identity": {k: manifest[k] for k in ("upstream_commit", "model", "returned_model", "model_revision", "checkpoint_sha256")}, "runtime_identity": None,
                "warmup_calls": WARMUP_CALLS, "measured_calls": MEASURED_CALLS, "maximum_calls": MAXIMUM_CALLS,
                "retries": RETRIES, "gpu_allocation_seconds": GPU_ALLOCATION_SECONDS, "wall_seconds": WALL_SECONDS,
                "planned_ids_sha256": hashlib.sha256(_canonical([(r["case_id"], r["id"]) for r in planned]).encode()).hexdigest(), "started_unix": time()}
    _write_new(output / "run-metadata.json", metadata)
    for name in ("intents.jsonl", "results.jsonl", "traces.jsonl", "scored-results.jsonl"):
        _empty_new(output / name)
    stop_reason: str | None = None
    all_valid = False
    try:
        _validate_checkpoint_tree(checkpoint, asset_manifest)
        runtime = _load_runtime(checkpoint, runtime_factory)
        normalized_identity, runtime_identity = _runtime_identity(runtime, manifest)
        metadata["runtime_identity"] = runtime_identity
        # Persist the hardware/runtime identity before the first possible forward.
        _replace_atomic(output / "run-metadata.json", metadata)
        if not callable(getattr(runtime, "execute", None)) or not callable(getattr(runtime, "preflight", None)):
            raise ValueError("jeff_runtime_api_missing")
        # The native preflight sees every exact request before the warm-up starts.
        preflight_records = []
        for _phase, _case, _candidate, observation in call_plan:
            if monotonic() - started >= WALL_SECONDS:
                raise ValueError("jeff_wall_cap_preflight")
            request = _request(observation, instructions)
            report = runtime.preflight(request)
            if (not isinstance(report, Mapping) or type(report.get("input_tokens")) is not int or report["input_tokens"] < 1
                    or type(report.get("state_characters")) is not int or report["state_characters"] < 1):
                raise ValueError("jeff_preflight_report_invalid")
            preflight_records.append({"request_sha256": hashlib.sha256(_canonical(request).encode()).hexdigest(),
                                      "input_tokens": report["input_tokens"], "state_characters": report["state_characters"]})
        _write_new(output / "preflight.json", {"schema": "averitec-jeff-preflight/v1", "complete": True, "calls": MAXIMUM_CALLS,
                   "records": preflight_records, "min_input_tokens": min(r["input_tokens"] for r in preflight_records),
                   "max_input_tokens": max(r["input_tokens"] for r in preflight_records),
                   "min_state_characters": min(r["state_characters"] for r in preflight_records),
                   "max_state_characters": max(r["state_characters"] for r in preflight_records)})
        Adapter = _load_adapter_factory(adapter_factory)
        tracing = importlib.import_module("tracing").TraceRecorder
        for ordinal, (phase, case_id, candidate_id, observation) in enumerate(call_plan, 1):
            if monotonic() - started >= WALL_SECONDS:
                stop_reason = "wall_cap"
                break
            recorder = tracing(output / "traces.jsonl", run_id, f"{ordinal}:{phase}:{case_id or 'warmup'}")
            request = _request(observation, instructions)
            _append(output / "intents.jsonl", {"ordinal": ordinal, "phase": phase, "case_id": case_id, "id": candidate_id,
                                                "trace_id": recorder.trace_id, "request_sha256": hashlib.sha256(_canonical(request).encode()).hexdigest()})
            adapter = Adapter(execute=runtime.execute, preflight=runtime.preflight, tracer=recorder)
            result = _result_from_adapter(adapter.choose(observation, instructions, ["include", "exclude"]), phase=phase,
                                          case_id=case_id, candidate_id=candidate_id, trace_id=recorder.trace_id, request=request)
            _append(output / "results.jsonl", result)
            if phase == "measured":
                _append(output / "scored-results.jsonl", {k: result[k] for k in ("case_id", "id", "action", "outcome")})
            if result["outcome"] != "ok":
                stop_reason = "call_failure"
                break
            if _trace_events(output / "traces.jsonl", recorder.trace_id) is None:
                stop_reason = "trace_failure"
                break
    except Exception as exc:
        stop_reason = "preflight_or_runtime_failure" if not any(_rows(output / "intents.jsonl")) else "runner_failure"
        code = str(exc) if isinstance(exc, (ValueError, RuntimeError)) and re.fullmatch(r"jeff_[a-z0-9_:.-]{1,150}", str(exc)) else type(exc).__name__
        _append(output / "traces.jsonl", {"event": "runner_error", "error_type": type(exc).__name__, "error_code": code})
    complete = stop_reason is None and len(_rows(output / "results.jsonl")) == MAXIMUM_CALLS
    all_valid = complete and all(r.get("outcome") == "ok" for r in _rows(output / "results.jsonl"))
    return _receipt(output, metadata, complete=complete, all_valid=all_valid, stop_reason=stop_reason)


def _expected_result_from_payload(payload: Any) -> tuple[str, str | None] | None:
    """Independently validate the native response shape used by JeffEvidenceSelector."""
    if not isinstance(payload, dict) or payload.get("model") != RETURNED_MODEL:
        return None
    answer = payload.get("answers", {}).get("next_action") if isinstance(payload.get("answers"), dict) else None
    usage = payload.get("usage")
    if not isinstance(answer, dict) or answer.get("type") != "choice" or not isinstance(usage, dict):
        return None
    probs, choice, confidence = answer.get("probabilities"), answer.get("choice"), answer.get("confidence")
    valid_number = lambda v: type(v) in (int, float) and 0 <= v <= 1
    if (not isinstance(probs, dict) or set(probs) != {"include", "exclude"} or not all(valid_number(v) for v in probs.values())
            or abs(sum(probs.values()) - 1) > 1e-6 or choice not in probs or probs[choice] != max(probs.values())
            or not valid_number(confidence) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens"))):
        return None
    return "ok", choice


def verify_receipt(output: Path, candidates: Path, launch_manifest: Path, assets: Path) -> dict[str, Any]:
    """Offline, standard-library-only verification; it never imports or runs Jeff."""
    instructions = HERE / "selector_instructions.txt"
    manifest = load_launch(launch_manifest, candidates, instructions, assets)
    metadata, receipt = _load_json(output / "run-metadata.json", "jeff_metadata_json"), _load_json(output / "receipt.json", "jeff_receipt_json")
    metadata_keys = {"schema", "run_id", "source_commit", "launch_manifest_sha256", "candidates_sha256", "instructions_sha256", "asset_manifest_sha256", "identity", "runtime_identity", "warmup_calls", "measured_calls", "maximum_calls", "retries", "gpu_allocation_seconds", "wall_seconds", "planned_ids_sha256", "started_unix"}
    receipt_keys = {"schema", "run_id", "complete", "all_valid", "stop_reason", "planned_ids_sha256", "journal_sha256"}
    if not isinstance(metadata, dict) or set(metadata) != metadata_keys or metadata.get("schema") != RUN_SCHEMA or not isinstance(receipt, dict) or set(receipt) != receipt_keys or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("jeff_receipt_shape")
    identity = {k: manifest[k] for k in ("upstream_commit", "model", "returned_model", "model_revision", "checkpoint_sha256")}
    if (metadata.get("identity") != identity or metadata.get("source_commit") != manifest["source_commit"]
            or metadata.get("launch_manifest_sha256") != _sha256(launch_manifest) or metadata.get("candidates_sha256") != _sha256(candidates)
            or metadata.get("instructions_sha256") != _sha256(instructions) or metadata.get("asset_manifest_sha256") != _sha256(assets)
            or tuple(metadata.get(k) for k in ("warmup_calls", "measured_calls", "maximum_calls", "retries", "gpu_allocation_seconds", "wall_seconds")) != (1, 1000, 1001, 0, GPU_ALLOCATION_SECONDS, WALL_SECONDS)):
        raise ValueError("jeff_receipt_binding")
    if _normalise_runtime_identity(metadata.get("runtime_identity")) != identity:
        raise ValueError("jeff_runtime_identity_binding")
    journal_names = {"run-metadata.json", "preflight.json", "intents.jsonl", "results.jsonl", "traces.jsonl", "scored-results.jsonl"}
    hashes = receipt.get("journal_sha256")
    if not isinstance(hashes, dict) or set(hashes) != journal_names or any(hashes[n] != _sha256(output / n) for n in journal_names):
        raise ValueError("jeff_journal_drift")
    if receipt.get("complete") is not True or receipt.get("all_valid") is not True or receipt.get("stop_reason") is not None:
        raise ValueError("jeff_receipt_incomplete")
    planned, intents, results, scored = plan_observations(candidates), _rows(output / "intents.jsonl"), _rows(output / "results.jsonl"), _rows(output / "scored-results.jsonl")
    expected_pairs = [(r["case_id"], r["id"]) for r in planned]
    planned_hash = hashlib.sha256(_canonical(expected_pairs).encode()).hexdigest()
    if metadata.get("planned_ids_sha256") != planned_hash or receipt.get("planned_ids_sha256") != planned_hash or receipt.get("run_id") != metadata.get("run_id"):
        raise ValueError("jeff_plan_binding")
    if len(intents) != MAXIMUM_CALLS or len(results) != MAXIMUM_CALLS or len(scored) != MEASURED_CALLS:
        raise ValueError("jeff_call_count")
    calls = [("warmup", None, None, {"claim": "Synthetic warm-up only; do not use external knowledge.", "candidate": {"id": "WARMUP", "text": "Synthetic candidate used solely to initialize the native runtime.", "url": "https://example.invalid/warmup"}}), *(("measured", r["case_id"], r["id"], r["observation"]) for r in planned)]
    preflight = _load_json(output / "preflight.json", "jeff_preflight_json")
    expected_hashes = [hashlib.sha256(_canonical(_request(observation, instructions.read_text(encoding="utf-8"))).encode()).hexdigest() for _phase, _case, _candidate, observation in calls]
    if (not isinstance(preflight, dict) or set(preflight) != {"schema", "complete", "calls", "records", "min_input_tokens", "max_input_tokens", "min_state_characters", "max_state_characters"}
            or preflight.get("schema") != "averitec-jeff-preflight/v1" or preflight.get("complete") is not True or preflight.get("calls") != MAXIMUM_CALLS
            or not isinstance(preflight.get("records"), list) or len(preflight["records"]) != MAXIMUM_CALLS):
        raise ValueError("jeff_preflight_binding")
    records = preflight["records"]
    if ([r.get("request_sha256") for r in records] != expected_hashes
            or any(set(r) != {"request_sha256", "input_tokens", "state_characters"} or type(r["input_tokens"]) is not int or r["input_tokens"] < 1 or type(r["state_characters"]) is not int or r["state_characters"] < 1 for r in records)
            or preflight["min_input_tokens"] != min(r["input_tokens"] for r in records) or preflight["max_input_tokens"] != max(r["input_tokens"] for r in records)
            or preflight["min_state_characters"] != min(r["state_characters"] for r in records) or preflight["max_state_characters"] != max(r["state_characters"] for r in records)):
        raise ValueError("jeff_preflight_binding")
    traces = _rows(output / "traces.jsonl")
    trace_ids = {row["trace_id"] for row in intents}
    request_starts = [r for r in traces if r.get("event") == "span_start" and r.get("name") == "controller.request"]
    request_ends = [r for r in traces if r.get("event") == "span_end" and r.get("trace_id") in trace_ids]
    if len(request_starts) != MAXIMUM_CALLS or len(request_ends) != MAXIMUM_CALLS or {r.get("trace_id") for r in request_starts} != trace_ids:
        raise ValueError("jeff_trace_call_cap")
    for ordinal, ((phase, case_id, candidate_id, observation), intent, result) in enumerate(zip(calls, intents, results), 1):
        request = _request(observation, instructions.read_text(encoding="utf-8"))
        request_hash = hashlib.sha256(_canonical(request).encode()).hexdigest()
        if (intent != {"ordinal": ordinal, "phase": phase, "case_id": case_id, "id": candidate_id, "trace_id": intent.get("trace_id"), "request_sha256": request_hash}
                or not isinstance(intent.get("trace_id"), str) or result.get("phase") != phase or result.get("case_id") != case_id
                or result.get("id") != candidate_id or result.get("trace_id") != intent["trace_id"] or result.get("request_sha256") != request_hash
                or result.get("model") != MODEL or result.get("returned_model") != RETURNED_MODEL or result.get("outcome") != "ok" or result.get("action") not in {"include", "exclude"}):
            raise ValueError("jeff_intent_result_binding")
        starts = [r for r in traces if r.get("event") == "span_start" and r.get("trace_id") == intent["trace_id"] and r.get("name") == "controller.request"]
        ends = [r for r in traces if r.get("event") == "span_end" and r.get("trace_id") == intent["trace_id"]]
        if len(starts) != 1 or len(ends) != 1 or starts[0].get("span_id") != ends[0].get("span_id") or starts[0].get("model") != MODEL or starts[0].get("input") != request:
            raise ValueError("jeff_trace_request_binding")
        expected = _expected_result_from_payload(ends[0].get("output"))
        if expected != ("ok", result["action"]):
            raise ValueError("jeff_trace_outcome_binding")
    expected_scored = [{k: r[k] for k in ("case_id", "id", "action", "outcome")} for r in results[1:]]
    if scored != expected_scored:
        raise ValueError("jeff_score_output_binding")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.plan:
        load_launch(args.manifest, args.candidates, HERE / "selector_instructions.txt", args.assets)
        plan = plan_observations(args.candidates)
        print(_canonical({"measured_calls": len(plan), "warmup_calls": 1, "maximum_calls": 1001,
                          "planned_ids_sha256": hashlib.sha256(_canonical([(r["case_id"], r["id"]) for r in plan]).encode()).hexdigest()}))
        return
    if args.verify:
        print(_canonical(verify_receipt(args.output, args.candidates, args.manifest, args.assets)))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required with --execute")
    receipt = execute(args.candidates, args.manifest, args.output, args.assets, args.checkpoint)
    print(_canonical(receipt))
    if not receipt["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
