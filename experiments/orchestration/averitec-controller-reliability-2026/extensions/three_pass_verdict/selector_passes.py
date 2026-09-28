"""Three new, independent selector passes on the sealed 100 x 10 pool.

The launch file is deliberately separate from historical one-pass manifests.
No reference, grade, or verdict-gold path is accepted here. Each arm/pass gets
one new output directory and cannot resume an uncertain request.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from time import monotonic, time
from typing import Any, Callable
from urllib.request import HTTPRedirectHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "evidence_selection"
EXPERIMENT = HERE.parents[1]
REPO = HERE.parents[4]
for directory in (EVIDENCE, EXPERIMENT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_selector as previous  # noqa: E402
import jeff_runner as previous_jeff  # noqa: E402
from recovery_runtime import RECOVERY_PROFILE_IDENTITIES, create_recovery_selector  # noqa: E402
from selector_runtime import BINARY_ACTIONS, create_selector  # noqa: E402
from tracing import TraceRecorder  # noqa: E402
import providers as provider_module  # noqa: E402
import recovery_runtime as recovery_module  # noqa: E402
import hosted_budget  # noqa: E402
import selector_dispatch  # noqa: E402

SCHEMA = "averitec-three-pass-selector-launch/v1"
RUN_SCHEMA = "averitec-three-pass-selector-run/v1"
RECEIPT_SCHEMA = "averitec-three-pass-selector-receipt/v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
ARMS = {
    "jev": ("jev-1.13.0", "baseline"),
    "lfm": ("LiquidAI/LFM2.5-1.2B-Instruct", "lfm_native"),
    "laya_typed": ("convaiinnovations/laya-typed-decisions", "native_binary_choice"),
    "gemini": ("gemini-3.1-flash-lite", "gemini_native_timeout120_v2"),
    "qwen": ("Qwen/Qwen3.5-4B", "qwen_baseline_warmup180_v2"),
    "lfm26": ("LiquidAI/LFM2.5-2.6B", "lfm26_native_4096_v2"),
    "jeff": (previous_jeff.MODEL, "native_binary_choice"),
}
HTTP_ARMS = frozenset({"jev", "lfm", "gemini", "qwen", "lfm26"})
NATIVE_ARMS = frozenset({"jeff", "laya_typed"})
REQUIRED_CODE = frozenset({
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/annotation.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/selector_passes.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/hosted_budget.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/selector_dispatch.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/config.yaml",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/jev_preflight.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/run_selector.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_runtime.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/recovery_runtime.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/jeff_runner.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/jeff_adapter.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/jeff_native.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/selector_instructions.txt",
    "experiments/orchestration/averitec-controller-reliability-2026/providers.py",
    "experiments/orchestration/averitec-controller-reliability-2026/gemini_provider.py",
    "experiments/orchestration/averitec-controller-reliability-2026/generation_profiles.py",
    "experiments/orchestration/averitec-controller-reliability-2026/tracing.py",
})
WARMUP = {"claim": "Synthetic selector deployment check.", "candidate": {
    "id": "synthetic", "text": "This is a synthetic plumbing check.", "url": "https://example.invalid/synthetic"}}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def wire_bytes(value: Any) -> bytes:
    """Match both existing HTTP adapters' JSON serializer, including escapes."""
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("three_pass_invalid_json") from None


def _head() -> str:
    marker = REPO / "SOURCE_COMMIT"
    if marker.exists():
        commit = marker.read_text(encoding="ascii").strip()
    else:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if not HEX40.fullmatch(commit):
        raise ValueError("three_pass_source_commit_invalid")
    return commit


def _safe_code(name: Any) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("three_pass_code_path")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("three_pass_code_path")
    return path


def check_gate_scope(gate: Path, launch: Path, arm: str | None = None,
                     pass_number: int | None = None) -> dict[str, Any]:
    review = _read(gate)
    common = {"manifest_sha256": digest(launch), "decision": "approved", "reviewer": "gpt-6-astra"}
    if not isinstance(review, dict) or any(review.get(key) != value for key, value in common.items()):
        raise ValueError("three_pass_astra_gate")
    if review.get("schema") == "averitec-astra-launch-gate/v1":
        if review != {"schema": "averitec-astra-launch-gate/v1", **common}:
            raise ValueError("three_pass_astra_gate")
    elif review.get("schema") == "averitec-astra-launch-gate/v2":
        allowed_arms, allowed_passes = review.get("allowed_arms"), review.get("allowed_passes")
        if (set(review) != {"schema", *common, "allowed_arms", "allowed_passes"}
                or not isinstance(allowed_arms, list) or not allowed_arms
                or allowed_arms != sorted(set(allowed_arms)) or not set(allowed_arms) <= set(ARMS)
                or not isinstance(allowed_passes, list) or not allowed_passes
                or allowed_passes != sorted(set(allowed_passes))
                or not set(allowed_passes) <= {1, 2, 3}):
            raise ValueError("three_pass_astra_gate")
        if (arm is not None and arm not in allowed_arms) or (pass_number is not None and pass_number not in allowed_passes):
            raise ValueError("three_pass_astra_gate_scope")
    else:
        raise ValueError("three_pass_astra_gate")
    return review


def load_launch(path: Path, candidates: Path, *, gate: Path | None = None,
                arm: str | None = None, pass_number: int | None = None,
                frozen_source: bool = False) -> dict[str, Any]:
    """Check the exact offline binding. Execute additionally requires Astra gate."""
    required_arm = arm
    value = _read(path)
    keys = {"schema", "source_commit", "candidates_sha256", "instructions_sha256", "code_sha256",
            "passes", "arms", "retries", "total_measured_calls", "resource_caps",
            "jeff_asset_manifest_sha256", "jeff_checkpoint_sha256", "hosted_budget_seed_sha256"}
    if not isinstance(value, dict) or set(value) != keys or value["schema"] != SCHEMA:
        raise ValueError("three_pass_manifest_shape")
    if (((not isinstance(value["source_commit"], str) or not HEX40.fullmatch(value["source_commit"]))
         if frozen_source else value["source_commit"] != _head())
            or value["retries"] != 0 or value["passes"] != [1, 2, 3] or value["total_measured_calls"] != 21000):
        raise ValueError("three_pass_manifest_policy")
    instructions = EVIDENCE / "selector_instructions.txt"
    if (value["candidates_sha256"] != digest(candidates)
            or (not HEX64.fullmatch(value["instructions_sha256"]) if frozen_source
                else value["instructions_sha256"] != digest(instructions))):
        raise ValueError("three_pass_input_drift")
    code = value["code_sha256"]
    if not isinstance(code, dict) or not REQUIRED_CODE.issubset(code):
        raise ValueError("three_pass_code_missing")
    for name, expected in code.items():
        relative = _safe_code(name)
        if (not isinstance(expected, str) or not HEX64.fullmatch(expected)
                or (not frozen_source and (not (REPO / relative).is_file() or digest(REPO / relative) != expected))):
            raise ValueError("three_pass_code_drift")
    arms = value["arms"]
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("three_pass_arms")
    for arm, (model, profile) in ARMS.items():
        row = arms[arm]
        if (not isinstance(row, dict) or set(row) != {"model", "profile", "returned_model", "wall_seconds", "maximum_calls", "gpu_allocation_seconds", "token_caps", "token_preflight_sha256"}
                or row["model"] != model or row["profile"] != profile or row["maximum_calls"] != 1001
                or not isinstance(row["returned_model"], str) or not row["returned_model"]
                or type(row["wall_seconds"]) is not int or row["wall_seconds"] < 60
                or type(row["gpu_allocation_seconds"]) is not int or row["gpu_allocation_seconds"] < 0):
            raise ValueError("three_pass_arm_policy")
        if arm == "jeff" and row["returned_model"] != previous_jeff.RETURNED_MODEL:
            raise ValueError("three_pass_jeff_identity")
        if arm != "jeff" and row["returned_model"] != model:
            raise ValueError("three_pass_returned_model")
        caps_row = row["token_caps"]
        if arm in HTTP_ARMS:
            if (not isinstance(row["token_preflight_sha256"], str) or not HEX64.fullmatch(row["token_preflight_sha256"])
                    or not isinstance(caps_row, dict)
                    or set(caps_row) != {"input_per_call", "output_per_call", "input_total", "output_total", "context_limit"}
                    or any(type(v) is not int or v <= 0 for v in caps_row.values())
                    or caps_row["input_total"] < caps_row["input_per_call"]
                    or caps_row["output_total"] < caps_row["output_per_call"]
                    or caps_row["input_per_call"] + caps_row["output_per_call"] > caps_row["context_limit"]):
                raise ValueError("three_pass_token_caps")
        elif row["token_preflight_sha256"] is not None or caps_row is not None:
            raise ValueError("three_pass_native_token_caps")
    if any(not isinstance(value.get(name), str) or not HEX64.fullmatch(value[name])
           for name in ("jeff_asset_manifest_sha256", "jeff_checkpoint_sha256", "hosted_budget_seed_sha256")):
        raise ValueError("three_pass_jeff_asset_binding")
    caps = value["resource_caps"]
    if (not isinstance(caps, dict) or set(caps) != {"maximum_gpu_seconds", "maximum_api_requests", "maximum_jobs"}
            or any(type(v) is not int or v <= 0 for v in caps.values())
            or caps["maximum_api_requests"] < 3 * 1001 * 2
            or caps["maximum_gpu_seconds"] < 3 * sum(row["gpu_allocation_seconds"] for row in arms.values())):
        raise ValueError("three_pass_resource_caps")
    previous.plan_observations(candidates)  # validates all 1000 and gold-free schema
    if gate is not None:
        check_gate_scope(gate, path, required_arm, pass_number)
    return value


def plan(path: Path, candidates: Path) -> dict[str, Any]:
    manifest = load_launch(path, candidates)
    pairs = [(row["case_id"], row["id"]) for row in previous.plan_observations(candidates)]
    return {"schema": "averitec-three-pass-selector-plan/v1", "passes": manifest["passes"],
            "arms": list(manifest["arms"]), "cases": 100, "pairs_per_arm_pass": len(pairs),
            "total_measured_calls": 21000, "total_warmups": 21,
            "planned_ids_sha256": hashlib.sha256(canonical(pairs).encode()).hexdigest(),
            "launch_manifest_sha256": digest(path), "model_calls": 0}


def _new_json(path: Path, value: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write((canonical(value) + "\n").encode())
        target.flush()
        os.fsync(target.fileno())


def _empty(path: Path) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.flush()
        os.fsync(target.fileno())


def _append(path: Path, value: Any) -> None:
    with path.open("ab") as target:
        target.write((canonical(value) + "\n").encode())
        target.flush()
        os.fsync(target.fileno())


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("three_pass_journal_invalid") from None
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("three_pass_journal_invalid")
    return rows


def _input_usage_allowed(arm: str, actual: Any, planned: Any, margin: Any) -> bool:
    """Apply the same input-token rule during execution and receipt audit."""
    if (type(actual) is not int or actual < 0 or type(planned) is not int or planned < 0
            or type(margin) is not int or margin < 0):
        return False
    if arm == "jev":
        # The provider has no tokenizer count; planned is a conservative
        # serialized-byte estimate, not an expected exact usage value.
        return actual <= planned
    if arm == "gemini":
        return actual <= planned + margin
    return actual == planned


def _warmup_usage_valid(arm: str, usage: Any) -> bool:
    """Jeff's classifier reports no generated output-token count by design."""
    if not isinstance(usage, dict) or type(usage.get("input_tokens")) is not int or usage["input_tokens"] < 0:
        return False
    output = usage.get("output_tokens")
    if arm == "jeff":
        return output is None
    return type(output) is int and output >= 0


def _native_token_usage(arm: str, results: list[dict[str, Any]]) -> dict[str, int | None]:
    """Aggregate reported native usage without inventing a token count.

    Jeff classifies labels rather than generating text, so its output-token
    count is not a measured zero. A missing input or Laya output count makes
    that aggregate unknown instead of silently omitting the call.
    """
    if arm not in NATIVE_ARMS:
        raise ValueError("three_pass_native_usage_arm")
    totals: dict[str, int | None] = {"input_tokens": 0,
                                    "output_tokens": None if arm == "jeff" else 0}
    for result in results:
        usage = result.get("usage")
        for key in ("input_tokens", "output_tokens"):
            if arm == "jeff" and key == "output_tokens":
                continue
            value = usage.get(key) if isinstance(usage, dict) else None
            if totals[key] is None or type(value) is not int or value < 0:
                totals[key] = None
            else:
                totals[key] += value
    return totals


def _audit_native_usage(arm: str, source_commit: str, results: list[dict[str, Any]],
                        reported: Any, *, frozen_source: bool) -> dict[str, Any] | None:
    recomputed = _native_token_usage(arm, results)
    if reported == recomputed:
        return None
    # The frozen v13 Laya runner wrote a zero aggregate despite recording
    # per-request usage. Keep its immutable receipt and expose the discrepancy.
    if (frozen_source and arm == "laya_typed"
            and source_commit == "f9f62f46b3dd566d9ef9e10f7eef40c2f9747602"
            and reported == {"input_tokens": 0, "output_tokens": 0}):
        return {"reason": "historical_v13_laya_zero_aggregate",
                "reported": reported, "recomputed_from_results": recomputed}
    raise ValueError("three_pass_native_token_usage_receipt")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@contextmanager
def _guard_http_request(arm: str, expected_hash: str, endpoint: str | None, deadline: float):
    """Permit one exact HTTP request, with a hard deadline and no redirect."""
    if threading.current_thread() is not threading.main_thread() or deadline <= 0:
        raise ValueError("three_pass_http_deadline_unavailable")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] or previous_timer[1]:
        raise ValueError("three_pass_existing_alarm")
    previous_handler = signal.getsignal(signal.SIGALRM)
    original_jev, original_v2 = provider_module._http_post, recovery_module._post
    calls = 0
    if arm == "jev":
        expected_url = provider_module.TYPESAFE_SYSTEM_ONE_URL
    elif arm == "gemini":
        expected_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"
    elif endpoint and endpoint.startswith("http://127.0.0.1:") and endpoint.endswith("/v1"):
        expected_url = endpoint + "/chat/completions"
    else:
        raise ValueError("three_pass_http_endpoint_unpinned")

    def dispatch(url: str, body: dict, headers: dict, timeout: float) -> dict:
        nonlocal calls
        calls += 1
        if calls != 1 or url != expected_url or hashlib.sha256(wire_bytes(body)).hexdigest() != expected_hash:
            raise ValueError("three_pass_http_request_drift")
        request = Request(url, data=wire_bytes(body),
                          headers={"Content-Type": "application/json", **headers}, method="POST")
        with build_opener(_NoRedirect()).open(request, timeout=min(timeout, deadline)) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("three_pass_http_response_not_object")
        return value

    def alarm(_signum: int, _frame: Any) -> None:
        raise TimeoutError("three_pass_http_wall_deadline")

    try:
        provider_module._http_post = dispatch
        recovery_module._post = dispatch
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, deadline)
        yield lambda: calls
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        provider_module._http_post = original_jev
        recovery_module._post = original_v2


def _trace_response_rows(rows: list[dict[str, Any]], trace_id: str, arm: str) -> tuple[bool, str | None]:
    starts = {r.get("span_id") for r in rows if r.get("trace_id") == trace_id
              and r.get("event") == "span_start" and isinstance(r.get("span_id"), str)}
    ends = {r.get("span_id") for r in rows if r.get("trace_id") == trace_id
            and r.get("event") == "span_end" and isinstance(r.get("span_id"), str)}
    requests = {r.get("span_id") for r in rows if r.get("trace_id") == trace_id
                and r.get("event") == "span_start" and r.get("name") == "controller.request"}
    models = []
    for row in rows:
        if row.get("trace_id") == trace_id and row.get("event") == "span_end" and row.get("span_id") in requests:
            payload = row.get("output")
            if isinstance(payload, dict):
                models.append(payload.get("modelVersion") if arm == "gemini" else payload.get("model"))
    found = [m for m in models if isinstance(m, str)]
    return bool(starts) and starts == ends and len(requests) == 1, found[0] if len(found) == 1 else None


def _trace_response(path: Path, trace_id: str, arm: str, offset: int = 0) -> tuple[bool, str | None]:
    """Inspect only the current call's trace suffix, including raw model ID."""
    with path.open("rb") as source:
        source.seek(offset)
        rows = [json.loads(line) for line in source if line.strip()]
    return _trace_response_rows(rows, trace_id, arm)


def _make_selector(arm: str, *, endpoint: str | None, laya_checkpoint: Path | None, keys: dict[str, str]) -> Any:
    if arm in {"gemini", "qwen", "lfm26"}:
        return create_recovery_selector(arm, endpoint=endpoint, keys=keys)
    return create_selector(arm, endpoint=endpoint, laya_checkpoint=laya_checkpoint, keys=keys)


def _make_jeff(*, checkpoint: Path, assets: Path, expected_assets: str, expected_checkpoint: str) -> tuple[Any, dict[str, Any]]:
    if digest(assets) != expected_assets:
        raise ValueError("three_pass_jeff_asset_manifest_drift")
    asset = _read(assets)
    if not isinstance(asset, dict) or asset.get("schema") != previous_jeff.ASSET_SCHEMA:
        raise ValueError("three_pass_jeff_assets")
    if asset.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError("three_pass_jeff_checkpoint_drift")
    previous_jeff._validate_checkpoint_tree(checkpoint, asset)
    runtime = previous_jeff._load_runtime(checkpoint, None)
    identity = getattr(runtime, "identity", None)
    if callable(identity):
        identity = identity()
    normalized = previous_jeff._normalise_runtime_identity(identity)
    expected = {"upstream_commit": previous_jeff.UPSTREAM_COMMIT, "model": previous_jeff.MODEL,
                "returned_model": previous_jeff.RETURNED_MODEL, "model_revision": previous_jeff.MODEL_REVISION,
                "checkpoint_sha256": asset.get("checkpoint_sha256")}
    if normalized != expected or asset.get("model") != previous_jeff.MODEL or asset.get("model_revision") != previous_jeff.MODEL_REVISION:
        raise ValueError("three_pass_jeff_runtime_identity")
    if not callable(getattr(runtime, "preflight", None)) or not callable(getattr(runtime, "execute", None)):
        raise ValueError("three_pass_jeff_runtime_api")
    return runtime, dict(identity)


def _local_identity(arm: str, selector: Any, expected_model: str, expected_returned: str, expected_profile: str) -> dict[str, Any]:
    if arm == "laya_typed":
        identity = selector.identity()
        if (identity.get("model") != expected_model or identity.get("returned_model") != expected_returned
                or identity.get("checkpoint_verification") != "verified"
                or not str(identity.get("actual_device", "")).startswith("cuda")):
            raise ValueError("three_pass_laya_runtime_unverified")
        return identity
    model, returned, profile = previous._identity(selector)
    if model != expected_model or returned != expected_returned:
        raise ValueError("three_pass_selector_identity")
    if arm == "jev" and profile is None and expected_profile == "baseline":
        return {"name": "baseline", "model": model, "adapter": "JevController"}
    if not isinstance(profile, dict) or profile.get("name") != expected_profile:
        raise ValueError("three_pass_profile_identity")
    if arm in RECOVERY_PROFILE_IDENTITIES and profile != RECOVERY_PROFILE_IDENTITIES[arm]:
        raise ValueError("three_pass_recovery_profile_drift")
    return profile


def _preflight_native(arm: str, selector: Any, observations: list[dict[str, Any]], instructions: str) -> list[dict[str, Any]]:
    reports = []
    for observation in observations:
        if arm == "jeff":
            request = previous_jeff._request(observation, instructions)
            report = selector.preflight(request)
            request_hash = hashlib.sha256(canonical(request).encode()).hexdigest()
        elif arm == "laya_typed":
            report = selector.preflight(observation, instructions, list(BINARY_ACTIONS))
            request_hash = hashlib.sha256(canonical(observation).encode()).hexdigest()
        else:
            return []
        if not isinstance(report, Mapping) or not report:
            raise ValueError("three_pass_preflight_invalid")
        if arm == "jeff":
            if any(type(report.get(k)) is not int or report[k] < 1 for k in ("input_tokens", "state_characters")):
                raise ValueError("three_pass_preflight_invalid")
        elif any(type(v) is not int or v < 0 for v in report.values()):
            raise ValueError("three_pass_preflight_invalid")
        reports.append({"request_sha256": request_hash, "report": dict(report)})
    return reports


class _CaptureSpan:
    def update(self, **_kwargs: Any) -> None:
        pass


class _CaptureTracer:
    @contextmanager
    def span(self, *_args: Any, **_kwargs: Any):
        yield _CaptureSpan()


def _capture_http_requests(arm: str, selector: Any, observations: list[dict[str, Any]], instructions: str) -> list[dict[str, Any]]:
    """Run adapters with their transport replaced by a local capture stub.

    The stub always raises before I/O. This produces the exact adapter-built
    request bodies for comparison with an independently counted token report.
    """
    module = recovery_module if arm in {"gemini", "qwen", "lfm26"} else provider_module
    function = "_post" if module is recovery_module else "_http_post"
    original = getattr(module, function)
    captured: list[dict[str, Any]] = []
    provider = previous._provider(selector)
    original_tracer = getattr(provider, "tracer", None)
    original_key = getattr(provider, "key", None)
    def intercept(_url: str, body: dict[str, Any], *_args: Any, **_kwargs: Any) -> Any:
        captured.append(json.loads(wire_bytes(body)))
        raise OSError("offline_preflight_transport_blocked")
    setattr(module, function, intercept)
    provider.tracer = _CaptureTracer()
    if arm in {"jev", "gemini"}:
        provider.key = "offline-preflight-placeholder"
    try:
        for phase, observation in [("warmup", observations[0]), *( ("measured", item) for item in observations[1:])]:
            if hasattr(selector, "set_phase"):
                selector.set_phase(phase)
            before = len(captured)
            selector.choose(observation, instructions, list(BINARY_ACTIONS))
            if len(captured) != before + 1:
                raise ValueError("three_pass_http_capture_failed")
    finally:
        setattr(module, function, original)
        provider.tracer = original_tracer
        if arm in {"jev", "gemini"}:
            provider.key = original_key
    return captured


def _load_http_preflight(path: Path, manifest_hash: str, arm: str, model: str, profile: str,
                         caps: dict[str, int], observations: list[dict[str, Any]], requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if digest(path) != manifest_hash:
        raise ValueError("three_pass_token_preflight_drift")
    value = _read(path)
    keys = {"schema", "arm", "model", "profile", "method", "method_identity_sha256", "context_limit_tokens",
            "output_request_limit_tokens", "output_limit_enforced", "input_margin_tokens", "records"}
    if (not isinstance(value, dict) or set(value) != keys or value["schema"] != "averitec-selector-token-preflight/v2"
            or value["arm"] != arm or value["model"] != model or value["profile"] != profile
            or value["method"] not in {"exact_pinned_tokenizer", "provider_count_tokens", "estimated_wire_bytes_plus_1024"}
            or (arm == "jev" and value["method"] != "estimated_wire_bytes_plus_1024")
            or (arm == "gemini" and value["method"] != "provider_count_tokens")
            or (arm in {"qwen", "lfm", "lfm26"} and value["method"] != "exact_pinned_tokenizer")
            or not isinstance(value["method_identity_sha256"], str) or not HEX64.fullmatch(value["method_identity_sha256"])
            or value["context_limit_tokens"] != caps["context_limit"]
            or type(value["input_margin_tokens"]) is not int
            or (arm == "gemini" and not 1 <= value["input_margin_tokens"] <= 4096)
            or (arm != "gemini" and value["input_margin_tokens"] != 0)
            or type(value["output_request_limit_tokens"]) is not int or value["output_request_limit_tokens"] < 1
            or value["output_request_limit_tokens"] > caps["output_per_call"]
            or (arm == "jev" and (value["output_request_limit_tokens"] != 1024
                                   or value["output_limit_enforced"] is not False
                                   or value["context_limit_tokens"] != 32768))
            or (arm != "jev" and value["output_limit_enforced"] is not True)):
        raise ValueError("three_pass_token_preflight_shape")
    records = value["records"]
    if not isinstance(records, list) or len(records) != 1001 or len(observations) != 1001 or len(requests) != 1001:
        raise ValueError("three_pass_token_preflight_count")
    for record, observation, request in zip(records, observations, requests):
        serialized = wire_bytes(request)
        if (not isinstance(record, dict) or set(record) != {"observation_sha256", "request_sha256", "serialized_bytes", "input_tokens"}
                or record["observation_sha256"] != hashlib.sha256(canonical(observation).encode()).hexdigest()
                or record["request_sha256"] != hashlib.sha256(serialized).hexdigest()
                or record["serialized_bytes"] != len(serialized)
                or type(record["input_tokens"]) is not int or record["input_tokens"] < 1
                or record["input_tokens"] + value["input_margin_tokens"] > caps["input_per_call"]
                or record["input_tokens"] + value["input_margin_tokens"] + value["output_request_limit_tokens"] > caps["context_limit"]):
            raise ValueError("three_pass_token_preflight_record")
        if arm == "jev" and (len(serialized) > 2048 or record["input_tokens"] != len(serialized) + 1024):
            raise ValueError("three_pass_jev_estimated_reserve")
        # Confirm the output limit is present in the exact existing adapter request.
        request_limit = request.get("max_tokens")
        if arm == "gemini":
            request_limit = request.get("generationConfig", {}).get("maxOutputTokens")
        if arm != "jev" and request_limit != value["output_request_limit_tokens"]:
            raise ValueError("three_pass_output_request_limit_drift")
    if arm != "jev" and sum(r["input_tokens"] + value["input_margin_tokens"] for r in records) > caps["input_total"]:
        raise ValueError("three_pass_input_budget_insufficient")
    if arm != "jev" and 1001 * value["output_request_limit_tokens"] > caps["output_total"]:
        raise ValueError("three_pass_output_budget_insufficient")
    return records


def execute(arm: str, pass_number: int, candidates: Path, launch: Path, gate: Path, output: Path, *,
            endpoint: str | None = None, laya_checkpoint: Path | None = None,
            jeff_assets: Path | None = None, jeff_checkpoint: Path | None = None,
            token_preflight: Path | None = None, hosted_ledger: Path | None = None,
            dispatch_claim: Path | None = None,
            keys: dict[str, str] | None = None, selector_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """One arm/pass, no resume; fake selector_factory is only for offline tests."""
    manifest = load_launch(launch, candidates, gate=gate, arm=arm, pass_number=pass_number)
    if arm not in ARMS or pass_number not in (1, 2, 3):
        raise ValueError("three_pass_arm_or_pass")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("three_pass_output_must_be_new_directory")
    if selector_factory is None:
        if dispatch_claim is None:
            raise ValueError("three_pass_dispatch_claim_required")
        if arm in {"jev", "gemini"}:
            selector_dispatch.verify_registered_claim(dispatch_claim, launch, arm, pass_number, output)
        else:
            selector_dispatch.verify_claim(dispatch_claim, launch, arm, pass_number, output)
    planned = previous.plan_observations(candidates)
    call_plan = [("warmup", None, None, WARMUP)] + [("measured", r["case_id"], r["id"], r["observation"]) for r in planned]
    row = manifest["arms"][arm]
    instructions = (EVIDENCE / "selector_instructions.txt").read_text(encoding="utf-8")
    if arm == "jeff" and selector_factory is None:
        if jeff_assets is None or jeff_checkpoint is None:
            raise ValueError("three_pass_jeff_assets_required")
        selector, runtime_identity = _make_jeff(
            checkpoint=jeff_checkpoint, assets=jeff_assets,
            expected_assets=manifest["jeff_asset_manifest_sha256"],
            expected_checkpoint=manifest["jeff_checkpoint_sha256"])
    else:
        selector = selector_factory(arm, endpoint=endpoint, laya_checkpoint=laya_checkpoint, keys=keys or {}) if selector_factory else _make_selector(arm, endpoint=endpoint, laya_checkpoint=laya_checkpoint, keys=keys or {})
        runtime_identity = _local_identity(arm, selector, row["model"], row["returned_model"], row["profile"])
    observations = [item[3] for item in call_plan]
    qualification = "test_only" if selector_factory is not None else "real_runtime"
    if selector_factory is not None:
        preflight = []
        preflight_method = "injected_test_double"
        preflight_complete = False
        output_request_limit = None
        input_margin = 0
    elif arm in HTTP_ARMS:
        if token_preflight is None:
            raise ValueError("three_pass_exact_token_preflight_required")
        requests = _capture_http_requests(arm, selector, observations, instructions)
        preflight = _load_http_preflight(token_preflight, row["token_preflight_sha256"], arm, row["model"], row["profile"],
                                         row["token_caps"], observations, requests)
        preflight_method = ("estimated_wire_byte_reserve_provider_context_assumed" if arm == "jev"
                            else "exact_request_hash_and_token_report")
        preflight_complete = True
        output_request_limit = _read(token_preflight)["output_request_limit_tokens"]
        input_margin = _read(token_preflight)["input_margin_tokens"]
    else:
        preflight = _preflight_native(arm, selector, observations, instructions)
        if len(preflight) != 1001:
            raise ValueError("three_pass_native_preflight_incomplete")
        preflight_method = "native_runtime_exact_requests"
        preflight_complete = True
        output_request_limit = None
        input_margin = 0
    if selector_factory is None and arm in {"jev", "gemini"}:
        if hosted_ledger is None or hosted_ledger.resolve() != hosted_budget.LEDGER:
            raise ValueError("three_pass_shared_hosted_ledger_required")
        if hosted_budget.snapshot(hosted_ledger)["seed_sha256"] != manifest["hosted_budget_seed_sha256"]:
            raise ValueError("three_pass_hosted_budget_seed_drift")
    os.mkdir(output, 0o700)
    run_id = hashlib.sha256((digest(launch) + arm + str(pass_number) + digest(candidates)).encode()).hexdigest()[:32]
    pair_hash = hashlib.sha256(canonical([(r["case_id"], r["id"]) for r in planned]).encode()).hexdigest()
    metadata = {"schema": RUN_SCHEMA, "run_id": run_id, "arm": arm, "pass": pass_number, "model": row["model"],
                "profile": row["profile"], "runtime_identity": runtime_identity, "source_commit": manifest["source_commit"],
                "manifest_sha256": digest(launch), "gate_sha256": digest(gate), "candidates_sha256": digest(candidates),
                "instructions_sha256": manifest["instructions_sha256"], "planned_ids_sha256": pair_hash,
                "token_preflight_sha256": digest(token_preflight) if token_preflight else None,
                "hosted_ledger": str(hosted_ledger.resolve()) if arm in {"jev", "gemini"} and selector_factory is None else None,
                "hosted_budget_seed_sha256": manifest["hosted_budget_seed_sha256"] if arm in {"jev", "gemini"} and selector_factory is None else None,
                "dispatch_claim_sha256": digest(dispatch_claim) if dispatch_claim else None,
                "token_caps": row["token_caps"], "qualification": qualification,
                "maximum_calls": 1001, "wall_seconds": row["wall_seconds"], "retries": 0, "started_unix": time()}
    _new_json(output / "run-metadata.json", metadata)
    _new_json(output / "preflight.json", {"schema": "averitec-three-pass-preflight/v1", "arm": arm,
                                           "complete": preflight_complete, "method": preflight_method,
                                           "lossless_context_verified": arm != "jev" and preflight_complete,
                                           "calls": len(preflight), "records": preflight,
                                           "input_margin_tokens": input_margin,
                                           "output_request_limit_tokens": output_request_limit})
    for name in ("intents.jsonl", "results.jsonl", "traces.jsonl"):
        _empty(output / name)
    stopped = False
    def mark_stop(_signal: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in old:
        signal.signal(s, mark_stop)
    begun = monotonic()
    stop_reason = None
    counts: dict[str, int] = {}
    used_input = used_output = 0
    try:
        for ordinal, (phase, case_id, candidate_id, observation) in enumerate(call_plan, 1):
            if stopped or monotonic() - begun >= row["wall_seconds"]:
                stop_reason = "signal_before_call" if stopped else "wall_cap"
                break
            if arm in HTTP_ARMS and selector_factory is None:
                if (used_input + preflight[ordinal - 1]["input_tokens"] + input_margin > row["token_caps"]["input_total"]
                        or used_output + output_request_limit > row["token_caps"]["output_total"]):
                    stop_reason = "token_cap_before_call"
                    break
            hosted_request_id = None
            if arm in {"jev", "gemini"} and selector_factory is None:
                hosted_request_id = f"selector/{arm}/pass-{pass_number}/ordinal-{ordinal}"
                remaining = row["wall_seconds"] - (monotonic() - begun)
                if remaining <= 0:
                    stop_reason = "wall_cap"
                    break
                try:
                    hosted_budget.reserve(hosted_ledger, request_id=hosted_request_id, arm=arm, block="selector",
                                          manifest_sha256=digest(launch),
                                          request_sha256=preflight[ordinal - 1]["request_sha256"],
                                          input_tokens=preflight[ordinal - 1]["input_tokens"] + input_margin,
                                          output_tokens=output_request_limit,
                                          wall_seconds=min(math.ceil(remaining), 30 if arm == "jev" else 120))
                except Exception as exc:
                    stop_reason = str(exc) if isinstance(exc, hosted_budget.BudgetStop) else "shared_hosted_budget_unavailable"
                    break
            recorder = TraceRecorder(output / "traces.jsonl", run_id, f"{pass_number}:{ordinal}:{case_id or 'warmup'}", {"arm": arm, "pass": pass_number, "phase": phase})
            trace_offset = (output / "traces.jsonl").stat().st_size
            _append(output / "intents.jsonl", {"ordinal": ordinal, "phase": phase, "case_id": case_id,
                                                "id": candidate_id, "trace_id": recorder.trace_id,
                                                "hosted_request_id": hosted_request_id,
                                                "observation_sha256": hashlib.sha256(canonical(observation).encode()).hexdigest()})
            try:
                call_started = monotonic()
                if arm == "jeff" and selector_factory is None:
                    adapter = previous_jeff._load_adapter_factory(None)(execute=selector.execute, preflight=selector.preflight, tracer=recorder)
                    result = adapter.choose(observation, instructions, list(BINARY_ACTIONS))
                else:
                    previous._set_tracer(selector, recorder)
                    if hasattr(selector, "set_phase"):
                        selector.set_phase(phase)
                    remaining = row["wall_seconds"] - (monotonic() - begun)
                    if arm in HTTP_ARMS and selector_factory is None:
                        with _guard_http_request(arm, preflight[ordinal - 1]["request_sha256"], endpoint, remaining) as call_count:
                            result = previous._bounded_choose(selector, observation, instructions, remaining)
                            if call_count() != 1:
                                raise ValueError("three_pass_http_dispatch_count")
                    else:
                        result = previous._bounded_choose(selector, observation, instructions, remaining)
                outcome = getattr(result, "outcome", None)
                action = getattr(result, "action", None)
                if outcome not in {"ok", "invalid_output", "transport_error", "version_mismatch"}:
                    outcome, action = "invalid_output", None
                if outcome != "ok" or action not in BINARY_ACTIONS:
                    action = None
                trace_complete, raw_model = _trace_response(output / "traces.jsonl", recorder.trace_id, arm, trace_offset)
                if arm == "laya_typed" and trace_complete:
                    raw_model = row["returned_model"]  # pinned local checkpoint identity
                entry = {"phase": phase, "case_id": case_id, "id": candidate_id, "action": action, "outcome": outcome,
                         "model": getattr(result, "model", None), "returned_model": getattr(result, "returned_model", None),
                         "trace_returned_model": raw_model, "error_code": getattr(result, "error_code", None),
                         "usage": {"input_tokens": getattr(result, "input_tokens", None), "output_tokens": getattr(result, "output_tokens", None)},
                         "latency_ms": getattr(result, "latency_ms", None), "trace_id": recorder.trace_id,
                         "trace_complete": trace_complete}
            except BaseException as exc:
                entry = {"phase": phase, "case_id": case_id, "id": candidate_id, "action": None, "outcome": "runner_error",
                         "model": row["model"], "returned_model": None, "trace_returned_model": None,
                         "error_code": type(exc).__name__, "usage": {"input_tokens": None, "output_tokens": None},
                         "latency_ms": None, "trace_id": recorder.trace_id, "trace_complete": False}
                stop_reason = "exception_during_call"
            _append(output / "results.jsonl", entry)
            counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
            if stop_reason:
                break
            if entry["outcome"] == "transport_error":
                stop_reason = "unknown_transport_outcome_no_reissue"
                break
            if arm in HTTP_ARMS and selector_factory is None:
                usage = entry["usage"]
                if (type(usage["input_tokens"]) is not int or type(usage["output_tokens"]) is not int):
                    stop_reason = "token_usage_unknown"
                    break
                used_input += usage["input_tokens"]
                used_output += usage["output_tokens"]
                if (not _input_usage_allowed(arm, usage["input_tokens"], preflight[ordinal - 1]["input_tokens"], input_margin)
                        or usage["input_tokens"] > row["token_caps"]["input_per_call"]
                        or usage["output_tokens"] > row["token_caps"]["output_per_call"]
                        or used_input > row["token_caps"]["input_total"]
                        or used_output > row["token_caps"]["output_total"]):
                    stop_reason = "token_cap_or_count_mismatch"
                    break
            if (entry["model"] != row["model"] or entry["outcome"] == "version_mismatch"
                    or (entry["trace_returned_model"] is not None and entry["trace_returned_model"] != row["returned_model"])):
                stop_reason = "identity_failure"
                break
            if arm in HTTP_ARMS and selector_factory is None and entry["trace_returned_model"] is None:
                stop_reason = "trace_identity_unconfirmed"
                break
            if entry["outcome"] == "ok" and (not entry["trace_complete"] or entry["trace_returned_model"] != row["returned_model"]):
                stop_reason = "trace_identity_unconfirmed"
                break
            if hosted_request_id is not None:
                try:
                    hosted_budget.settle(hosted_ledger, hosted_request_id,
                                         input_tokens=entry["usage"]["input_tokens"],
                                         output_tokens=entry["usage"]["output_tokens"],
                                         elapsed_seconds=monotonic() - call_started)
                except (hosted_budget.BudgetStop, ValueError):
                    stop_reason = "shared_hosted_budget_settlement_unconfirmed"
                    break
            if phase == "warmup" and (entry["outcome"] != "ok" or entry["returned_model"] != row["returned_model"]
                                      or entry["trace_returned_model"] != row["returned_model"]
                                      or not _warmup_usage_valid(arm, entry["usage"])):
                stop_reason = "warmup_failure"
                break
            if phase == "warmup" and arm == "lfm26" and not previous._trace_has_lfm26_warmup_evidence(output / "traces.jsonl", recorder.trace_id):
                stop_reason = "warmup_reasoning_failure"
                break
            # An invalid measured response is retained and the next distinct ID is attempted.
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
    complete = stop_reason is None and len(_rows(output / "results.jsonl")) == 1001
    token_usage = ({"input_tokens": used_input, "output_tokens": used_output}
                   if arm in HTTP_ARMS else _native_token_usage(arm, _rows(output / "results.jsonl")))
    receipt = {"schema": RECEIPT_SCHEMA, "run_id": run_id, "arm": arm, "pass": pass_number,
               "complete": complete, "all_valid": complete and counts == {"ok": 1001},
               "qualification": qualification, "token_usage": token_usage,
               "stop_reason": stop_reason, "outcome_counts": dict(sorted(counts.items())),
               "planned_ids_sha256": pair_hash,
               "journal_sha256": {name: digest(output / name) for name in ("run-metadata.json", "preflight.json", "intents.jsonl", "results.jsonl", "traces.jsonl")}}
    _new_json(output / "receipt.json", receipt)
    return receipt


def verify(output: Path, candidates: Path, launch: Path, gate: Path,
           *, frozen_source: bool = False) -> dict[str, Any]:
    """Audit a receipt; frozen_source accepts hash-pinned historical code bytes."""
    manifest = load_launch(launch, candidates, gate=gate, frozen_source=frozen_source)
    metadata, receipt = _read(output / "run-metadata.json"), _read(output / "receipt.json")
    if not isinstance(metadata, dict) or not isinstance(receipt, dict) or metadata.get("schema") != RUN_SCHEMA or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("three_pass_receipt_shape")
    arm, pass_number = metadata.get("arm"), metadata.get("pass")
    check_gate_scope(gate, launch, arm, pass_number)
    if arm not in ARMS or pass_number not in (1, 2, 3) or receipt.get("arm") != arm or receipt.get("pass") != pass_number:
        raise ValueError("three_pass_receipt_arm_pass")
    if metadata.get("manifest_sha256") != digest(launch) or metadata.get("gate_sha256") != digest(gate) or metadata.get("candidates_sha256") != digest(candidates):
        raise ValueError("three_pass_receipt_binding")
    if receipt.get("qualification") == "real_runtime" and not isinstance(metadata.get("dispatch_claim_sha256"), str):
        raise ValueError("three_pass_dispatch_claim_missing")
    expected_ledger = str(hosted_budget.LEDGER) if arm in {"jev", "gemini"} and receipt.get("qualification") == "real_runtime" else None
    if (metadata.get("hosted_ledger") != expected_ledger
            and not (frozen_source and expected_ledger and isinstance(metadata.get("hosted_ledger"), str)
                     and metadata["hosted_ledger"])):
        raise ValueError("three_pass_hosted_ledger_binding")
    expected_seed = manifest["hosted_budget_seed_sha256"] if expected_ledger else None
    if metadata.get("hosted_budget_seed_sha256") != expected_seed:
        raise ValueError("three_pass_hosted_budget_seed_binding")
    if receipt.get("qualification") != metadata.get("qualification") or receipt["qualification"] not in {"real_runtime", "test_only"}:
        raise ValueError("three_pass_qualification")
    preflight = _read(output / "preflight.json")
    if (not isinstance(preflight, dict) or preflight.get("schema") != "averitec-three-pass-preflight/v1"
            or preflight.get("arm") != arm or preflight.get("complete") is not (receipt["qualification"] == "real_runtime")):
        raise ValueError("three_pass_preflight_receipt")
    if receipt["qualification"] == "real_runtime":
        if preflight.get("calls") != 1001 or not isinstance(preflight.get("records"), list) or len(preflight["records"]) != 1001:
            raise ValueError("three_pass_preflight_count")
        if arm in HTTP_ARMS and (metadata.get("token_preflight_sha256") != manifest["arms"][arm]["token_preflight_sha256"]
                                 or metadata.get("token_caps") != manifest["arms"][arm]["token_caps"]):
            raise ValueError("three_pass_token_preflight_binding")
    elif preflight.get("method") != "injected_test_double" or preflight.get("calls") != 0:
        raise ValueError("three_pass_test_preflight")
    for name, expected in receipt.get("journal_sha256", {}).items():
        if name not in {"run-metadata.json", "preflight.json", "intents.jsonl", "results.jsonl", "traces.jsonl"} or digest(output / name) != expected:
            raise ValueError("three_pass_journal_drift")
    if set(receipt.get("journal_sha256", {})) != {"run-metadata.json", "preflight.json", "intents.jsonl", "results.jsonl", "traces.jsonl"}:
        raise ValueError("three_pass_journal_set")
    pairs = [(r["case_id"], r["id"]) for r in previous.plan_observations(candidates)]
    pair_hash = hashlib.sha256(canonical(pairs).encode()).hexdigest()
    if metadata.get("planned_ids_sha256") != pair_hash or receipt.get("planned_ids_sha256") != pair_hash:
        raise ValueError("three_pass_plan_drift")
    intents, results = _rows(output / "intents.jsonl"), _rows(output / "results.jsonl")
    if len(intents) > 1001 or len(results) > len(intents) or [(r.get("case_id"), r.get("id")) for r in intents[1:]] != pairs[:max(0, len(intents)-1)]:
        raise ValueError("three_pass_intent_plan")
    if (len(intents) == 0 or intents[0].get("phase") != "warmup" or
            any(row.get("ordinal") != n for n, row in enumerate(intents, 1))):
        raise ValueError("three_pass_intent_order")
    if len({r.get("trace_id") for r in intents}) != len(intents):
        raise ValueError("three_pass_trace_ids")
    trace_groups: dict[str, list[dict[str, Any]]] = {}
    for trace in _rows(output / "traces.jsonl"):
        trace_groups.setdefault(str(trace.get("trace_id")), []).append(trace)
    for intent, result in zip(intents, results):
        if (intent.get("phase"), intent.get("case_id"), intent.get("id"), intent.get("trace_id")) != (result.get("phase"), result.get("case_id"), result.get("id"), result.get("trace_id")):
            raise ValueError("three_pass_result_join")
        if result.get("outcome") == "ok" and (result.get("action") not in BINARY_ACTIONS
                                              or result.get("returned_model") != manifest["arms"][arm]["returned_model"]
                                              or result.get("trace_returned_model") != manifest["arms"][arm]["returned_model"]
                                              or result.get("trace_complete") is not True):
            raise ValueError("three_pass_result_invalid")
        if arm in {"jeff", "laya_typed"} and result.get("outcome") == "ok" and not _warmup_usage_valid(arm, result.get("usage")):
            raise ValueError("three_pass_native_usage_semantics")
        trace_complete, raw_model = _trace_response_rows(trace_groups.get(str(result.get("trace_id")), []), result["trace_id"], arm)
        if result.get("trace_complete") is not trace_complete:
            raise ValueError("three_pass_trace_binding")
        if arm == "laya_typed" and trace_complete:
            raw_model = manifest["arms"][arm]["returned_model"]
        if result.get("trace_returned_model") != raw_model:
            raise ValueError("three_pass_trace_identity_binding")
    if receipt.get("complete") is True and (len(intents), len(results)) != (1001, 1001):
        raise ValueError("three_pass_false_complete")
    if receipt.get("complete") is True and any(row.get("outcome") == "transport_error" for row in results):
        raise ValueError("three_pass_transport_continued")
    if arm in HTTP_ARMS and receipt["qualification"] == "real_runtime":
        caps = manifest["arms"][arm]["token_caps"]
        margin = preflight.get("input_margin_tokens")
        output_request_limit = preflight.get("output_request_limit_tokens")
        if (type(margin) is not int or margin < 0 or type(output_request_limit) is not int
                or not 0 < output_request_limit <= caps["output_per_call"]):
            raise ValueError("three_pass_preflight_budget_shape")
        consumed_input = consumed_output = 0
        for ordinal, result in enumerate(results):
            if result.get("outcome") == "transport_error":
                break
            usage = result.get("usage")
            if not isinstance(usage, dict) or type(usage.get("input_tokens")) is not int or type(usage.get("output_tokens")) is not int:
                break
            if not _input_usage_allowed(arm, usage["input_tokens"], preflight["records"][ordinal]["input_tokens"], margin):
                raise ValueError("three_pass_token_count_drift")
            consumed_input += usage["input_tokens"]
            consumed_output += usage["output_tokens"]
            if (usage["input_tokens"] > caps["input_per_call"] or usage["output_tokens"] > output_request_limit
                    or consumed_input > caps["input_total"] or consumed_output > caps["output_total"]):
                raise ValueError("three_pass_token_cap_violation")
        if receipt.get("token_usage") != {"input_tokens": consumed_input, "output_tokens": consumed_output}:
            raise ValueError("three_pass_token_usage_receipt")
    elif arm in NATIVE_ARMS:
        if arm == "jeff" and any(not isinstance(row.get("usage"), dict)
                                 or row["usage"].get("output_tokens") is not None for row in results):
            raise ValueError("three_pass_jeff_output_token_semantics")
        usage_audit = _audit_native_usage(arm, manifest["source_commit"], results,
                                          receipt.get("token_usage"), frozen_source=frozen_source)
        if usage_audit is not None:
            receipt = {**receipt, "historical_native_usage_audit": usage_audit}
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--astra-gate", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--pass-number", type=int, choices=(1, 2, 3))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--laya-checkpoint", type=Path)
    parser.add_argument("--jeff-assets", type=Path)
    parser.add_argument("--jeff-checkpoint", type=Path)
    parser.add_argument("--token-preflight", type=Path)
    parser.add_argument("--hosted-ledger", type=Path)
    parser.add_argument("--dispatch-claim", type=Path)
    args = parser.parse_args()
    if args.plan:
        result = plan(args.launch, args.candidates)
    elif args.execute:
        if not all((args.astra_gate, args.arm, args.pass_number, args.output)):
            parser.error("--execute requires --astra-gate, --arm, --pass-number, and --output")
        result = execute(args.arm, args.pass_number, args.candidates, args.launch, args.astra_gate, args.output,
                         endpoint=args.endpoint, laya_checkpoint=args.laya_checkpoint,
                         jeff_assets=args.jeff_assets, jeff_checkpoint=args.jeff_checkpoint,
                         token_preflight=args.token_preflight, hosted_ledger=args.hosted_ledger,
                         dispatch_claim=args.dispatch_claim)
    else:
        if not args.astra_gate or not args.output:
            parser.error("--verify requires --astra-gate and --output")
        result = verify(args.output, args.candidates, args.launch, args.astra_gate)
    print(canonical(result))


if __name__ == "__main__":
    main()
