"""Pointwise, leakage-safe evidence-selector adapters.

This module deliberately owns only the binary decision boundary.  A caller
provides one claim and one saved candidate at a time; it must not supply gold
labels, ratings, rankings, or any other contextual fields.  The same explicit
instructions are passed through unchanged to every provider.

The native Laya path is offline-testable but not real-runtime qualified here:
its package is imported lazily and a local typed checkpoint is required unless
an agent is injected by a test harness.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from gemini_provider import GeminiController
from providers import Controller, ControllerResult, JevController, OpenAIController


INCLUDE = "include"
EXCLUDE = "exclude"
BINARY_ACTIONS = (INCLUDE, EXCLUDE)
LAYA_TYPED_MODEL = "convaiinnovations/laya-typed-decisions"
LAYA_PACKAGE_VERSION = "0.3.3"
LAYA_TOKEN_BUDGET = (1024, 256)
_LAYA_QUESTION = "evidence_selection"
_TYPED_CHECKPOINT_DIGESTS = {
    "encoder/config.json": ("git-sha1", "d4be4829750fb04c0aa8b9897c3ea827f76c0109"),
    "model.safetensors": ("sha256", "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e"),
    "rl_agent_config.json": ("git-sha1", "5f0e1d5f2366fe8ba2ff330dffaeed53b469e97e"),
    "tokenizer/tokenizer.json": ("git-sha1", "2f4d8583e507b7466d2490e2d6c045647a822698"),
    "tokenizer/tokenizer_config.json": ("git-sha1", "ed1ffabc2ce11120754705709569e365e46da71a"),
}


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_selector_observation(observation: Any) -> dict[str, Any]:
    """Return a copy of the only observation schema allowed to a selector.

    URL is deliberately required: the saved candidate contract supplies it,
    and accepting absent fields would create model-specific observations.
    """
    if not isinstance(observation, dict) or set(observation) != {"claim", "candidate"}:
        raise ValueError("selector_invalid_observation_keys")
    claim = observation.get("claim")
    candidate = observation.get("candidate")
    if not _nonempty_string(claim) or not isinstance(candidate, dict):
        raise ValueError("selector_invalid_observation")
    if set(candidate) != {"id", "text", "url"}:
        raise ValueError("selector_invalid_candidate_keys")
    if not all(_nonempty_string(candidate.get(key)) for key in ("id", "text", "url")):
        raise ValueError("selector_invalid_candidate")
    return {"claim": claim, "candidate": {key: candidate[key] for key in ("id", "text", "url")}}


def _validate_request(observation: Any, instructions: Any, actions: Any) -> tuple[dict[str, Any], str, list[str]]:
    clean = validate_selector_observation(observation)
    if not _nonempty_string(instructions):
        raise ValueError("selector_invalid_instructions")
    if not isinstance(actions, list) or len(actions) != 2 or set(actions) != set(BINARY_ACTIONS):
        raise ValueError("selector_invalid_actions")
    return clean, instructions, list(actions)


class Selector(Protocol):
    def choose(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> ControllerResult: ...


class ProviderSelector:
    """Schema gate around a study provider, preserving ``ControllerResult``."""

    def __init__(self, provider: Controller) -> None:
        self.provider = provider

    def choose(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> ControllerResult:
        clean, fixed_instructions, binary_actions = _validate_request(observation, instructions, actions)
        return self.provider.choose(clean, fixed_instructions, binary_actions)

    @property
    def resolved_profile(self) -> dict[str, Any] | None:
        value = getattr(self.provider, "resolved_profile", None)
        return value if isinstance(value, dict) else None


class _NullTraceSpan:
    def update(self, **_kwargs: Any) -> "_NullTraceSpan":
        return self


@contextmanager
def _trace_request(tracer: Any, *, body: dict[str, Any], model: str):
    if tracer is None:
        yield _NullTraceSpan()
        return
    with tracer.span("controller.request", kind="generation", input=body, model=model) as span:
        yield span


class _CpuFallbackBlocked(RuntimeError):
    pass


class _NoCpuFallbackModel:
    """Prevent Laya's caught-error path from issuing a CPU second forward."""

    def __init__(self, model: Any, blocked: Any) -> None:
        self._model = model
        self._blocked = blocked

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs.get("device")
        if target is not None and str(target).startswith("cpu"):
            self._blocked()
            raise _CpuFallbackBlocked("laya_cpu_fallback_blocked")
        return self._model.to(*args, **kwargs)


def _safe_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _file_digest(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if algorithm == "git-sha1":
        size = path.stat().st_size
        digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    raise ValueError("laya_unknown_digest_algorithm")


class NativeLayaEvidenceSelector:
    """Pinned typed-only Laya binary selector with lossless-token preflight."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        *,
        model: str = LAYA_TYPED_MODEL,
        device: str = "cuda",
        timeout_seconds: float = 120,
        tracer: Any | None = None,
        _agent: Any | None = None,
        _package: Any | None = None,
        _torch: Any | None = None,
    ) -> None:
        if model != LAYA_TYPED_MODEL:
            raise ValueError("laya_typed_model_required")
        if checkpoint_dir is None and _agent is None:
            raise ValueError("laya_local_checkpoint_required")
        self.model, self.checkpoint_dir, self.requested_device = model, (None if checkpoint_dir is None else Path(checkpoint_dir)), device
        self.timeout_seconds, self.tracer = timeout_seconds, tracer
        self._agent, self._package, self._torch = _agent, _package, _torch
        self._config: dict[str, Any] | None = None
        self._actual_device: str | None = None
        self._fallback_blocked = False
        self._injected_agent = _agent is not None
        self._checkpoint_verified = False
        if _agent is not None:
            self._bind_agent()

    def _bind_agent(self) -> None:
        cfg = getattr(self._agent, "cfg", None)
        if not isinstance(cfg, dict) or (cfg.get("max_len"), cfg.get("head_max_len")) != LAYA_TOKEN_BUDGET:
            raise RuntimeError("laya_checkpoint_model_mismatch")
        actual = str(getattr(self._agent, "device", ""))
        if not actual or (self.requested_device.startswith("cuda") and not actual.startswith("cuda")):
            raise RuntimeError("laya_cuda_fallback")
        self._config, self._actual_device = dict(cfg), actual
        if self.requested_device.startswith("cuda") and not isinstance(getattr(self._agent, "model", None), _NoCpuFallbackModel):
            model = getattr(self._agent, "model", None)
            if model is not None:
                self._agent.model = _NoCpuFallbackModel(model, lambda: setattr(self, "_fallback_blocked", True))

    def _ensure_agent(self) -> Any:
        if self._fallback_blocked:
            raise RuntimeError("laya_cuda_fallback_blocked")
        if self._agent is not None:
            return self._agent
        try:
            package = self._package or importlib.import_module("laya")
        except Exception:
            raise RuntimeError("laya_package_unavailable") from None
        if getattr(package, "__version__", None) != LAYA_PACKAGE_VERSION:
            raise RuntimeError("laya_package_version_mismatch")
        if self.checkpoint_dir is None or not self.checkpoint_dir.is_dir():
            raise RuntimeError("laya_checkpoint_missing")
        for relative, (algorithm, expected) in _TYPED_CHECKPOINT_DIGESTS.items():
            path = self.checkpoint_dir / relative
            if not path.is_file():
                raise RuntimeError("laya_checkpoint_incomplete")
            if _file_digest(path, algorithm) != expected:
                raise RuntimeError("laya_checkpoint_digest_mismatch")
        self._checkpoint_verified = True
        try:
            self._agent = package.load(str(self.checkpoint_dir), device=self.requested_device)
        except Exception:
            raise RuntimeError("laya_checkpoint_load_failed") from None
        self._package = package
        self._bind_agent()
        return self._agent

    def _sync(self) -> None:
        if self._actual_device is None or not self._actual_device.startswith("cuda"):
            return
        try:
            runtime = self._torch or importlib.import_module("torch")
            self._torch = runtime
            runtime.cuda.synchronize(self._actual_device)
        except Exception:
            raise RuntimeError("laya_cuda_sync_failed") from None

    def _preflight(self, state: str, instructions: str, actions: list[str]) -> dict[str, int]:
        agent = self._ensure_agent()
        tokenizer = getattr(agent, "tok", None)
        if tokenizer is None:
            raise RuntimeError("laya_tokenizer_unavailable")
        mask, marker = getattr(tokenizer, "mask_token", None), getattr(tokenizer, "mask_token_id", None)
        cls_id, sep_id = getattr(tokenizer, "cls_token_id", None), getattr(tokenizer, "sep_token_id", None)
        if not isinstance(mask, str) or not isinstance(marker, int) or not isinstance(cls_id, int) or not isinstance(sep_id, int):
            raise RuntimeError("laya_tokenizer_special_tokens")

        def encode(text: str) -> list[int]:
            encoded = tokenizer(text, add_special_tokens=False)
            values = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
            if not isinstance(values, list) or any(not isinstance(item, int) for item in values):
                raise RuntimeError("laya_tokenizer_output_invalid")
            return values

        instruction_ids = encode("choice question: " + instructions.replace(mask, " "))
        raw_options = [[marker] + encode(" " + action.replace(mask, " ")) for action in actions]
        if any(len(option) > 49 for option in raw_options):
            raise ValueError("laya_preflight_option_truncated")
        option_total = sum(len(option) for option in raw_options)
        head_budget = LAYA_TOKEN_BUDGET[1] - option_total
        if head_budget < 16 or len(instruction_ids) > max(8, head_budget):
            raise ValueError("laya_preflight_instructions_truncated")
        head_tokens = 1 + len(instruction_ids) + 1 + option_total + 1
        state_ids = encode(state.replace(mask, " "))
        state_budget = LAYA_TOKEN_BUDGET[0] - head_tokens - 1
        if len(state_ids) > state_budget:
            raise ValueError("laya_preflight_state_truncated")
        return {"instruction_tokens": len(instruction_ids), "state_tokens": len(state_ids), "total_tokens": head_tokens + len(state_ids) + 1}

    def preflight(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> dict[str, int]:
        """Validate the exact typed request without running ``predict``."""
        clean, fixed_instructions, binary_actions = _validate_request(observation, instructions, actions)
        return self._preflight(_compact_json(clean), fixed_instructions, binary_actions)

    def identity(self) -> dict[str, Any]:
        self._ensure_agent()
        actual_package_version = getattr(self._package, "__version__", None)
        verified = self._checkpoint_verified and not self._injected_agent
        return {"adapter": "native_laya_evidence_selector/v1", "model": self.model, "returned_model": self.model, "returned_model_identity_source": "pinned_local_typed_checkpoint" if verified else "test_injected_unverified", "package": "laya", "package_version": actual_package_version, "requested_device": self.requested_device, "actual_device": self._actual_device, "max_tokens": 1024, "head_max_tokens": 256, "cpu_fallback_policy": "blocked_before_retry_forward", "output_token_semantics": "zero_non_autoregressive_tokens", "checkpoint_verification": "verified" if verified else "test_injected_unverified", "runtime_verification": "offline_only_unverified"}

    def choose(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> ControllerResult:
        started = perf_counter()
        try:
            clean, fixed_instructions, binary_actions = _validate_request(observation, instructions, actions)
            state = _compact_json(clean)
            preflight = self._preflight(state, fixed_instructions, binary_actions)
            agent = self._ensure_agent()
        except (ValueError, RuntimeError) as exc:
            return ControllerResult(None, "invalid_output", (perf_counter() - started) * 1000, None, 0, self.model, error_code=str(exc), returned_model=self.model)
        questions = {_LAYA_QUESTION: {"type": "choice", "instructions": fixed_instructions, "criteria": {action: None for action in binary_actions}}}
        body = {"model": self.model, "state": state, "questions": questions}
        with _trace_request(self.tracer, body=body, model=self.model) as trace:
            try:
                self._sync()
                started = perf_counter()
                payload = agent.predict(state, questions)
                self._sync()
                latency = (perf_counter() - started) * 1000
            except Exception:
                code = "laya_cuda_fallback_blocked" if self._fallback_blocked else "laya_native_inference_failed"
                trace.update(metadata={"outcome": "transport_error", "error_code": code})
                return ControllerResult(None, "transport_error", 0.0, None, 0, self.model, error_code=code, returned_model=self.model)
            usage = payload.get("usage") if isinstance(payload, dict) else None
            input_tokens = usage.get("input_tokens") if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int) and not isinstance(usage.get("input_tokens"), bool) and usage.get("input_tokens") >= 0 else None
            trace.update(output={"answers": payload.get("answers") if isinstance(payload, dict) else None, "usage": usage}, usage={"input": input_tokens, "output": 0} if input_tokens is not None else {"output": 0}, metadata={"native_forward_latency_ms": latency, "preflight_total_tokens": preflight["total_tokens"]})
        answer = payload.get("answers", {}).get(_LAYA_QUESTION) if isinstance(payload, dict) and isinstance(payload.get("answers"), dict) else None
        if not isinstance(answer, dict) or answer.get("choice") not in binary_actions:
            return ControllerResult(None, "invalid_output", latency, input_tokens, 0, self.model, error_code="laya_choice_invalid", returned_model=self.model)
        raw = answer.get("probabilities")
        if not isinstance(raw, dict) or set(raw) != set(binary_actions):
            return ControllerResult(None, "invalid_output", latency, input_tokens, 0, self.model, error_code="laya_probabilities_invalid", returned_model=self.model)
        probabilities = {action: _safe_probability(raw[action]) for action in binary_actions}
        if any(value is None for value in probabilities.values()) or abs(sum(probabilities.values()) - 1.0) > 0.02:  # type: ignore[arg-type]
            return ControllerResult(None, "invalid_output", latency, input_tokens, 0, self.model, error_code="laya_probabilities_invalid", returned_model=self.model)
        confidence = _safe_probability(answer.get("confidence"))
        valid_usage = (isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int)
                       and not isinstance(usage.get("input_tokens"), bool) and usage["input_tokens"] >= 0
                       and isinstance(usage.get("output_tokens"), int) and not isinstance(usage.get("output_tokens"), bool)
                       and usage["output_tokens"] == 0)
        if not valid_usage:
            return ControllerResult(None, "invalid_output", latency, input_tokens, 0, self.model, error_code="laya_usage_invalid", returned_model=self.model)
        if confidence is None or confidence > 1:
            return ControllerResult(None, "invalid_output", latency, input_tokens, 0, self.model, error_code="laya_confidence_invalid", returned_model=self.model)
        return ControllerResult(answer["choice"], "ok", latency, input_tokens, 0, self.model, probabilities=probabilities, confidence=confidence, returned_model=self.model)  # type: ignore[arg-type]


def create_selector(
    controller: str,
    endpoint: str | None = None,
    keys: Mapping[str, str] | None = None,
    tracer: Any | None = None,
    *,
    laya_checkpoint: str | Path | None = None,
    laya_device: str = "cuda",
    timeout_seconds: float = 30,
    laya_agent: Any | None = None,
    laya_package: Any | None = None,
    laya_torch: Any | None = None,
    seed: int = 20260919,
) -> Selector:
    """Create one approved selector arm; keys are injected, never read here."""
    if not isinstance(controller, str):
        raise ValueError("selector_unknown_controller")
    supplied = dict(keys or {})
    if controller == "jev":
        return ProviderSelector(JevController(key=supplied.get("jev") or supplied.get("TYPESAFE_API_KEY"), timeout_seconds=timeout_seconds, tracer=tracer))
    if controller == "qwen":
        if not endpoint:
            raise ValueError("selector_endpoint_required")
        return ProviderSelector(OpenAIController(endpoint, "Qwen/Qwen3.5-4B", seed=seed, timeout_seconds=timeout_seconds, tracer=tracer, generation_profile="baseline"))
    if controller == "lfm":
        if not endpoint:
            raise ValueError("selector_endpoint_required")
        return ProviderSelector(OpenAIController(endpoint, "LiquidAI/LFM2.5-1.2B-Instruct", seed=seed, timeout_seconds=timeout_seconds, tracer=tracer, generation_profile="lfm_native"))
    if controller == "lfm26":
        if not endpoint:
            raise ValueError("selector_endpoint_required")
        return ProviderSelector(OpenAIController(endpoint, "LiquidAI/LFM2.5-2.6B", seed=seed, timeout_seconds=timeout_seconds, tracer=tracer, generation_profile="lfm26_native"))
    if controller == "gemini":
        return ProviderSelector(GeminiController(key=supplied.get("gemini") or supplied.get("GEMINI_API_KEY"), seed=seed, timeout_seconds=timeout_seconds, tracer=tracer, generation_profile="gemini_native"))
    if controller in {"laya_typed", "laya"}:
        return NativeLayaEvidenceSelector(laya_checkpoint, device=laya_device, timeout_seconds=timeout_seconds, tracer=tracer, _agent=laya_agent, _package=laya_package, _torch=laya_torch)
    raise ValueError("selector_unknown_controller")
