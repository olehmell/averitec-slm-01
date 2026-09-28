"""Native Laya controller for the isolated checkpoint extension.

The adapter deliberately keeps Laya and torch imports lazy.  Unit tests and
planning commands can therefore import this module without either dependency.
The runtime performs one native ``choice`` forward pass and never retries.
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
from typing import Any

from engine import ACTIONS
from prompt_variants import present_observation
from providers import ControllerResult


PACKAGE_VERSION = "0.3.3"
BASE_MODEL_ID = "convaiinnovations/laya"
TYPED_MODEL_ID = "convaiinnovations/laya-typed-decisions"
MODEL_IDS = (BASE_MODEL_ID, TYPED_MODEL_ID)
EXPECTED_BUDGETS = {
    BASE_MODEL_ID: (512, 192),
    TYPED_MODEL_ID: (1024, 256),
}
PRESENTATION_NAME = "v2"
INSTRUCTION_PROFILE = "laya_native_v1"
RUNTIME_WRAPPER = "block_laya_cpu_retry_v1"
QUESTION_ID = "next_action"

# This is intentionally shorter than v4 while preserving the same ordered
# decision procedure.  Rules live here, while the nine criteria remain bare
# action labels so Laya's fixed head budget is not consumed by duplicated prose.
NATIVE_INSTRUCTIONS = """Choose one action NOW. Tools must run in order: decompose, queries, retrieve, qa, coverage, select, verdict. Apply the first matching rule: A) If remaining_decisions < 8 - count(completed_tools), choose abort. B) If failed_attempts_on_pending_tool = 2, choose abort. C) If count(completed_tools) = 7, choose finish. D) Otherwise choose pending_tool. ok or empty completes a tool; timeout or invalid leaves it pending. After one failure retry it; after two choose abort. pending_tool is the first unfinished tool. remaining_decisions includes now and finish. Metrics, including zero or null, never permit skipping. Return one action only."""


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _action_error(actions: list[str]) -> str | None:
    if (
        not isinstance(actions, list)
        or len(actions) != len(ACTIONS)
        or set(actions) != set(ACTIONS)
        or any(not isinstance(action, str) or not action for action in actions)
        or len(set(actions)) != len(actions)
    ):
        return "laya_invalid_actions"
    return None


class _NullTraceSpan:
    def update(self, **_kwargs: Any) -> "_NullTraceSpan":
        return self


class _CpuFallbackBlocked(RuntimeError):
    """Private sentinel raised before Laya can perform its CPU retry."""


class _NoCpuFallbackModel:
    """Delegate model calls but reject Laya's fallback transfer to CPU."""

    def __init__(self, model: Any, on_block: Any) -> None:
        self._model = model
        self._on_block = on_block

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs.get("device")
        if target is not None and str(target).startswith("cpu"):
            self._on_block()
            raise _CpuFallbackBlocked("laya_cpu_fallback_blocked")
        return self._model.to(*args, **kwargs)


@contextmanager
def _trace_request(tracer: Any, *, body: dict[str, Any], model: str):
    if tracer is None:
        yield _NullTraceSpan()
        return
    with tracer.span("controller.request", kind="generation", input=body, model=model) as span:
        yield span


def _safe_number(value: Any, *, low: float = 0.0, high: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < low or (high is not None and result > high):
        return None
    return result


def _safe_usage(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("input_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class NativeLayaController:
    """One-forward native Laya controller with lossless-token preflight."""

    def __init__(
        self,
        model: str,
        checkpoint_dir: str | Path | None,
        *,
        device: str = "cuda",
        timeout_seconds: float = 120,
        tracer: Any | None = None,
        _agent: Any | None = None,
        _package: Any | None = None,
        _torch: Any | None = None,
    ) -> None:
        if model not in MODEL_IDS:
            raise ValueError("laya_unknown_model")
        if checkpoint_dir is None and _agent is None:
            raise ValueError("laya_local_checkpoint_required")
        self.model = model
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.requested_device = device
        self.timeout_seconds = timeout_seconds
        self.tracer = tracer
        self._agent = _agent
        self._package = _package
        self._torch = _torch
        self._closed = False
        self._actual_device: str | None = None
        self._config: dict[str, Any] | None = None
        self._gpu_fallback_blocked = False
        if _agent is not None:
            self._bind_loaded_agent()

    def _bind_loaded_agent(self) -> None:
        if self._agent is None:
            raise RuntimeError("laya_agent_unavailable")
        cfg = getattr(self._agent, "cfg", None)
        if not isinstance(cfg, dict):
            raise RuntimeError("laya_config_unavailable")
        max_len, head_max_len = cfg.get("max_len"), cfg.get("head_max_len")
        if (
            not isinstance(max_len, int)
            or isinstance(max_len, bool)
            or not isinstance(head_max_len, int)
            or isinstance(head_max_len, bool)
            or max_len <= 0
            or head_max_len <= 0
        ):
            raise RuntimeError("laya_config_invalid")
        if (max_len, head_max_len) != EXPECTED_BUDGETS[self.model]:
            raise RuntimeError("laya_checkpoint_model_mismatch")
        actual = str(getattr(self._agent, "device", ""))
        if not actual:
            raise RuntimeError("laya_device_unavailable")
        if self.requested_device.startswith("cuda") and not actual.startswith("cuda"):
            raise RuntimeError("laya_cuda_fallback")
        self._actual_device = actual
        self._config = dict(cfg)
        if self.requested_device.startswith("cuda"):
            model = getattr(self._agent, "model", None)
            if model is not None and not isinstance(model, _NoCpuFallbackModel):
                def mark_blocked() -> None:
                    self._gpu_fallback_blocked = True

                self._agent.model = _NoCpuFallbackModel(model, mark_blocked)

    def _ensure_agent(self) -> Any:
        if self._closed:
            raise RuntimeError("laya_controller_closed")
        if self._gpu_fallback_blocked:
            raise RuntimeError("laya_cuda_fallback_blocked")
        if self._agent is not None:
            return self._agent
        try:
            package = self._package or importlib.import_module("laya")
        except Exception:
            raise RuntimeError("laya_package_unavailable") from None
        if getattr(package, "__version__", None) != PACKAGE_VERSION:
            raise RuntimeError("laya_package_version_mismatch")
        if self.checkpoint_dir is None or not self.checkpoint_dir.is_dir():
            raise RuntimeError("laya_checkpoint_missing")
        required = (
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "encoder/config.json",
        )
        if any(not (self.checkpoint_dir / relative).is_file() for relative in required):
            # Laya falls back to the encoder repo when tokenizer/encoder assets
            # are absent.  Reject incomplete snapshots to preserve offline runs.
            raise RuntimeError("laya_checkpoint_incomplete")
        try:
            self._agent = package.load(str(self.checkpoint_dir), device=self.requested_device)
        except Exception:
            raise RuntimeError("laya_checkpoint_load_failed") from None
        self._package = package
        self._bind_loaded_agent()
        return self._agent

    def _torch_runtime(self) -> Any:
        if self._torch is not None:
            return self._torch
        try:
            self._torch = importlib.import_module("torch")
        except Exception:
            raise RuntimeError("laya_torch_unavailable") from None
        return self._torch

    def _synchronize(self) -> None:
        if self._actual_device is None or not self._actual_device.startswith("cuda"):
            return
        try:
            self._torch_runtime().cuda.synchronize(self._actual_device)
        except Exception:
            raise RuntimeError("laya_cuda_sync_failed") from None

    def _present(self, observation: dict[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            presented = present_observation(observation, "v4")
            return presented, _compact_json(presented)
        except (TypeError, ValueError):
            raise ValueError("laya_invalid_observation") from None

    def _preflight_presented(self, state_text: str, actions: list[str]) -> dict[str, Any]:
        invalid = _action_error(actions)
        if invalid:
            raise ValueError(invalid)
        agent = self._ensure_agent()
        tokenizer = getattr(agent, "tok", None)
        if tokenizer is None or self._config is None:
            raise RuntimeError("laya_tokenizer_unavailable")
        max_len = int(self._config["max_len"])
        head_max_len = int(self._config["head_max_len"])
        mask_token = getattr(tokenizer, "mask_token", None)
        marker_id = getattr(tokenizer, "mask_token_id", None)
        cls_id = getattr(tokenizer, "cls_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        if not isinstance(mask_token, str) or None in (marker_id, cls_id, sep_id):
            raise RuntimeError("laya_tokenizer_special_tokens")

        def encode(text: str) -> list[int]:
            value = tokenizer(text, add_special_tokens=False)
            # Hugging Face BatchEncoding follows the Mapping/UserDict contract,
            # rather than inheriting from the builtin dict type.
            ids = value.get("input_ids") if isinstance(value, Mapping) else None
            if not isinstance(ids, list) or any(not isinstance(item, int) for item in ids):
                raise RuntimeError("laya_tokenizer_output_invalid")
            return ids

        # Mirror laya.common.build_sequence v0.3.3 exactly, while retaining the
        # untruncated components so any truncation is rejected before inference.
        instruction_ids = encode(
            "choice question: " + NATIVE_INSTRUCTIONS.replace(mask_token, " ")
        )
        raw_options = [
            [marker_id] + encode(" " + action.replace(mask_token, " ")) for action in actions
        ]
        if any(len(option) > 49 for option in raw_options):
            raise ValueError("laya_preflight_option_truncated")
        option_ids = [option[:49] for option in raw_options]
        option_budget = head_max_len - sum(len(option) for option in option_ids)
        recapped = False
        if option_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
            recapped = any(len(option) > per for option in option_ids)
            option_ids = [option[:per] for option in option_ids]
            option_budget = head_max_len - sum(len(option) for option in option_ids)
        if recapped:
            raise ValueError("laya_preflight_options_truncated")
        instruction_limit = max(8, option_budget)
        if len(instruction_ids) > instruction_limit:
            raise ValueError("laya_preflight_instructions_truncated")

        head_tokens = 1 + len(instruction_ids) + 1 + sum(len(item) for item in option_ids) + 1
        state_ids = encode(state_text.replace(mask_token, " "))
        state_budget = max(0, max_len - head_tokens - 1)
        if len(state_ids) > state_budget:
            raise ValueError("laya_preflight_state_truncated")
        total_tokens = head_tokens + len(state_ids) + 1
        if total_tokens > max_len:
            raise ValueError("laya_preflight_total_truncated")
        return {
            "lossless": True,
            "model": self.model,
            "max_tokens": max_len,
            "head_max_tokens": head_max_len,
            "instruction_tokens": len(instruction_ids),
            "option_tokens": sum(len(item) for item in option_ids),
            "state_tokens": len(state_ids),
            "state_budget": state_budget,
            "total_tokens": total_tokens,
            "action_order_sha256": _digest_text(_compact_json(actions)),
            "state_sha256": _digest_text(state_text),
        }

    def preflight(self, observation: dict[str, Any], actions: list[str]) -> dict[str, Any]:
        """Prove the exact native request fits without any Laya truncation."""
        _presented, state_text = self._present(observation)
        return self._preflight_presented(state_text, actions)

    def identity(self) -> dict[str, Any]:
        """Return safe, reproducibility-relevant runtime identity."""
        self._ensure_agent()
        assert self._config is not None and self._actual_device is not None
        package_version = getattr(self._package, "__version__", PACKAGE_VERSION)
        return {
            "adapter": "native_laya_controller/v1",
            "model": self.model,
            "expected_returned_model": self.model,
            "returned_model_identity_source": "pinned_local_manifest",
            "cpu_fallback_policy": "blocked_before_retry_forward",
            "runtime_wrapper": RUNTIME_WRAPPER,
            "package": "laya",
            "package_version": package_version,
            "checkpoint_source": "local_snapshot",
            "checkpoint_name": self.checkpoint_dir.name if self.checkpoint_dir is not None else "injected",
            "requested_device": self.requested_device,
            "actual_device": self._actual_device,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self._config["max_len"],
            "head_max_tokens": self._config["head_max_len"],
            "instruction_profile": INSTRUCTION_PROFILE,
            "instructions_sha256": _digest_text(NATIVE_INSTRUCTIONS),
            "observation_presentation": PRESENTATION_NAME,
            "question_type": "choice",
            "output_token_semantics": "zero_non_autoregressive_tokens",
            "confidence_semantics": "native_normalized_entropy_not_chosen_probability",
        }

    def choose(
        self, observation: dict[str, Any], instructions: str, actions: list[str]
    ) -> ControllerResult:
        """Run exactly one native forward pass; ``instructions`` is intentionally ignored."""
        del instructions  # The fixed, hash-bound native profile is authoritative.
        presented, state_text = self._present(observation)
        preflight = self._preflight_presented(state_text, actions)
        agent = self._ensure_agent()
        criteria = {action: None for action in actions}
        questions = {QUESTION_ID: {
            "type": "choice",
            "instructions": NATIVE_INSTRUCTIONS,
            "criteria": criteria,
        }}
        trace_body = {
            "model": self.model,
            "state": state_text,
            "questions": questions,
            "presentation": PRESENTATION_NAME,
            "instruction_profile": INSTRUCTION_PROFILE,
            "runtime_wrapper": RUNTIME_WRAPPER,
        }
        with _trace_request(self.tracer, body=trace_body, model=self.model) as trace:
            try:
                self._synchronize()
                started = perf_counter()
                payload = agent.predict(state_text, questions)
                self._synchronize()
                latency_ms = (perf_counter() - started) * 1000
            except Exception:
                error_code = ("laya_cuda_fallback_blocked" if self._gpu_fallback_blocked
                              else "laya_native_inference_failed")
                trace.update(metadata={"outcome": "transport_error", "error_code": error_code})
                return ControllerResult(
                    action=None, outcome="transport_error", latency_ms=0.0,
                    input_tokens=None, output_tokens=0, model=self.model,
                    error_code=error_code,
                    returned_model=self.model,
                )

            safe_payload = {
                "model": payload.get("model") if isinstance(payload, dict) else None,
                "answers": ({QUESTION_ID: payload.get("answers", {}).get(QUESTION_ID)}
                            if isinstance(payload, dict) and isinstance(payload.get("answers"), dict)
                            else None),
                "usage": payload.get("usage") if isinstance(payload, dict) else None,
            }
            input_tokens = _safe_usage(payload.get("usage")) if isinstance(payload, dict) else None
            trace.update(
                output=safe_payload,
                usage={"input": input_tokens, "output": 0} if input_tokens is not None else {"output": 0},
                metadata={
                    "outcome": "received",
                    "confidence_semantics": "native_normalized_entropy_not_chosen_probability",
                    "preflight_total_tokens": preflight["total_tokens"],
                },
            )

        answer = (payload.get("answers", {}).get(QUESTION_ID)
                  if isinstance(payload, dict) and isinstance(payload.get("answers"), dict) else None)
        if not isinstance(answer, dict):
            return ControllerResult(None, "invalid_output", latency_ms, input_tokens, 0,
                                    self.model, error_code="laya_answer_missing",
                                    returned_model=self.model)
        action = answer.get("choice")
        raw_probabilities = answer.get("probabilities")
        confidence = _safe_number(answer.get("confidence"), high=1.0)
        if not isinstance(action, str) or action not in actions:
            return ControllerResult(None, "invalid_output", latency_ms, input_tokens, 0,
                                    self.model, error_code="laya_choice_invalid",
                                    returned_model=self.model)
        if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != set(actions):
            return ControllerResult(None, "invalid_output", latency_ms, input_tokens, 0,
                                    self.model, error_code="laya_probabilities_invalid",
                                    returned_model=self.model)
        probabilities: dict[str, float] = {}
        for name in actions:
            value = _safe_number(raw_probabilities.get(name))
            if value is None:
                return ControllerResult(None, "invalid_output", latency_ms, input_tokens, 0,
                                        self.model, error_code="laya_probabilities_invalid",
                                        returned_model=self.model)
            probabilities[name] = value
        if abs(sum(probabilities.values()) - 1.0) > 0.02:
            return ControllerResult(None, "invalid_output", latency_ms, input_tokens, 0,
                                    self.model, error_code="laya_probabilities_invalid",
                                    returned_model=self.model)
        if confidence is None:
            return ControllerResult(None, "invalid_output", latency_ms, input_tokens, 0,
                                    self.model, error_code="laya_confidence_invalid",
                                    returned_model=self.model)
        output_tokens = (payload.get("usage", {}).get("output_tokens")
                         if isinstance(payload.get("usage"), dict) else None)
        if output_tokens != 0:
            return ControllerResult(None, "invalid_output", latency_ms, input_tokens, 0,
                                    self.model, error_code="laya_output_usage_invalid",
                                    returned_model=self.model)
        return ControllerResult(
            action=action,
            outcome="ok",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=0,
            model=self.model,
            probabilities=probabilities,
            confidence=confidence,
            returned_model=self.model,
        )

    @property
    def expected_returned_model(self) -> str:
        """Pinned manifest identity; native payload labels are not deployment IDs."""
        return self.model

    def close(self) -> None:
        """Release the model and best-effort CUDA cache without surfacing teardown errors."""
        if self._closed:
            return
        actual = self._actual_device
        self._agent = None
        self._closed = True
        if actual is not None and actual.startswith("cuda"):
            try:
                runtime = self._torch_runtime()
                runtime.cuda.synchronize(actual)
                runtime.cuda.empty_cache()
            except Exception:
                pass
