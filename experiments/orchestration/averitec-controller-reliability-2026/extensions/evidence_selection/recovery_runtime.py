"""Dedicated v2 recovery adapters for the independent selector rerun.

They intentionally do not reuse mutable generation-profile definitions from
the original study.  Each adapter records its complete request policy and
keeps parsed provider responses in the local trace journal for identity audit.
"""
from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from providers import ControllerResult
from selector_runtime import BINARY_ACTIONS, validate_selector_observation


GEMINI_MODEL = "gemini-3.1-flash-lite"
QWEN_MODEL = "Qwen/Qwen3.5-4B"
LFM26_MODEL = "LiquidAI/LFM2.5-2.6B"
RECOVERY_PROFILE_IDENTITIES = {
    "gemini": {"name": "gemini_native_timeout120_v2", "provider": "gemini_generate_content", "model": GEMINI_MODEL,
               "expected_version": GEMINI_MODEL, "generation_config": {"temperature": 1.0, "maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "minimal"}},
               "timeouts_seconds": {"warmup": 120, "measured": 120}},
    "qwen": {"name": "qwen_baseline_warmup180_v2", "provider": "openai_chat_completions", "model": QWEN_MODEL,
             "request_parameters": {"temperature": 0, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False}},
             "timeouts_seconds": {"warmup": 180, "measured": 30}},
    "lfm26": {"name": "lfm26_native_4096_v2", "provider": "openai_chat_completions", "model": LFM26_MODEL,
              "request_parameters": {"temperature": 0.1, "max_tokens": 4096, "top_k": 50, "repetition_penalty": 1.1},
              "reasoning_mode": "always", "timeouts_seconds": {"warmup": 60, "measured": 60}},
}


def _safe_model(value: Any) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 100 and all(c.isalnum() or c in "._-/" for c in value) else None


def _usage(value: Any, incoming: str, outgoing: str) -> tuple[int | None, int | None]:
    if not isinstance(value, dict):
        return None, None
    def number(name: str) -> int | None:
        item = value.get(name)
        return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None
    return number(incoming), number(outgoing)


def _gemini_usage(value: Any) -> tuple[int | None, int | None]:
    """Match the original adapter's reported-token semantics, including thought tokens."""
    if not isinstance(value, dict):
        return None, None
    prompt, candidate = _usage(value, "promptTokenCount", "candidatesTokenCount")
    if candidate is None:
        return prompt, None
    if "thoughtsTokenCount" not in value:
        return prompt, candidate
    thoughts = value.get("thoughtsTokenCount")
    if not isinstance(thoughts, int) or isinstance(thoughts, bool) or thoughts < 0:
        return prompt, None
    return prompt, candidate + thoughts


def _result(action: str | None, outcome: str, started: float, model: str, *, input_tokens: int | None = None,
            output_tokens: int | None = None, error_code: str | None = None, returned_model: str | None = None,
            latency_ms: float | None = None) -> ControllerResult:
    return ControllerResult(action, outcome, (perf_counter() - started) * 1000 if latency_ms is None else latency_ms, input_tokens, output_tokens, model,
                            error_code=error_code, returned_model=returned_model)  # type: ignore[arg-type]


def _post(url: str, body: dict[str, Any], headers: Mapping[str, str], timeout: float) -> dict[str, Any]:
    request = Request(url, data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
                      headers={"Content-Type": "application/json", **headers}, method="POST")
    with urlopen(request, timeout=timeout) as response:  # nosec B310: caller-provided local deployment endpoint
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("response_not_object")
    return value


class _V2Base:
    model: str
    tracer: Any
    timeout_seconds: float
    expected_returned_model: str

    def _trace(self, body: dict[str, Any]):
        return self.tracer.span("controller.request", kind="generation", input=body, model=self.model)


class GeminiSelectorV2(_V2Base):
    """Gemini native policy with a fixed 120-second timeout for every call."""

    def __init__(self, *, key: str | None = None, tracer: Any | None = None, seed: int = 20260919) -> None:
        self.model = GEMINI_MODEL
        self.expected_returned_model = GEMINI_MODEL
        self.key = key if key is not None else os.environ.get("GEMINI_API_KEY")
        self.tracer, self.seed, self.timeout_seconds = tracer, seed, 120.0
        self.current_phase = "measured"

    @property
    def resolved_profile(self) -> dict[str, Any]:
        return json.loads(json.dumps(RECOVERY_PROFILE_IDENTITIES["gemini"]))

    def set_phase(self, phase: str) -> None:
        if phase not in {"warmup", "measured"}:
            raise ValueError("v2_unknown_phase")
        self.current_phase, self.timeout_seconds = phase, 120.0

    def choose(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> ControllerResult:
        started = perf_counter()
        if actions != list(BINARY_ACTIONS):
            return _result(None, "invalid_output", started, self.model, error_code="invalid_actions")
        try:
            clean = validate_selector_observation(observation)
        except ValueError:
            return _result(None, "invalid_output", started, self.model, error_code="invalid_input")
        if not isinstance(instructions, str) or not instructions or not self.key:
            return _result(None, "transport_error", started, self.model, error_code="missing_api_key")
        body = {"systemInstruction": {"parts": [{"text": instructions}]}, "contents": [{"role": "user", "parts": [{"text": "Observation:\n" + json.dumps(clean, separators=(",", ":")) + "\nAllowed actions (preserve this vocabulary and order): " + json.dumps(actions) + "\nReturn exactly one JSON object with the action."}]}],
                "generationConfig": {"temperature": 1.0, "maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "minimal"}, "seed": self.seed, "responseMimeType": "application/json", "responseJsonSchema": {"type": "object", "properties": {"action": {"type": "string", "enum": actions}}, "required": ["action"], "additionalProperties": False}}}
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + self.model + ":generateContent"
        with self._trace(body) as trace:
            started = perf_counter()
            try:
                payload = _post(url, body, {"x-goog-api-key": self.key}, self.timeout_seconds)
            except HTTPError as exc:
                trace.update(metadata={"outcome": "transport_error", "error_code": "http_" + str(exc.code)})
                return _result(None, "transport_error", started, self.model, error_code="http_" + str(exc.code))
            except (URLError, OSError):
                trace.update(metadata={"outcome": "transport_error", "error_code": "transport_failure"})
                return _result(None, "transport_error", started, self.model, error_code="transport_failure")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                trace.update(metadata={"outcome": "invalid_output", "error_code": "invalid_json"})
                return _result(None, "invalid_output", started, self.model, error_code="invalid_json")
            input_tokens, output_tokens = _gemini_usage(payload.get("usageMetadata"))
            raw = {key: payload.get(key) for key in ("candidates", "promptFeedback", "usageMetadata", "modelVersion")}
            trace.update(output=raw, usage={key: value for key, value in (("input", input_tokens), ("output", output_tokens)) if value is not None})
            request_latency_ms = (perf_counter() - started) * 1000
        returned = _safe_model(payload.get("modelVersion"))
        if payload.get("modelVersion") != self.expected_returned_model:
            if not isinstance(payload.get("modelVersion"), str):
                return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="missing_model_identity", returned_model=returned, latency_ms=request_latency_ms)
            return _result(None, "version_mismatch", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="model_mismatch", returned_model=returned, latency_ms=request_latency_ms)
        prompt_feedback = payload.get("promptFeedback")
        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
            return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="prompt_blocked", returned_model=returned, latency_ms=request_latency_ms)
        candidates = payload.get("candidates")
        candidate = candidates[0] if isinstance(candidates, list) and len(candidates) == 1 and isinstance(candidates[0], dict) else None
        ratings = candidate.get("safetyRatings") if isinstance(candidate, dict) else None
        if isinstance(ratings, list) and any(isinstance(rating, dict) and rating.get("blocked") is True for rating in ratings):
            return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="safety_blocked", returned_model=returned, latency_ms=request_latency_ms)
        if not isinstance(candidate, dict) or candidate.get("finishReason") != "STOP":
            return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="truncated_or_invalid", returned_model=returned, latency_ms=request_latency_ms)
        parts = candidate.get("content", {}).get("parts") if isinstance(candidate.get("content"), dict) else None
        text_parts: list[str] = []
        valid_parts = isinstance(parts, list)
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    valid_parts = False
                    break
                if part.get("thought") is True:
                    continue
                value = part.get("text")
                if not isinstance(value, str):
                    valid_parts = False
                    break
                text_parts.append(value)
        text = "".join(text_parts) if valid_parts else ""
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if input_tokens is None or output_tokens is None or not valid_parts or not isinstance(decoded, dict) or set(decoded) != {"action"} or decoded.get("action") not in actions:
            return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_action_or_usage", returned_model=returned, latency_ms=request_latency_ms)
        return _result(decoded["action"], "ok", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, returned_model=returned, latency_ms=request_latency_ms)


class OpenAISelectorV2(_V2Base):
    """Qwen/LFM26 recovery deployment with a sealed local request policy."""

    def __init__(self, controller: str, endpoint: str, *, tracer: Any | None = None, seed: int = 20260919) -> None:
        if controller not in {"qwen", "lfm26"} or not isinstance(endpoint, str) or not endpoint:
            raise ValueError("v2_endpoint_or_controller")
        self.controller, self.model, self.endpoint = controller, (QWEN_MODEL if controller == "qwen" else LFM26_MODEL), endpoint.rstrip("/") + ("" if endpoint.rstrip("/").endswith("/chat/completions") else "/chat/completions")
        self.expected_returned_model, self.tracer, self.seed = self.model, tracer, seed
        self.current_phase, self.timeout_seconds = "measured", (30.0 if controller == "qwen" else 60.0)

    @property
    def resolved_profile(self) -> dict[str, Any]:
        return json.loads(json.dumps(RECOVERY_PROFILE_IDENTITIES[self.controller]))

    def set_phase(self, phase: str) -> None:
        if phase not in {"warmup", "measured"}:
            raise ValueError("v2_unknown_phase")
        self.current_phase = phase
        self.timeout_seconds = 180.0 if self.controller == "qwen" and phase == "warmup" else (30.0 if self.controller == "qwen" else 60.0)

    def choose(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> ControllerResult:
        started = perf_counter()
        if actions != list(BINARY_ACTIONS):
            return _result(None, "invalid_output", started, self.model, error_code="invalid_actions")
        try:
            clean = validate_selector_observation(observation)
        except ValueError:
            return _result(None, "invalid_output", started, self.model, error_code="invalid_input")
        if not isinstance(instructions, str) or not instructions:
            return _result(None, "invalid_output", started, self.model, error_code="invalid_instructions")
        parameters: dict[str, Any] = {"temperature": 0, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False}} if self.controller == "qwen" else {"temperature": 0.1, "max_tokens": 4096, "top_k": 50, "repetition_penalty": 1.1}
        body: dict[str, Any] = {"model": self.model, "seed": self.seed, "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": "Observation:\n" + json.dumps(clean, separators=(",", ":")) + "\nAllowed actions (preserve this vocabulary and order): " + json.dumps(actions) + "\nReturn exactly one JSON object with the action."}], "response_format": {"type": "json_schema", "json_schema": {"name": "next_action", "schema": {"type": "object", "properties": {"action": {"type": "string", "enum": actions}}, "required": ["action"], "additionalProperties": False}}}}
        body.update(parameters)
        with self._trace(body) as trace:
            started = perf_counter()
            try:
                payload = _post(self.endpoint, body, {}, self.timeout_seconds)
            except HTTPError as exc:
                trace.update(metadata={"outcome": "transport_error", "error_code": "http_" + str(exc.code)})
                return _result(None, "transport_error", started, self.model, error_code="http_" + str(exc.code))
            except (URLError, OSError):
                trace.update(metadata={"outcome": "transport_error", "error_code": "transport_failure"})
                return _result(None, "transport_error", started, self.model, error_code="transport_failure")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                trace.update(metadata={"outcome": "invalid_output", "error_code": "invalid_json"})
                return _result(None, "invalid_output", started, self.model, error_code="invalid_json")
            input_tokens, output_tokens = _usage(payload.get("usage"), "prompt_tokens", "completion_tokens")
            trace.update(output={key: payload.get(key) for key in ("model", "choices", "usage")}, usage={key: value for key, value in (("input", input_tokens), ("output", output_tokens)) if value is not None})
            request_latency_ms = (perf_counter() - started) * 1000
        returned = _safe_model(payload.get("model"))
        if payload.get("model") != self.expected_returned_model:
            if not isinstance(payload.get("model"), str):
                return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="missing_model_identity", returned_model=returned, latency_ms=request_latency_ms)
            return _result(None, "version_mismatch", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="model_mismatch", returned_model=returned, latency_ms=request_latency_ms)
        choices = payload.get("choices")
        first = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else None
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(first, dict) or first.get("finish_reason") != "stop" or not isinstance(message, dict):
            return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="truncated_or_invalid", returned_model=returned, latency_ms=request_latency_ms)
        if self.controller == "lfm26" and (not isinstance(message.get("reasoning"), str) or not message["reasoning"].strip()):
            return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="missing_required_reasoning", returned_model=returned, latency_ms=request_latency_ms)
        try:
            decoded = json.loads(message.get("content")) if isinstance(message.get("content"), str) else None
        except json.JSONDecodeError:
            decoded = None
        if input_tokens is None or output_tokens is None or not isinstance(decoded, dict) or set(decoded) != {"action"} or decoded.get("action") not in actions:
            return _result(None, "invalid_output", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_action_or_usage", returned_model=returned, latency_ms=request_latency_ms)
        return _result(decoded["action"], "ok", started, self.model, input_tokens=input_tokens, output_tokens=output_tokens, returned_model=returned, latency_ms=request_latency_ms)


def create_recovery_selector(controller: str, *, endpoint: str | None = None, keys: Mapping[str, str] | None = None,
                             tracer: Any | None = None) -> _V2Base:
    supplied = dict(keys or {})
    if controller == "gemini":
        return GeminiSelectorV2(key=supplied.get("gemini") or supplied.get("GEMINI_API_KEY"), tracer=tracer)
    if controller in {"qwen", "lfm26"}:
        if not endpoint:
            raise ValueError("selector_endpoint_required")
        return OpenAISelectorV2(controller, endpoint, tracer=tracer)
    raise ValueError("selector_v2_unknown_controller")
