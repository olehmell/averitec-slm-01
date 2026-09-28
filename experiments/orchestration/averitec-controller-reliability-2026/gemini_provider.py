"""Gemini GenerateContent controller for the reliability study.

This adapter intentionally uses the REST contract directly rather than an SDK
so its request, trace, timeout, and failure behaviour remain equivalent to the
other dependency-free study adapters.  It is single-turn only: thought
signatures are neither accepted nor replayed between controller decisions.
"""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError

from providers import (
    ControllerResult,
    _actions_or_error,
    _http_post,
    _safe_model_id,
    _trace_request,
)


GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_MODEL = "gemini-3.1-flash-lite"
# GenerateContent returns this stable alias in ``modelVersion``.  The Models
# metadata version is retained separately in the committed study config; it is
# not an asserted generation deployment revision.
DEFAULT_EXPECTED_VERSION = "gemini-3.1-flash-lite"

# Gemini 3.1 Flash-Lite supports both levels.  ``minimal`` is deliberately
# named rather than represented as thinking-off: Gemini 3 Flash-Lite cannot
# fully disable thinking.  The high profile exists only when explicitly named.
_GEMINI_PROFILES: dict[str, dict[str, Any]] = {
    "gemini_native": {
        "temperature": 1.0,
        "maxOutputTokens": 2048,
        "thinkingConfig": {"thinkingLevel": "minimal"},
    },
    "gemini_thinking_high": {
        "temperature": 1.0,
        "maxOutputTokens": 2048,
        "thinkingConfig": {"thinkingLevel": "high"},
        "timeout_seconds": 60,
    },
}


def gemini_profile_identity(name: str, model: str, expected_version: str) -> dict[str, Any]:
    """Return a receipt-safe, complete Gemini request-policy identity.

    The adapter is pinned to the study's stable Flash-Lite alias and expected
    deployed version.  A caller must choose the high-thinking profile by name;
    no profile is upgraded automatically.
    """
    if name not in _GEMINI_PROFILES:
        raise ValueError("unknown_generation_profile")
    if model != DEFAULT_MODEL:
        raise ValueError("generation_profile_model_incompatible")
    if _safe_model_id(expected_version) != expected_version:
        raise ValueError("invalid_expected_version")
    profile = _GEMINI_PROFILES[name]
    request_parameters = {
        key: value for key, value in profile.items() if key != "timeout_seconds"
    }
    # ``json`` round-tripping prevents a caller from mutating module policy via
    # a receipt object and guarantees only JSON-native values leave this module.
    return {
        "name": name,
        "provider": "gemini_generate_content",
        "api_version": "v1beta",
        "model": model,
        "expected_version": expected_version,
        "generation_config": json.loads(json.dumps(request_parameters)),
        "timeout_seconds": profile.get("timeout_seconds"),
    }


def _result(
    *, action: str | None, outcome: str, started: float, model: str,
    input_tokens: int | None = None, output_tokens: int | None = None,
    error_code: str | None = None, returned_model: str | None = None,
    latency_ms: float | None = None,
) -> ControllerResult:
    return ControllerResult(
        action=action,
        outcome=outcome,  # type: ignore[arg-type]  # Closed set is enforced at call sites.
        latency_ms=(perf_counter() - started) * 1000 if latency_ms is None else latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        error_code=error_code,
        returned_model=returned_model,
    )


def _gemini_usage(value: Any) -> tuple[int | None, int | None]:
    """Map Gemini usage without counting thoughts twice through total tokens."""
    if not isinstance(value, dict):
        return None, None

    def number(name: str) -> int | None:
        item = value.get(name)
        return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None

    prompt = number("promptTokenCount")
    # Candidate tokens are the required final-output measurement.  A thought
    # count may be omitted (zero), but a malformed supplied value makes the
    # aggregate unknowable rather than an invitation to underreport it.
    candidate = number("candidatesTokenCount")
    if candidate is None:
        return prompt, None
    if "thoughtsTokenCount" not in value:
        return prompt, candidate
    thoughts = number("thoughtsTokenCount")
    output = candidate + thoughts if thoughts is not None else None
    return prompt, output


def _gemini_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep response material needed for audit, excluding opaque response ids."""
    return {
        key: payload.get(key)
        for key in ("candidates", "promptFeedback", "usageMetadata", "modelVersion")
    }


def _final_text(candidate: dict[str, Any]) -> str | None:
    """Concatenate non-thought text parts only; thought summaries are not output."""
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return None
    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            return None
        if part.get("thought") is True:
            continue
        text = part.get("text")
        if not isinstance(text, str):
            return None
        text_parts.append(text)
    return "".join(text_parts) if text_parts else None


class GeminiController:
    """One-request Gemini controller with strict model and JSON validation."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        expected_version: str = DEFAULT_EXPECTED_VERSION,
        key: str | None = None,
        seed: int = 20260918,
        timeout_seconds: float = 30,
        tracer: Any | None = None,
        generation_profile: str = "gemini_native",
    ) -> None:
        self.model = model
        self.expected_returned_model = expected_version
        self.key = key if key is not None else os.environ.get("GEMINI_API_KEY")
        self.seed = seed
        self.generation_profile = generation_profile
        self._profile = gemini_profile_identity(generation_profile, model, expected_version)
        self.timeout_seconds = self._profile["timeout_seconds"] or timeout_seconds
        self.tracer = tracer

    @property
    def resolved_profile(self) -> dict[str, Any]:
        """A JSON-safe profile identity for run and deployment receipts."""
        return gemini_profile_identity(
            self.generation_profile, self.model, self.expected_returned_model
        )

    def choose(
        self, observation: dict[str, Any], instructions: str, actions: list[str]
    ) -> ControllerResult:
        started = perf_counter()
        invalid = _actions_or_error(actions)
        if invalid:
            return _result(action=None, outcome="invalid_output", started=started, model=self.model, error_code=invalid)
        if not isinstance(observation, dict) or not isinstance(instructions, str):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model, error_code="invalid_input")
        if not self.key:
            return _result(action=None, outcome="transport_error", started=started, model=self.model, error_code="missing_api_key")

        schema = {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": actions}},
            "required": ["action"],
            "additionalProperties": False,
        }
        generation_config = dict(self._profile["generation_config"])
        generation_config["seed"] = self.seed
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseJsonSchema"] = schema
        body = {
            "systemInstruction": {"parts": [{"text": instructions}]},
            "contents": [{"role": "user", "parts": [{"text": (
                "Observation:\n" + json.dumps(observation, separators=(",", ":"))
                + "\nAllowed actions (preserve this vocabulary and order): " + json.dumps(actions)
                + "\nReturn exactly one JSON object with the action."
            )}]}],
            "generationConfig": generation_config,
        }
        endpoint = GEMINI_GENERATE_CONTENT_URL.format(model=self.model)
        with _trace_request(self.tracer, body=body, model=self.model) as trace:
            started = perf_counter()
            try:
                payload = _http_post(
                    endpoint, body, {"x-goog-api-key": self.key}, self.timeout_seconds
                )
            except HTTPError as exc:
                code = f"http_{exc.code}"
                trace.update(metadata={"outcome": "transport_error", "error_code": code})
                return _result(action=None, outcome="transport_error", started=started, model=self.model, error_code=code)
            except (URLError, OSError):
                trace.update(metadata={"outcome": "transport_error", "error_code": "transport_failure"})
                return _result(action=None, outcome="transport_error", started=started, model=self.model, error_code="transport_failure")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                trace.update(metadata={"outcome": "invalid_output", "error_code": "invalid_json"})
                return _result(action=None, outcome="invalid_output", started=started, model=self.model, error_code="invalid_json")
            input_tokens, output_tokens = _gemini_usage(payload.get("usageMetadata"))
            trace.update(
                output=_gemini_trace_payload(payload),
                usage={key: value for key, value in (("input", input_tokens), ("output", output_tokens)) if value is not None},
            )
            request_latency_ms = (perf_counter() - started) * 1000

        returned_model = _safe_model_id(payload.get("modelVersion"))
        if payload.get("modelVersion") != self.expected_returned_model:
            return _result(
                action=None, outcome="version_mismatch", started=started, model=self.model,
                input_tokens=input_tokens, output_tokens=output_tokens, error_code="model_mismatch",
                returned_model=returned_model, latency_ms=request_latency_ms,
            )
        prompt_feedback = payload.get("promptFeedback")
        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="prompt_blocked",
                           returned_model=returned_model, latency_ms=request_latency_ms)
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="empty_candidates",
                           returned_model=returned_model, latency_ms=request_latency_ms)
        if len(candidates) != 1 or not isinstance(candidates[0], dict):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_candidates",
                           returned_model=returned_model, latency_ms=request_latency_ms)
        candidate = candidates[0]
        safety_ratings = candidate.get("safetyRatings")
        if isinstance(safety_ratings, list) and any(
            isinstance(rating, dict) and rating.get("blocked") is True for rating in safety_ratings
        ):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="safety_blocked",
                           returned_model=returned_model, latency_ms=request_latency_ms)
        finish = candidate.get("finishReason")
        if finish != "STOP":
            error_code = "truncated_or_invalid" if finish == "MAX_TOKENS" else "safety_or_invalid_finish"
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code=error_code,
                           returned_model=returned_model, latency_ms=request_latency_ms)
        content = _final_text(candidate)
        try:
            decoded = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            decoded = None
        if not isinstance(decoded, dict) or set(decoded) != {"action"} or decoded.get("action") not in actions:
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_action_json",
                           returned_model=returned_model, latency_ms=request_latency_ms)
        return _result(action=decoded["action"], outcome="ok", started=started, model=self.model,
                       input_tokens=input_tokens, output_tokens=output_tokens,
                       returned_model=returned_model, latency_ms=request_latency_ms)
