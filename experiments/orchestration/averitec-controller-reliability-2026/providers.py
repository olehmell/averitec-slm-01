"""Small, dependency-free controller adapters used by the reliability study.

The adapters deliberately return a result for every attempted call.  They do
not retry: a transport or output failure is itself an experimental outcome.
They also never include a provider response body, endpoint, or exception text
in ``error_code`` because all of those can carry credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
import math
import os
from time import perf_counter
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from generation_profiles import generation_profile_identity, resolve_generation_profile


Outcome = Literal["ok", "invalid_output", "transport_error", "version_mismatch"]
TYPESAFE_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"


@dataclass(frozen=True)
class ControllerResult:
    action: str | None
    outcome: Outcome
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    model: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    error_code: str | None = None
    # ``model`` remains the requested, pinned identifier.  This separate
    # field records an unexpected provider identifier only when it is safe to
    # publish as an opaque model id.
    returned_model: str | None = None


class Controller(Protocol):
    def choose(
        self, observation: dict[str, Any], instructions: str, actions: list[str]
    ) -> ControllerResult: ...


class _NullTraceSpan:
    def update(self, **_kwargs: Any) -> "_NullTraceSpan":
        return self


@contextmanager
def _trace_request(tracer: Any, *, body: dict[str, Any], model: str):
    """Create an optional request generation without endpoint or auth headers."""
    if tracer is None:
        yield _NullTraceSpan()
        return
    with tracer.span("controller.request", kind="generation", input=body, model=model) as span:
        yield span


def _jev_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in ("model", "answers", "usage")}


def _openai_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # ``choices`` intentionally retains message.content, including invalid JSON,
    # for post-run diagnosis. It excludes arbitrary top-level provider fields.
    return {key: payload.get(key) for key in ("model", "choices", "usage")}


def _safe_usage(value: Any, input_name: str, output_name: str) -> tuple[int | None, int | None]:
    if not isinstance(value, dict):
        return None, None

    def number(name: str) -> int | None:
        item = value.get(name)
        return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None

    return number(input_name), number(output_name)


def _valid_probability_distribution(probabilities: dict[str, float]) -> bool:
    """Validate raw choice probabilities without rescaling rounded values.

    TypeSafe responses can be rounded.  We therefore accept an absolute
    total-error of at most 0.02, while retaining the original values exactly.
    """
    if not probabilities:
        return False
    values = list(probabilities.values())
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(float(item)) or float(item) < 0 for item in values):
        return False
    total = float(sum(values))
    return abs(total - 1.0) <= 0.02


def _safe_model_id(value: Any) -> str | None:
    """Allow opaque model ids, never URLs, headers, or arbitrary server text."""
    if (not isinstance(value, str) or not value or len(value) > 100
            or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/" for char in value)):
        return None
    return value


def _actions_or_error(actions: list[str]) -> str | None:
    if (not isinstance(actions, list) or not actions
            or any(not isinstance(action, str) or not action for action in actions)
            or len(set(actions)) != len(actions)):
        return "invalid_actions"
    return None


def _choice_criteria_or_error(
    choice_criteria: dict[str, str] | None, actions: list[str]
) -> dict[str, str | None] | str:
    """Validate static Jev Choice descriptions and order them by ``actions``."""
    if choice_criteria is None:
        return {action: None for action in actions}
    if (not isinstance(choice_criteria, dict)
            or set(choice_criteria) != set(actions)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in choice_criteria.items())):
        return "invalid_choice_criteria"
    # The controller receives intentionally permuted action orders in the
    # option-order condition.  Keep those orders in the provider body even
    # though the static criteria mapping may have a canonical insertion order.
    return {action: choice_criteria[action] for action in actions}


def _result(
    *, action: str | None, outcome: Outcome, started: float, model: str,
    input_tokens: int | None = None, output_tokens: int | None = None,
    probabilities: dict[str, float] | None = None, confidence: float | None = None,
    error_code: str | None = None, returned_model: str | None = None, latency_ms: float | None = None,
) -> ControllerResult:
    return ControllerResult(
        action=action,
        outcome=outcome,
        latency_ms=(perf_counter() - started) * 1000 if latency_ms is None else latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        probabilities=probabilities,
        confidence=confidence,
        error_code=error_code,
        returned_model=returned_model,
    )


def _http_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # nosec B310: URL is caller-controlled endpoint
        payload = response.read()
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("response_not_object")
    return decoded


class JevController:
    def __init__(self, model: str = "jev-1.13.0", key: str | None = None, timeout_seconds: float = 30,
                 tracer: Any | None = None, choice_criteria: dict[str, str] | None = None) -> None:
        self.model = model
        self.key = key if key is not None else os.environ.get("TYPESAFE_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.tracer = tracer
        # Copy once so a caller cannot mutate descriptions between states.
        self.choice_criteria = None if choice_criteria is None else dict(choice_criteria)

    def choose(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> ControllerResult:
        started = perf_counter()
        invalid = _actions_or_error(actions)
        if invalid:
            return _result(action=None, outcome="invalid_output", started=started, model=self.model, error_code=invalid)
        criteria = _choice_criteria_or_error(self.choice_criteria, actions)
        if isinstance(criteria, str):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model, error_code=criteria)
        if not isinstance(observation, dict) or not isinstance(instructions, str):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model, error_code="invalid_input")
        if not self.key:
            return _result(action=None, outcome="transport_error", started=started, model=self.model, error_code="missing_api_key")

        body = {
            "model": self.model,
            "state": observation,
            "questions": {"next_action": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }},
        }
        with _trace_request(self.tracer, body=body, model=self.model) as trace:
            # Start after durable trace-start I/O; end occurs after the result is
            # already measured, so instrumentation does not inflate latency.
            started = perf_counter()
            try:
                payload = _http_post(
                    TYPESAFE_SYSTEM_ONE_URL, body, {"Authorization": "Bearer " + self.key}, self.timeout_seconds
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
            trace_input_tokens, trace_output_tokens = _safe_usage(payload.get("usage"), "input_tokens", "output_tokens")
            trace.update(output=_jev_trace_payload(payload), usage={key: value for key, value in (("input", trace_input_tokens), ("output", trace_output_tokens)) if value is not None})
            request_latency_ms = (perf_counter() - started) * 1000

        input_tokens, output_tokens = _safe_usage(payload.get("usage"), "input_tokens", "output_tokens")
        returned_model = payload.get("model")
        if returned_model != self.model:
            return _result(action=None, outcome="version_mismatch", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="model_mismatch",
                           returned_model=_safe_model_id(returned_model), latency_ms=request_latency_ms)
        answer = payload.get("answers", {}).get("next_action") if isinstance(payload.get("answers"), dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_answer", latency_ms=request_latency_ms)
        choice, probabilities = answer.get("choice"), answer.get("probabilities")
        if not isinstance(choice, str) or choice not in actions or not isinstance(probabilities, dict) or set(probabilities) != set(actions):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_choice", latency_ms=request_latency_ms)
        copied = dict(probabilities)
        confidence = answer.get("confidence")
        if (not _valid_probability_distribution(copied)
                or not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0
                # Equal top probabilities are valid: the provider may choose
                # either tied option, so compare values rather than key order.
                or copied[choice] != max(copied.values())):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_probabilities", latency_ms=request_latency_ms)
        return _result(action=choice, outcome="ok", started=started, model=self.model,
                       input_tokens=input_tokens, output_tokens=output_tokens,
                       probabilities=copied, confidence=float(confidence),
                       returned_model=_safe_model_id(returned_model), latency_ms=request_latency_ms)


class OpenAIController:
    def __init__(self, endpoint: str, model: str, seed: int = 20260918, structured: bool = True,
                 timeout_seconds: float = 30, tracer: Any | None = None,
                 generation_profile: str = "baseline") -> None:
        self.endpoint = endpoint.rstrip("/") + "/chat/completions" if not endpoint.rstrip("/").endswith("/chat/completions") else endpoint
        self.model = model
        self.seed = seed
        self.structured = structured
        self.generation_profile = generation_profile
        self._profile = resolve_generation_profile(generation_profile, model)
        # A profile-level timeout is part of a named deployment policy.  It is
        # mutable only so the isolated warmup can temporarily bound one call.
        self.timeout_seconds = self._profile.timeout_seconds or timeout_seconds
        self.tracer = tracer

    @property
    def resolved_profile(self) -> dict[str, Any]:
        """A JSON-safe profile identity for run and deployment receipts."""
        return generation_profile_identity(self.generation_profile, self.model)

    def choose(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> ControllerResult:
        started = perf_counter()
        invalid = _actions_or_error(actions)
        if invalid:
            return _result(action=None, outcome="invalid_output", started=started, model=self.model, error_code=invalid)
        if not isinstance(observation, dict) or not isinstance(instructions, str):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model, error_code="invalid_input")
        schema = {
            "type": "object", "properties": {"action": {"type": "string", "enum": actions}},
            "required": ["action"], "additionalProperties": False,
        }
        body: dict[str, Any] = {
            "model": self.model, "seed": self.seed,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": "Observation:\n" + json.dumps(observation, separators=(",", ":"))
                 + "\nAllowed actions (preserve this vocabulary and order): " + json.dumps(actions)
                 + "\nReturn exactly one JSON object with the action."},
            ],
        }
        body.update(self._profile.request_parameters(self.model))
        if self.structured:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "next_action", "schema": schema}}
        with _trace_request(self.tracer, body=body, model=self.model) as trace:
            started = perf_counter()
            try:
                payload = _http_post(self.endpoint, body, {}, self.timeout_seconds)
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
            trace_input_tokens, trace_output_tokens = _safe_usage(payload.get("usage"), "prompt_tokens", "completion_tokens")
            trace.update(output=_openai_trace_payload(payload), usage={key: value for key, value in (("input", trace_input_tokens), ("output", trace_output_tokens)) if value is not None})
            request_latency_ms = (perf_counter() - started) * 1000

        input_tokens, output_tokens = _safe_usage(payload.get("usage"), "prompt_tokens", "completion_tokens")
        if payload.get("model") != self.model:
            return _result(action=None, outcome="version_mismatch", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="model_mismatch",
                           returned_model=_safe_model_id(payload.get("model")), latency_ms=request_latency_ms)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_choices", latency_ms=request_latency_ms)
        first = choices[0]
        if first.get("finish_reason") != "stop" or not isinstance(first.get("message"), dict):
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="truncated_or_invalid", latency_ms=request_latency_ms)
        if self._profile.reasoning_mode == 'always':
            reasoning = first['message'].get('reasoning')
            if not isinstance(reasoning, str) or not reasoning.strip():
                return _result(action=None, outcome='invalid_output', started=started, model=self.model,
                               input_tokens=input_tokens, output_tokens=output_tokens,
                               error_code='missing_required_reasoning', latency_ms=request_latency_ms)
        content = first["message"].get("content")
        try:
            decoded = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            decoded = None
        if not isinstance(decoded, dict) or set(decoded) != {"action"} or decoded.get("action") not in actions:
            return _result(action=None, outcome="invalid_output", started=started, model=self.model,
                           input_tokens=input_tokens, output_tokens=output_tokens, error_code="invalid_action_json", latency_ms=request_latency_ms)
        return _result(action=decoded["action"], outcome="ok", started=started, model=self.model,
                       input_tokens=input_tokens, output_tokens=output_tokens,
                       returned_model=_safe_model_id(payload.get("model")), latency_ms=request_latency_ms)
