from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import providers


OBSERVATION = {"schema": "controller/v1", "remaining": 3}
ACTIONS = ["search", "qa", "finish"]


class Response:
    def __init__(self, payload: object):
        self.payload = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TraceSpan:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class Tracer:
    def __init__(self):
        self.calls = []
        self.current = TraceSpan()

    def span_context(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self

    def span(self, *args, **kwargs):
        return self.span_context(*args, **kwargs)

    def __enter__(self):
        return self.current

    def __exit__(self, *_args):
        return False


def _capture(monkeypatch, payload):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return Response(payload)

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    return captured


def _jev_payload(*, model="jev-1.13.0", choice="search", usage=True):
    result = {"model": model, "answers": {"next_action": {
        "type": "choice", "choice": choice,
        "probabilities": {"search": 0.8, "qa": 0.1, "finish": 0.1},
        "confidence": 0.42,
    }}}
    if usage:
        result["usage"] = {"input_tokens": 22, "output_tokens": 7}
    return result


def _openai_payload(*, model="LiquidAI/LFM2.5-1.2B-Instruct", content='{"action":"qa"}', finish="stop", usage=True):
    result = {"model": model, "choices": [{"finish_reason": finish, "message": {"content": content}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 31, "completion_tokens": 4}
    return result


def test_jev_request_is_pinned_has_full_action_enum_and_returns_entropy_confidence(monkeypatch) -> None:
    captured = _capture(monkeypatch, _jev_payload())
    result = providers.JevController(key="TOP_SECRET", timeout_seconds=9).choose(OBSERVATION, "choose", ACTIONS)

    assert result.outcome == "ok" and result.action == "search"
    assert result.model == "jev-1.13.0" and result.input_tokens == 22 and result.output_tokens == 7
    assert result.probabilities == {"search": 0.8, "qa": 0.1, "finish": 0.1}
    assert result.confidence == 0.42  # Provider-calibrated, not an accuracy estimate we recalculate.
    assert captured["url"] == providers.TYPESAFE_SYSTEM_ONE_URL and captured["timeout"] == 9
    assert captured["body"] == {
        "model": "jev-1.13.0", "state": OBSERVATION,
        "questions": {"next_action": {"type": "choice", "instructions": "choose",
                                       "criteria": {"search": None, "qa": None, "finish": None}}},
    }
    assert captured["headers"]["Authorization"] == "Bearer TOP_SECRET"


def test_jev_static_choice_criteria_are_validated_and_follow_action_order(monkeypatch) -> None:
    captured = _capture(monkeypatch, _jev_payload(choice="search"))
    controller = providers.JevController(key="key", choice_criteria={
        "search": "Search the frozen corpus.",
        "qa": "Answer from retrieved passages.",
        "finish": "Finish only after every stage.",
    })
    reversed_actions = ["finish", "qa", "search"]
    result = controller.choose(OBSERVATION, "choose", reversed_actions)

    assert result.outcome == "ok"
    assert list(captured["body"]["questions"]["next_action"]["criteria"]) == reversed_actions
    assert captured["body"]["questions"]["next_action"]["criteria"] == {
        "finish": "Finish only after every stage.",
        "qa": "Answer from retrieved passages.",
        "search": "Search the frozen corpus.",
    }


@pytest.mark.parametrize("criteria", [
    {"search": "a", "qa": "b", "unexpected": "c"},
    {"search": "a", "qa": "b"},
    {"search": "a", "qa": "b", "finish": None},
])
def test_jev_rejects_invalid_choice_criteria_without_transport(monkeypatch, criteria) -> None:
    called = False

    def fail(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("transport must not run")

    monkeypatch.setattr(providers, "urlopen", fail)
    result = providers.JevController(key="key", choice_criteria=criteria).choose(OBSERVATION, "choose", ACTIONS)

    assert result.outcome == "invalid_output" and result.error_code == "invalid_choice_criteria"
    assert not called


def test_jev_version_mismatch_and_missing_usage_are_explicit(monkeypatch) -> None:
    _capture(monkeypatch, _jev_payload(model="jev-1.14.0", usage=False))
    result = providers.JevController(key="key").choose(OBSERVATION, "choose", ACTIONS)

    assert result.outcome == "version_mismatch" and result.action is None
    assert result.input_tokens is None and result.output_tokens is None
    assert result.error_code == "model_mismatch"
    assert result.returned_model == "jev-1.14.0"


def test_jev_rejects_invalid_probability_distribution(monkeypatch) -> None:
    bad = _jev_payload()
    bad["answers"]["next_action"]["probabilities"] = {"search": 0.9, "qa": 0.9, "finish": 0.2}
    bad["answers"]["next_action"]["confidence"] = 1.0
    _capture(monkeypatch, bad)

    result = providers.JevController(key="key").choose(OBSERVATION, "choose", ACTIONS)
    assert result.outcome == "invalid_output" and result.error_code == "invalid_probabilities"


def test_jev_keeps_rounded_raw_probabilities_and_allows_tied_winner(monkeypatch) -> None:
    payload = _jev_payload(choice="qa")
    payload["answers"]["next_action"]["probabilities"] = {"search": 0.49, "qa": 0.49, "finish": 0.01}
    payload["answers"]["next_action"]["confidence"] = 0.27
    _capture(monkeypatch, payload)

    result = providers.JevController(key="key").choose(OBSERVATION, "choose", ACTIONS)
    assert result.outcome == "ok" and result.action == "qa"
    assert result.probabilities == {"search": 0.49, "qa": 0.49, "finish": 0.01}
    assert result.confidence == 0.27


def test_controller_result_optional_provider_measurements_have_defaults() -> None:
    result = providers.ControllerResult("finish", "ok", 1.0, None, None, "oracle")
    assert result.probabilities is None and result.confidence is None and result.error_code is None


def test_transport_errors_never_expose_key_or_url_credentials(monkeypatch) -> None:
    secret = "TOP_SECRET"

    def fail(_request, timeout):
        raise HTTPError("https://user:URL_SECRET@provider.invalid", 401, "Bearer HEADER_SECRET", {}, BytesIO(b"BODY_SECRET"))

    monkeypatch.setattr(providers, "urlopen", fail)
    result = providers.JevController(key=secret).choose(OBSERVATION, "choose", ACTIONS)

    assert result.outcome == "transport_error" and result.error_code == "http_401"
    rendered = repr(result)
    for value in (secret, "URL_SECRET", "HEADER_SECRET", "BODY_SECRET"):
        assert value not in rendered


@pytest.mark.parametrize("payload", [b"not-json", ["not", "an", "object"]])
def test_invalid_provider_json_is_invalid_output(monkeypatch, payload) -> None:
    _capture(monkeypatch, payload)
    result = providers.OpenAIController("http://localhost:8000/v1", "LiquidAI/LFM2.5-1.2B-Instruct").choose(OBSERVATION, "choose", ACTIONS)
    assert result.outcome == "invalid_output" and result.action is None


def test_openai_structured_request_and_exact_output_validation(monkeypatch) -> None:
    captured = _capture(monkeypatch, _openai_payload())
    result = providers.OpenAIController("http://localhost:8000/v1", "LiquidAI/LFM2.5-1.2B-Instruct", timeout_seconds=4).choose(OBSERVATION, "choose", ACTIONS)

    assert result.outcome == "ok" and result.action == "qa"
    assert result.probabilities is None and result.confidence is None
    assert result.input_tokens == 31 and result.output_tokens == 4
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["timeout"] == 4
    assert captured["body"]["temperature"] == 0 and captured["body"]["max_tokens"] == 160
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert captured["body"]["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"] == ACTIONS
    assert json.dumps(ACTIONS) in captured["body"]["messages"][1]["content"]
    assert "chat_template_kwargs" not in captured["body"]


@pytest.mark.parametrize("content,finish", [
    ('```json\\n{"action":"qa"}\\n```', "stop"),
    ('{"action":"qa","why":"extra"}', "stop"),
    ('{"action":"unknown"}', "stop"),
    ('{"action":"qa"}', "length"),
])
def test_openai_rejects_repaired_extra_unknown_or_truncated_output(monkeypatch, content, finish) -> None:
    _capture(monkeypatch, _openai_payload(content=content, finish=finish))
    result = providers.OpenAIController("http://localhost:8000/v1/chat/completions", "LiquidAI/LFM2.5-1.2B-Instruct").choose(OBSERVATION, "choose", ACTIONS)
    assert result.outcome == "invalid_output" and result.action is None


def test_openai_qwen_disables_thinking_but_lfm_does_not(monkeypatch) -> None:
    qwen = _capture(monkeypatch, _openai_payload(model="Qwen/Qwen3.5-4B", usage=False))
    result = providers.OpenAIController("http://localhost:8000/v1", "Qwen/Qwen3.5-4B", structured=False).choose(OBSERVATION, "choose", ACTIONS)
    assert result.outcome == "ok" and result.input_tokens is None and result.output_tokens is None
    assert qwen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in qwen["body"]
    assert json.dumps(ACTIONS) in qwen["body"]["messages"][1]["content"]


def test_openai_mismatch_keeps_requested_and_safe_returned_model(monkeypatch) -> None:
    _capture(monkeypatch, _openai_payload(model="served-qwen"))
    result = providers.OpenAIController("http://localhost:8000/v1", "Qwen/Qwen3.5-4B").choose(OBSERVATION, "choose", ACTIONS)
    assert result.outcome == "version_mismatch" and result.model == "Qwen/Qwen3.5-4B"
    assert result.returned_model == "served-qwen"


def test_provider_trace_captures_allowlisted_request_and_raw_invalid_content(monkeypatch) -> None:
    captured = _capture(monkeypatch, _openai_payload(content="not json"))
    tracer = Tracer()
    result = providers.OpenAIController("http://endpoint.invalid/v1", "LiquidAI/LFM2.5-1.2B-Instruct", tracer=tracer).choose(OBSERVATION, "choose", ACTIONS)

    assert result.outcome == "invalid_output"
    assert tracer.calls[0][0] == ("controller.request",)
    trace_input = tracer.calls[0][1]["input"]
    assert trace_input == captured["body"] and "Authorization" not in trace_input
    assert "endpoint.invalid" not in json.dumps(trace_input)
    assert tracer.current.updates == [{"output": {
        "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "choices": [{"finish_reason": "stop", "message": {"content": "not json"}}],
        "usage": {"prompt_tokens": 31, "completion_tokens": 4},
    }, "usage": {"input": 31, "output": 4}}]


def test_provider_trace_retains_reasoning_field_without_parsing_or_repair(monkeypatch) -> None:
    payload = _openai_payload(model="Qwen/Qwen3.5-4B", content='{"action":"qa"}')
    payload["choices"][0]["message"]["reasoning_content"] = "raw hidden-chain field"
    _capture(monkeypatch, payload)
    tracer = Tracer()

    result = providers.OpenAIController("http://localhost:8000/v1", "Qwen/Qwen3.5-4B", tracer=tracer).choose(
        OBSERVATION, "choose", ACTIONS
    )

    assert result.outcome == "ok"
    assert tracer.current.updates[0]["output"]["choices"][0]["message"]["reasoning_content"] == "raw hidden-chain field"
