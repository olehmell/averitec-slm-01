from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import ACTIONS
import gemini_provider


OBSERVATION = {
    "schema": "controller/v1",
    "remaining": 9,
    "completed_count": 2,
    "failed_count": 0,
}


class TraceSpan:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(self, **kwargs: object) -> None:
        self.updates.append(kwargs)


class Tracer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.current = TraceSpan()

    def span(self, *args: object, **kwargs: object) -> "Tracer":
        self.calls.append((args, kwargs))
        return self

    def __enter__(self) -> TraceSpan:
        return self.current

    def __exit__(self, *_args: object) -> bool:
        return False


def _payload(
    *,
    version: str = gemini_provider.DEFAULT_EXPECTED_VERSION,
    text: str = '{"action":"qa"}',
    finish: str = "STOP",
    usage: bool = True,
) -> dict[str, object]:
    result: dict[str, object] = {
        "modelVersion": version,
        "candidates": [{
            "content": {"parts": [{"text": text}]},
            "finishReason": finish,
            "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"}],
        }],
    }
    if usage:
        result["usageMetadata"] = {
            "promptTokenCount": 31,
            "candidatesTokenCount": 4,
            "thoughtsTokenCount": 9,
            "totalTokenCount": 44,
        }
    return result


def _capture(monkeypatch: pytest.MonkeyPatch, payload: object) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_post(url: str, body: dict[str, object], headers: dict[str, str], timeout: float) -> dict[str, object]:
        captured.update(url=url, body=body, headers=headers, timeout=timeout)
        if isinstance(payload, BaseException):
            raise payload
        assert isinstance(payload, dict)
        return payload

    monkeypatch.setattr(gemini_provider, "_http_post", fake_post)
    return captured


def test_native_request_has_all_nine_actions_json_schema_and_secret_only_in_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch, _payload())
    result = gemini_provider.GeminiController(key="TOP_SECRET", timeout_seconds=12).choose(
        OBSERVATION, "Follow the stage rules.", list(ACTIONS)
    )

    assert result.outcome == "ok" and result.action == "qa"
    assert result.model == gemini_provider.DEFAULT_MODEL
    assert result.returned_model == gemini_provider.DEFAULT_EXPECTED_VERSION
    # Candidate and thought tokens are disjoint fields; totalTokenCount would
    # count the prompt again and must not be used as controller output tokens.
    assert (result.input_tokens, result.output_tokens) == (31, 13)
    assert captured["url"] == gemini_provider.GEMINI_GENERATE_CONTENT_URL.format(
        model=gemini_provider.DEFAULT_MODEL
    )
    assert captured["headers"] == {"x-goog-api-key": "TOP_SECRET"}
    assert captured["timeout"] == 12
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["systemInstruction"] == {"parts": [{"text": "Follow the stage rules."}]}
    config = body["generationConfig"]
    assert isinstance(config, dict)
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": list(ACTIONS)}},
        "required": ["action"],
        "additionalProperties": False,
    }
    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert config["temperature"] == 1.0 and config["maxOutputTokens"] == 2048
    assert config["seed"] == 20260918
    assert json.dumps(body).find("TOP_SECRET") == -1
    contents = body["contents"]
    assert isinstance(contents, list)
    assert json.dumps(list(ACTIONS)) in contents[0]["parts"][0]["text"]


def test_profile_identity_is_json_safe_pinned_and_high_is_explicit() -> None:
    native = gemini_provider.gemini_profile_identity(
        "gemini_native", gemini_provider.DEFAULT_MODEL, gemini_provider.DEFAULT_EXPECTED_VERSION
    )
    high = gemini_provider.GeminiController(key="key", generation_profile="gemini_thinking_high")

    assert json.loads(json.dumps(native)) == native
    assert native["generation_config"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert high.resolved_profile["generation_config"]["thinkingConfig"] == {"thinkingLevel": "high"}
    assert high.timeout_seconds == 60
    with pytest.raises(ValueError, match="unknown_generation_profile"):
        gemini_provider.GeminiController(key="key", generation_profile="baseline")
    with pytest.raises(ValueError, match="generation_profile_model_incompatible"):
        gemini_provider.GeminiController(model="gemini-elsewhere", key="key")


def test_thought_parts_are_retained_in_trace_but_excluded_from_final_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    candidate = payload["candidates"][0]
    candidate["content"]["parts"] = [
        {"thought": True, "text": "private thought summary"},
        {"text": '{"action":"retrieve"}'},
    ]
    _capture(monkeypatch, payload)
    tracer = Tracer()

    result = gemini_provider.GeminiController(key="key", tracer=tracer).choose(
        OBSERVATION, "choose", list(ACTIONS)
    )

    assert result.outcome == "ok" and result.action == "retrieve"
    trace_output = tracer.current.updates[0]["output"]
    assert trace_output["candidates"][0]["content"]["parts"][0]["text"] == "private thought summary"
    assert tracer.current.updates[0]["usage"] == {"input": 31, "output": 13}


@pytest.mark.parametrize(
    ("text", "finish", "expected"),
    [
        ('```json\\n{"action":"qa"}\\n```', "STOP", "invalid_action_json"),
        ('{"action":"qa","why":"extra"}', "STOP", "invalid_action_json"),
        ('{"action":"unknown"}', "STOP", "invalid_action_json"),
        ('{"action":"qa"}', "MAX_TOKENS", "truncated_or_invalid"),
        ('{"action":"qa"}', "SAFETY", "safety_or_invalid_finish"),
    ],
)
def test_strict_json_and_finish_fail_closed(
    monkeypatch: pytest.MonkeyPatch, text: str, finish: str, expected: str
) -> None:
    _capture(monkeypatch, _payload(text=text, finish=finish))
    result = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", list(ACTIONS))

    assert result.outcome == "invalid_output" and result.action is None
    assert result.error_code == expected


def test_empty_and_prompt_safety_failures_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = {
        "modelVersion": gemini_provider.DEFAULT_EXPECTED_VERSION,
        "promptFeedback": {"blockReason": "SAFETY"},
        "usageMetadata": {"promptTokenCount": 4},
    }
    _capture(monkeypatch, blocked)
    result = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", list(ACTIONS))
    assert result.outcome == "invalid_output" and result.error_code == "prompt_blocked"
    assert result.input_tokens == 4 and result.output_tokens is None

    _capture(monkeypatch, {"modelVersion": gemini_provider.DEFAULT_EXPECTED_VERSION})
    result = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", list(ACTIONS))
    assert result.outcome == "invalid_output" and result.error_code == "empty_candidates"


def test_prompt_and_candidate_safety_blocks_win_over_nominal_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_blocked = _payload()
    prompt_blocked["promptFeedback"] = {"blockReason": "SAFETY"}
    _capture(monkeypatch, prompt_blocked)
    result = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", list(ACTIONS))
    assert result.outcome == "invalid_output" and result.error_code == "prompt_blocked"

    candidate_blocked = _payload()
    candidate_blocked["candidates"][0]["safetyRatings"] = [{"blocked": True}]
    _capture(monkeypatch, candidate_blocked)
    result = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", list(ACTIONS))
    assert result.outcome == "invalid_output" and result.error_code == "safety_blocked"


def test_multiple_candidates_are_rejected_without_selecting_the_first(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    payload["candidates"].append({
        "content": {"parts": [{"text": '{"action":"retrieve"}'}]},
        "finishReason": "STOP",
    })
    _capture(monkeypatch, payload)

    result = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", list(ACTIONS))
    assert result.outcome == "invalid_output" and result.error_code == "invalid_candidates"


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"promptTokenCount": 3, "candidatesTokenCount": 4}, (3, 4)),
        ({"promptTokenCount": 3, "thoughtsTokenCount": 4}, (3, None)),
        ({"promptTokenCount": 3, "candidatesTokenCount": 4, "thoughtsTokenCount": "bad"}, (3, None)),
        ({"promptTokenCount": 3, "candidatesTokenCount": 4, "thoughtsTokenCount": 0}, (3, 4)),
    ],
)
def test_usage_requires_candidate_tokens_and_never_silently_drops_malformed_thoughts(
    usage: dict[str, object], expected: tuple[int | None, int | None]
) -> None:
    assert gemini_provider._gemini_usage(usage) == expected


def test_model_version_is_exact_and_returned_identifier_is_safely_constrained(monkeypatch: pytest.MonkeyPatch) -> None:
    # Model metadata's availability version is not the GenerateContent
    # deployment identity observed in the qualification receipt.
    _capture(monkeypatch, _payload(version="3.1-flash-lite-05-2026"))
    result = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", list(ACTIONS))

    assert result.outcome == "version_mismatch" and result.error_code == "model_mismatch"
    assert result.model == gemini_provider.DEFAULT_MODEL
    assert result.returned_model == "3.1-flash-lite-05-2026"

    _capture(monkeypatch, _payload(version="https://secret.invalid/returned-model"))
    result = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", list(ACTIONS))
    assert result.outcome == "version_mismatch" and result.returned_model is None
    assert "secret.invalid" not in repr(result)


def test_missing_key_or_invalid_input_does_not_call_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("transport must not run")

    monkeypatch.setattr(gemini_provider, "_http_post", forbidden)
    no_key = gemini_provider.GeminiController(key="").choose(OBSERVATION, "choose", list(ACTIONS))
    bad_actions = gemini_provider.GeminiController(key="key").choose(OBSERVATION, "choose", ["qa", "qa"])

    assert no_key.outcome == "transport_error" and no_key.error_code == "missing_api_key"
    assert bad_actions.outcome == "invalid_output" and bad_actions.error_code == "invalid_actions"
    assert calls == 0


def test_transport_error_never_exposes_provider_or_key_details(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "TOP_SECRET"
    error = HTTPError(
        "https://user:URL_SECRET@provider.invalid", 401, "HEADER_SECRET", {}, BytesIO(b"BODY_SECRET")
    )
    _capture(monkeypatch, error)
    result = gemini_provider.GeminiController(key=secret).choose(OBSERVATION, "choose", list(ACTIONS))

    assert result.outcome == "transport_error" and result.error_code == "http_401"
    rendered = repr(result)
    for value in (secret, "URL_SECRET", "HEADER_SECRET", "BODY_SECRET"):
        assert value not in rendered
