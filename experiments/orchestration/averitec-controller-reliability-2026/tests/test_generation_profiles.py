from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import providers
from generation_profiles import generation_profile_identity, resolve_generation_profile


OBSERVATION = {"schema": "controller/v1", "remaining": 3}
ACTIONS = ["search", "qa", "finish"]


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _payload(model):
    return {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": {
            "content": '{"action":"qa"}',
            "reasoning": 'synthetic reasoning fixture' if model == 'LiquidAI/LFM2.5-2.6B' else None,
        }}],
    }


@pytest.mark.parametrize(("profile_name", "model", "expected"), [
    ("baseline", "LiquidAI/LFM2.5-1.2B-Instruct", {"temperature": 0, "max_tokens": 160}),
    ("lfm_native", "LiquidAI/LFM2.5-1.2B-Instruct", {
        "temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05, "max_tokens": 160,
    }),
    ("lfm26_native", "LiquidAI/LFM2.5-2.6B", {
        "temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 2048,
    }),
    ("qwen_native", "Qwen/Qwen3.5-4B", {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
        "repetition_penalty": 1, "max_tokens": 160,
        "chat_template_kwargs": {"enable_thinking": False},
    }),
    ("qwen_thinking", "Qwen/Qwen3.5-4B", {
        "temperature": 1, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
        "repetition_penalty": 1, "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
    }),
])
def test_named_profile_propagates_exact_request_parameters(monkeypatch, profile_name, model, expected):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return Response(_payload(model))

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    controller = providers.OpenAIController("http://localhost:8000/v1", model,
                                            timeout_seconds=11, generation_profile=profile_name)
    result = controller.choose(OBSERVATION, "choose", ACTIONS)

    assert result.outcome == "ok"
    assert {key: captured["body"][key] for key in expected} == expected
    assert captured["timeout"] == (60 if profile_name in {"qwen_thinking", "lfm26_native"} else 11)
    if profile_name == "lfm26_native":
        assert "chat_template_kwargs" not in captured["body"]
        assert controller.resolved_profile["reasoning_mode"] == "always"
    assert controller.resolved_profile == generation_profile_identity(profile_name, model)
    assert json.loads(json.dumps(controller.resolved_profile))["name"] == profile_name


def test_baseline_preserves_existing_qwen_nonthinking_behavior() -> None:
    profile = resolve_generation_profile("baseline", "Qwen/Qwen3.5-4B")
    assert profile.request_parameters("Qwen/Qwen3.5-4B") == {
        "temperature": 0,
        "max_tokens": 160,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_baseline_receipt_identity_is_unchanged_for_legacy_lfm() -> None:
    model = "LiquidAI/LFM2.5-1.2B-Instruct"
    assert generation_profile_identity("baseline", model) == {
        "name": "baseline",
        "model_family": "any",
        "model": model,
        "request_parameters": {"temperature": 0, "max_tokens": 160},
        "timeout_seconds": None,
    }


@pytest.mark.parametrize(("profile_name", "model"), [
    ("lfm_native", "Qwen/Qwen3.5-4B"),
    ("lfm_native", "LiquidAI/LFM2.5-2.6B"),
    ("baseline", "LiquidAI/LFM2.5-2.6B"),
    ("lfm26_native", "LiquidAI/LFM2.5-1.2B-Instruct"),
    ("lfm26_native", "LiquidAI/LFM2.5-2.6B-Instruct"),
    ("qwen_native", "LiquidAI/LFM2.5-1.2B-Instruct"),
    ("qwen_thinking", "LiquidAI/LFM2.5-1.2B-Instruct"),
])
def test_profile_model_mismatch_fails_closed_before_transport(monkeypatch, profile_name, model) -> None:
    called = False

    def fake_urlopen(*_args):
        nonlocal called
        called = True
        raise AssertionError("transport must not run")

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="generation_profile_model_incompatible"):
        providers.OpenAIController("http://localhost:8000/v1", model, generation_profile=profile_name)
    assert not called


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown_generation_profile"):
        providers.OpenAIController("http://localhost:8000/v1", "Qwen/Qwen3.5-4B", generation_profile="other")
