from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemini_provider
import providers
import recovery_runtime
import selector_runtime
from tracing import TraceRecorder


OBSERVATION = {"claim": "A material assertion.", "candidate": {"id": "p-1", "text": "A saved snippet.", "url": "https://example.test/p-1"}}
INSTRUCTIONS = "Include only direct evidence. Exclude topical context."
ACTIONS = ["include", "exclude"]


def _gemini_payload():
    return {"modelVersion": "gemini-3.1-flash-lite", "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "thoughtsTokenCount": 3},
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"thought": True, "text": "hidden"}, {"text": '{\"action\":\"include\"}'}]}}]}


def _openai_payload(model: str, *, reasoning: bool = False):
    message = {"content": '{\"action\":\"include\"}'}
    if reasoning:
        message["reasoning"] = "configured thinking"
    return {"model": model, "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            "choices": [{"finish_reason": "stop", "message": message}]}


def test_gemini_v2_request_is_byte_equivalent_to_v1(monkeypatch, tmp_path: Path) -> None:
    old, new = [], []
    monkeypatch.setattr(gemini_provider, "_http_post", lambda _url, body, _headers, _timeout: old.append(deepcopy(body)) or _gemini_payload())
    old_selector = selector_runtime.create_selector("gemini", keys={"gemini": "key"}).provider
    assert old_selector.choose(OBSERVATION, INSTRUCTIONS, ACTIONS).outcome == "ok"
    monkeypatch.setattr(recovery_runtime, "_post", lambda _url, body, _headers, _timeout: new.append(deepcopy(body)) or _gemini_payload())
    new_selector = recovery_runtime.GeminiSelectorV2(key="key", tracer=TraceRecorder(tmp_path / "gemini.jsonl", "r", "t"))
    assert new_selector.choose(OBSERVATION, INSTRUCTIONS, ACTIONS).outcome == "ok"
    assert old == new


def test_qwen_v2_request_is_byte_equivalent_to_v1(monkeypatch, tmp_path: Path) -> None:
    old, new = [], []
    monkeypatch.setattr(providers, "_http_post", lambda _url, body, _headers, _timeout: old.append(deepcopy(body)) or _openai_payload("Qwen/Qwen3.5-4B"))
    old_selector = selector_runtime.create_selector("qwen", endpoint="http://offline.test/v1").provider
    assert old_selector.choose(OBSERVATION, INSTRUCTIONS, ACTIONS).outcome == "ok"
    monkeypatch.setattr(recovery_runtime, "_post", lambda _url, body, _headers, _timeout: new.append(deepcopy(body)) or _openai_payload("Qwen/Qwen3.5-4B"))
    new_selector = recovery_runtime.OpenAISelectorV2("qwen", "http://offline.test/v1", tracer=TraceRecorder(tmp_path / "qwen.jsonl", "r", "t"))
    assert new_selector.choose(OBSERVATION, INSTRUCTIONS, ACTIONS).outcome == "ok"
    assert old == new


def test_lfm26_v2_request_changes_only_approved_max_tokens(monkeypatch, tmp_path: Path) -> None:
    old, new = [], []
    monkeypatch.setattr(providers, "_http_post", lambda _url, body, _headers, _timeout: old.append(deepcopy(body)) or _openai_payload("LiquidAI/LFM2.5-2.6B", reasoning=True))
    old_selector = selector_runtime.create_selector("lfm26", endpoint="http://offline.test/v1").provider
    assert old_selector.choose(OBSERVATION, INSTRUCTIONS, ACTIONS).outcome == "ok"
    monkeypatch.setattr(recovery_runtime, "_post", lambda _url, body, _headers, _timeout: new.append(deepcopy(body)) or _openai_payload("LiquidAI/LFM2.5-2.6B", reasoning=True))
    new_selector = recovery_runtime.OpenAISelectorV2("lfm26", "http://offline.test/v1", tracer=TraceRecorder(tmp_path / "lfm.jsonl", "r", "t"))
    assert new_selector.choose(OBSERVATION, INSTRUCTIONS, ACTIONS).outcome == "ok"
    assert old[0]["max_tokens"] == 2048 and new[0]["max_tokens"] == 4096
    old[0]["max_tokens"] = new[0]["max_tokens"]
    assert old == new
