from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selector_runtime as runtime


OBSERVATION = {"claim": "A material assertion.", "candidate": {"id": "p-1", "text": "A directly relevant saved snippet.", "url": "https://example.test/p-1"}}
INSTRUCTIONS = "Include only snippets that directly support or refute a material claim part. Exclude topical background."
ACTIONS = ["include", "exclude"]


class FakeProvider:
    def __init__(self) -> None:
        self.received = None

    def choose(self, observation, instructions, actions):
        self.received = observation, instructions, actions
        return runtime.ControllerResult("include", "ok", 1.0, 5, 1, "fake")


class FakeTokenizer:
    mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", 3, 1, 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split()) or 1))}


class FakeModel:
    def __init__(self) -> None:
        self.cpu_attempts = 0

    def to(self, device):
        if str(device).startswith("cpu"):
            self.cpu_attempts += 1
        return self


class FakeAgent:
    def __init__(self, payload=None) -> None:
        self.cfg = {"max_len": 1024, "head_max_len": 256}
        self.device, self.tok, self.model = "cuda:0", FakeTokenizer(), FakeModel()
        self.calls = 0
        self.payload = payload or {"answers": {"evidence_selection": {"choice": "include", "probabilities": {"include": 0.8, "exclude": 0.2}, "confidence": 0.7}}, "usage": {"input_tokens": 17, "output_tokens": 0}}

    def predict(self, state, questions):
        self.calls += 1
        assert questions["evidence_selection"]["instructions"] == INSTRUCTIONS
        assert list(questions["evidence_selection"]["criteria"]) == ACTIONS
        return self.payload


class FakeTorch:
    class cuda:
        @staticmethod
        def synchronize(device):
            assert device == "cuda:0"


class CpuFallbackAgent(FakeAgent):
    def predict(self, state, questions):
        self.calls += 1
        self.model.to("cpu")
        raise AssertionError("the blocked CPU fallback must interrupt predict")


def test_schema_is_lossless_and_rejects_leakage_and_missing_url() -> None:
    assert runtime.validate_selector_observation(OBSERVATION) == OBSERVATION
    for changed in (
        {**OBSERVATION, "gold": "supported"},
        {"claim": OBSERVATION["claim"], "candidate": {**OBSERVATION["candidate"], "rating": 5}},
        {"claim": OBSERVATION["claim"], "candidate": {"id": "p", "text": "text"}},
    ):
        with pytest.raises(ValueError):
            runtime.validate_selector_observation(changed)


def test_provider_selector_preserves_explicit_instructions_and_binary_order() -> None:
    fake = FakeProvider()
    result = runtime.ProviderSelector(fake).choose(OBSERVATION, INSTRUCTIONS, ACTIONS)
    assert result.action == "include"
    assert fake.received == (OBSERVATION, INSTRUCTIONS, ACTIONS)
    with pytest.raises(ValueError, match="selector_invalid_actions"):
        runtime.ProviderSelector(fake).choose(OBSERVATION, INSTRUCTIONS, ["include", "maybe"])
    with pytest.raises(ValueError, match="selector_invalid_observation_keys"):
        runtime.ProviderSelector(fake).choose({**OBSERVATION, "untrusted": "leak"}, INSTRUCTIONS, ACTIONS)
    assert fake.received == (OBSERVATION, INSTRUCTIONS, ACTIONS)


def test_factory_uses_approved_profiles_without_network_calls() -> None:
    assert runtime.create_selector("jev", keys={"jev": "injected"}).provider.model == "jev-1.13.0"
    assert runtime.create_selector("qwen", endpoint="http://offline.test/v1").provider.generation_profile == "baseline"
    assert runtime.create_selector("lfm", endpoint="http://offline.test/v1").provider.generation_profile == "lfm_native"
    assert runtime.create_selector("lfm26", endpoint="http://offline.test/v1").provider.generation_profile == "lfm26_native"
    assert runtime.create_selector("gemini", keys={"gemini": "injected"}).provider.generation_profile == "gemini_native"
    for name, kwargs in (("qwen", {"endpoint": "http://offline.test/v1"}), ("lfm", {"endpoint": "http://offline.test/v1"}), ("lfm26", {"endpoint": "http://offline.test/v1"}), ("gemini", {"keys": {"gemini": "injected"}})):
        assert runtime.create_selector(name, **kwargs).provider.seed == 20260919
    with pytest.raises(ValueError, match="selector_endpoint_required"):
        runtime.create_selector("qwen")


def test_native_laya_runs_one_binary_forward_and_records_typed_identity() -> None:
    agent = FakeAgent()
    selector = runtime.create_selector("laya_typed", laya_agent=agent, laya_torch=FakeTorch())
    result = selector.choose(OBSERVATION, INSTRUCTIONS, ACTIONS)
    assert result.outcome == "ok" and result.action == "include"
    assert result.probabilities == {"include": 0.8, "exclude": 0.2}
    assert result.input_tokens == 17 and result.output_tokens == 0 and agent.calls == 1
    assert selector.identity()["model"] == runtime.LAYA_TYPED_MODEL
    assert selector.identity()["max_tokens"] == 1024
    assert selector.identity()["checkpoint_verification"] == "test_injected_unverified"


def test_native_laya_rejects_bad_binary_output_without_retry() -> None:
    agent = FakeAgent({"answers": {"evidence_selection": {"choice": "include", "probabilities": {"include": 1.0}, "confidence": 0.7}}, "usage": {"input_tokens": 2, "output_tokens": 0}})
    result = runtime.NativeLayaEvidenceSelector(None, _agent=agent, _torch=FakeTorch()).choose(OBSERVATION, INSTRUCTIONS, ACTIONS)
    assert result.outcome == "invalid_output" and result.error_code == "laya_probabilities_invalid"
    assert agent.calls == 1


def test_native_laya_preflight_fails_closed_before_forward() -> None:
    agent = FakeAgent()
    too_long = {"claim": "x " * 2000, "candidate": OBSERVATION["candidate"]}
    result = runtime.NativeLayaEvidenceSelector(None, _agent=agent).choose(too_long, INSTRUCTIONS, ACTIONS)
    assert result.outcome == "invalid_output" and result.error_code == "laya_preflight_state_truncated"
    assert agent.calls == 0


def test_native_laya_public_preflight_does_not_predict() -> None:
    agent = FakeAgent()
    selector = runtime.NativeLayaEvidenceSelector(None, _agent=agent)
    receipt = selector.preflight(OBSERVATION, INSTRUCTIONS, ACTIONS)
    assert receipt["total_tokens"] > 0 and agent.calls == 0


@pytest.mark.parametrize("usage", [None, {"input_tokens": 3}, {"input_tokens": True, "output_tokens": 0}, {"input_tokens": 3, "output_tokens": True}])
def test_native_laya_rejects_missing_or_invalid_usage_without_retry(usage) -> None:
    agent = FakeAgent({"answers": {"evidence_selection": {"choice": "include", "probabilities": {"include": 0.8, "exclude": 0.2}, "confidence": 0.7}}, "usage": usage})
    result = runtime.NativeLayaEvidenceSelector(None, _agent=agent, _torch=FakeTorch()).choose(OBSERVATION, INSTRUCTIONS, ACTIONS)
    assert result.outcome == "invalid_output" and result.error_code == "laya_usage_invalid"
    assert agent.calls == 1


def test_native_laya_blocks_cpu_fallback_before_a_second_forward() -> None:
    agent = CpuFallbackAgent()
    result = runtime.NativeLayaEvidenceSelector(None, _agent=agent, _torch=FakeTorch()).choose(OBSERVATION, INSTRUCTIONS, ACTIONS)
    assert result.outcome == "transport_error" and result.error_code == "laya_cuda_fallback_blocked"
    assert agent.calls == 1 and agent.model._model.cpu_attempts == 0
