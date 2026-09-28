from __future__ import annotations

from collections import UserDict
import json
from pathlib import Path
import random
import sys

import pytest


EXPERIMENT = Path(__file__).resolve().parents[3]
EXTENSION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(EXTENSION))

import adapter
from engine import ACTIONS, State
from warmup import premeasure_warmup


class FakeTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 99
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        # Deterministic whitespace/punctuation approximation sufficient to test
        # budgeting mechanics without importing transformers.
        normalized = text
        for char in ",:;()=+-<>.\n":
            normalized = normalized.replace(char, f" {char} ")
        return UserDict({"input_ids": list(range(len(normalized.split())))})


class Device:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class FakeAgent:
    def __init__(self, *, max_len=512, head_max_len=192, device="cuda:0"):
        self.cfg = {"max_len": max_len, "head_max_len": head_max_len}
        self.tok = FakeTokenizer()
        self.device = Device(device)
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        actions = list(questions[adapter.QUESTION_ID]["criteria"])
        probabilities = {action: (0.6 if index == 0 else 0.4 / (len(actions) - 1))
                         for index, action in enumerate(actions)}
        return {
            "model": "laya-rl-agent",
            "answers": {adapter.QUESTION_ID: {
                "type": "choice", "choice": actions[0],
                "probabilities": probabilities, "confidence": 0.3142,
                "action": {"act_probability": 0.91},
            }},
            "usage": {"input_tokens": 211, "output_tokens": 0},
        }


class FakePackage:
    __version__ = adapter.PACKAGE_VERSION

    def __init__(self, agent):
        self.agent = agent
        self.loads = []

    def load(self, path, *, device):
        self.loads.append((path, device))
        return self.agent


class FakeCuda:
    def __init__(self):
        self.syncs = []
        self.empty_cache_calls = 0

    def synchronize(self, device):
        self.syncs.append(device)

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


class FallbackModel:
    def __init__(self):
        self.cpu_transfers = 0

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    def to(self, device):
        if str(device).startswith("cpu"):
            self.cpu_transfers += 1
        return self


class FallbackAgent(FakeAgent):
    """Minimal reproduction of laya 0.3.3's caught-error CPU retry."""

    def __init__(self):
        super().__init__()
        self.model = FallbackModel()
        self.first_forwards = 0
        self.second_forwards = 0

    def predict(self, state, questions):
        self.calls.append((state, questions))
        try:
            self.first_forwards += 1
            self.model("gpu batch")
        except RuntimeError:
            self.device = Device("cpu")
            self.model.to(self.device)
            self.second_forwards += 1
            self.model("cpu batch")


class TraceSpan:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class Tracer:
    def __init__(self):
        self.calls = []
        self.current = TraceSpan()

    def span(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self

    def __enter__(self):
        return self.current

    def __exit__(self, *_args):
        return False


def observation():
    state = State()
    state.completed = ["decompose", "queries"]
    state.attempts = 1
    state.calls = 4
    state.last_status = "timeout"
    state.metrics = {
        "facet_count": 2, "query_count": 4, "candidate_count": 0,
        "qa_count": 0, "selected_count": 0,
    }
    return state.observation()


def controller(*, model=adapter.BASE_MODEL_ID, max_len=512, head_max_len=192,
               device="cuda:0", tracer=None):
    agent = FakeAgent(max_len=max_len, head_max_len=head_max_len, device=device)
    torch = FakeTorch()
    value = adapter.NativeLayaController(
        model, None, device="cuda", tracer=tracer, _agent=agent, _torch=torch
    )
    return value, agent, torch


def test_choose_uses_fixed_native_profile_compact_v2_state_and_one_forward() -> None:
    value, agent, torch = controller()
    result = value.choose(observation(), "MALICIOUS CALLER INSTRUCTIONS", list(ACTIONS))

    assert result.outcome == "ok" and result.action == ACTIONS[0]
    assert result.returned_model == adapter.BASE_MODEL_ID
    assert result.output_tokens == 0 and result.input_tokens == 211
    assert result.confidence == 0.3142
    assert set(result.probabilities) == set(ACTIONS)
    assert len(agent.calls) == 1
    state_text, questions = agent.calls[0]
    state = json.loads(state_text)
    assert state["schema"] == "averitec-reliability-presentation/v2"
    assert state["pending_tool"] == "retrieve"
    assert "MALICIOUS" not in json.dumps(questions)
    question = questions[adapter.QUESTION_ID]
    assert question["instructions"] == adapter.NATIVE_INSTRUCTIONS
    assert question["criteria"] == {action: None for action in ACTIONS}
    assert torch.cuda.syncs == ["cuda:0", "cuda:0"]


def test_preflight_is_lossless_for_base_and_typed_in_all_planned_orders() -> None:
    orders = [list(ACTIONS), list(reversed(ACTIONS))]
    seeded = list(ACTIONS)
    random.Random(20260919).shuffle(seeded)
    orders.append(seeded)

    for model, max_len, head_max_len in (
        (adapter.BASE_MODEL_ID, 512, 192),
        (adapter.TYPED_MODEL_ID, 1024, 256),
    ):
        value, agent, _torch = controller(model=model, max_len=max_len, head_max_len=head_max_len)
        for actions in orders:
            report = value.preflight(observation(), actions)
            assert report["lossless"] is True
            assert report["total_tokens"] <= max_len
            assert report["state_tokens"] <= report["state_budget"]
        assert agent.calls == []


@pytest.mark.parametrize(
    "max_len,head_max_len,error",
    [
        (90, 192, "laya_preflight_state_truncated"),
        (512, 30, "laya_preflight_instructions_truncated"),
    ],
)
def test_preflight_rejects_native_truncation_before_forward(max_len, head_max_len, error) -> None:
    value, agent, _torch = controller()
    value._config = {"max_len": max_len, "head_max_len": head_max_len}
    with pytest.raises(ValueError, match=f"^{error}$"):
        value.choose(observation(), "ignored", list(ACTIONS))
    assert agent.calls == []


def test_preflight_rejects_missing_or_duplicate_global_actions() -> None:
    value, agent, _torch = controller()
    with pytest.raises(ValueError, match="^laya_invalid_actions$"):
        value.preflight(observation(), list(ACTIONS[:-1]) + [ACTIONS[0]])
    assert agent.calls == []


def test_cuda_fallback_is_fatal_at_construction_for_injected_agent() -> None:
    with pytest.raises(RuntimeError, match="^laya_cuda_fallback$"):
        controller(device="cpu")


def test_declared_model_must_match_native_checkpoint_budget() -> None:
    with pytest.raises(RuntimeError, match="^laya_checkpoint_model_mismatch$"):
        controller(model=adapter.TYPED_MODEL_ID, max_len=512, head_max_len=192)


def test_local_snapshot_must_be_complete_to_prevent_online_fallback(tmp_path) -> None:
    package = FakePackage(FakeAgent())
    value = adapter.NativeLayaController(
        adapter.BASE_MODEL_ID, tmp_path, device="cuda", _package=package, _torch=FakeTorch()
    )
    with pytest.raises(RuntimeError, match="^laya_checkpoint_incomplete$"):
        value.preflight(observation(), list(ACTIONS))
    assert package.loads == []


def test_complete_local_snapshot_is_loaded_lazily_with_pinned_package(tmp_path) -> None:
    for relative in (
        "rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json", "encoder/config.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    package = FakePackage(FakeAgent())
    value = adapter.NativeLayaController(
        adapter.BASE_MODEL_ID, tmp_path, device="cuda", _package=package, _torch=FakeTorch()
    )
    assert package.loads == []
    assert value.preflight(observation(), list(ACTIONS))["lossless"] is True
    assert package.loads == [(str(tmp_path), "cuda")]


def test_identity_binds_profile_runtime_and_non_generative_usage() -> None:
    value, _agent, _torch = controller(model=adapter.TYPED_MODEL_ID, max_len=1024, head_max_len=256)
    identity = value.identity()

    assert identity["package_version"] == "0.3.3"
    assert identity["model"] == adapter.TYPED_MODEL_ID
    assert identity["expected_returned_model"] == adapter.TYPED_MODEL_ID
    assert identity["returned_model_identity_source"] == "pinned_local_manifest"
    assert identity["cpu_fallback_policy"] == "blocked_before_retry_forward"
    assert identity["runtime_wrapper"] == "block_laya_cpu_retry_v1"
    assert identity["actual_device"] == "cuda:0"
    assert identity["max_tokens"] == 1024 and identity["head_max_tokens"] == 256
    assert identity["instruction_profile"] == "laya_native_v1"
    assert identity["instructions_sha256"] == adapter._digest_text(adapter.NATIVE_INSTRUCTIONS)
    assert identity["output_token_semantics"] == "zero_non_autoregressive_tokens"
    assert identity["confidence_semantics"] == "native_normalized_entropy_not_chosen_probability"


def test_trace_keeps_safe_raw_answer_and_entropy_semantics() -> None:
    tracer = Tracer()
    value, _agent, _torch = controller(tracer=tracer)
    result = value.choose(observation(), "ignored", list(ACTIONS))

    assert result.outcome == "ok"
    assert tracer.calls[0][0] == ("controller.request",)
    trace_input = tracer.calls[0][1]["input"]
    assert trace_input["instruction_profile"] == adapter.INSTRUCTION_PROFILE
    assert trace_input["runtime_wrapper"] == adapter.RUNTIME_WRAPPER
    assert trace_input["questions"][adapter.QUESTION_ID]["criteria"] == {
        action: None for action in ACTIONS
    }
    update = tracer.current.updates[0]
    assert update["output"]["answers"][adapter.QUESTION_ID]["confidence"] == 0.3142
    assert update["metadata"]["confidence_semantics"] == "native_normalized_entropy_not_chosen_probability"
    assert update["usage"] == {"input": 211, "output": 0}


@pytest.mark.parametrize("failure", [RuntimeError("secret body"), ValueError("token=secret")])
def test_native_failure_is_sanitized_and_not_retried(failure) -> None:
    value, agent, _torch = controller()

    def fail(*_args, **_kwargs):
        agent.calls.append("attempt")
        raise failure

    agent.predict = fail
    result = value.choose(observation(), "ignored", list(ACTIONS))
    assert result.outcome == "transport_error"
    assert result.error_code == "laya_native_inference_failed"
    assert result.action is None and result.output_tokens == 0
    assert result.returned_model == adapter.BASE_MODEL_ID
    assert agent.calls == ["attempt"]
    assert "secret" not in repr(result)


def test_laya_cuda_error_cpu_fallback_is_blocked_before_second_forward() -> None:
    agent = FallbackAgent()
    value = adapter.NativeLayaController(
        adapter.BASE_MODEL_ID, None, device="cuda", _agent=agent, _torch=FakeTorch()
    )

    result = value.choose(observation(), "ignored", list(ACTIONS))

    assert result.outcome == "transport_error"
    assert result.error_code == "laya_cuda_fallback_blocked"
    assert result.returned_model == adapter.BASE_MODEL_ID
    assert len(agent.calls) == 1
    assert agent.first_forwards == 1
    assert agent.second_forwards == 0
    assert agent.model._model.cpu_transfers == 0
    with pytest.raises(RuntimeError, match="^laya_cuda_fallback_blocked$"):
        value.preflight(observation(), list(ACTIONS))


def test_invalid_native_distribution_is_recorded_without_repair() -> None:
    value, agent, _torch = controller()
    original = agent.predict

    def invalid(state, questions):
        payload = original(state, questions)
        payload["answers"][adapter.QUESTION_ID]["probabilities"] = {
            action: 0.5 for action in ACTIONS
        }
        return payload

    agent.predict = invalid
    result = value.choose(observation(), "ignored", list(ACTIONS))
    assert result.outcome == "invalid_output"
    assert result.error_code == "laya_probabilities_invalid"
    assert result.action is None and len(agent.calls) == 1


def test_close_is_idempotent_and_best_effort_cuda_cleanup() -> None:
    value, _agent, torch = controller()
    value.close()
    value.close()
    assert torch.cuda.syncs == ["cuda:0"]
    assert torch.cuda.empty_cache_calls == 1
    with pytest.raises(RuntimeError, match="^laya_controller_closed$"):
        value.identity()


def test_existing_premeasure_warmup_accepts_pinned_local_identity_once() -> None:
    value, agent, _torch = controller()
    original_timeout = value.timeout_seconds

    receipt = premeasure_warmup(
        value,
        instructions=adapter.NATIVE_INSTRUCTIONS,
        actions=list(ACTIONS),
        expected_model=adapter.BASE_MODEL_ID,
    )

    assert receipt["status"] == "ok"
    assert receipt["model"] == adapter.BASE_MODEL_ID
    assert receipt["expected_returned_model"] == adapter.BASE_MODEL_ID
    assert receipt["returned_model"] == adapter.BASE_MODEL_ID
    assert len(agent.calls) == 1
    assert value.timeout_seconds == original_timeout == 120
