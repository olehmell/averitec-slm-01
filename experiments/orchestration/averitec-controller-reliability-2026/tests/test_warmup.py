from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from providers import ControllerResult
from warmup import (WARMUP_ACTIONS, WARMUP_OBSERVATION, WARMUP_TIMEOUT_SECONDS,
                    WarmupFailure, premeasure_warmup)


class Controller:
    def __init__(self, result: ControllerResult | None = None, error: Exception | None = None):
        self.model = "Qwen/Qwen3.5-4B"
        self.timeout_seconds = 30
        self.tracer = "measured-tracer"
        self.result = result
        self.error = error
        self.calls = []
        self.resolved_profile = {"name": "qwen_native", "model": self.model}

    def choose(self, observation, instructions, actions):
        self.calls.append({"observation": observation, "instructions": instructions,
                           "actions": actions, "timeout": self.timeout_seconds,
                           "tracer": self.tracer})
        if self.error:
            raise self.error
        assert self.result is not None
        return self.result


def _result(*, action="finish", outcome="ok", returned_model="Qwen/Qwen3.5-4B"):
    return ControllerResult(action=action, outcome=outcome, latency_ms=2.0,
                            input_tokens=3, output_tokens=4,
                            model="Qwen/Qwen3.5-4B", returned_model=returned_model)


def test_warmup_accepts_any_valid_action_without_oracle_correctness_and_restores_state() -> None:
    controller = Controller(_result(action="finish"))  # The synthetic state would expect decompose.
    receipt = premeasure_warmup(controller, instructions="same controller instructions", actions=list(WARMUP_ACTIONS),
                                 tracer="warmup-tracer")

    assert receipt["status"] == "ok"
    assert receipt["usage"] == {"input_tokens": 3, "output_tokens": 4}
    assert receipt["profile"] == controller.resolved_profile
    assert controller.timeout_seconds == 30 and controller.tracer == "measured-tracer"
    assert controller.calls == [{
        "observation": WARMUP_OBSERVATION,
        "instructions": "same controller instructions",
        "actions": list(WARMUP_ACTIONS),
        "timeout": WARMUP_TIMEOUT_SECONDS,
        "tracer": "warmup-tracer",
    }]


def test_warmup_failure_aborts_after_exactly_one_call_and_restores_state() -> None:
    controller = Controller(_result(action=None, outcome="invalid_output"))
    with pytest.raises(WarmupFailure) as raised:
        premeasure_warmup(controller, instructions="same controller instructions", actions=list(WARMUP_ACTIONS))

    assert raised.value.receipt.status == "failed"
    assert len(controller.calls) == 1
    assert controller.timeout_seconds == 30 and controller.tracer == "measured-tracer"


def test_warmup_detaches_an_existing_measured_tracer_when_no_warmup_tracer_is_supplied() -> None:
    controller = Controller(_result())
    premeasure_warmup(controller, instructions="same controller instructions", actions=list(WARMUP_ACTIONS))

    assert controller.calls[0]["tracer"] is None
    assert controller.tracer == "measured-tracer"


def test_warmup_nested_observation_mutation_cannot_contaminate_the_next_request() -> None:
    class MutatingController(Controller):
        incoming_qa_counts: list[object]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.incoming_qa_counts = []

        def choose(self, observation, instructions, actions):
            self.incoming_qa_counts.append(observation["metrics"]["qa_count"])
            observation["metrics"]["qa_count"] = 999
            return super().choose(observation, instructions, actions)

    controller = MutatingController(_result())
    premeasure_warmup(controller, instructions="same controller instructions", actions=list(WARMUP_ACTIONS))
    premeasure_warmup(controller, instructions="same controller instructions", actions=list(WARMUP_ACTIONS))

    assert controller.incoming_qa_counts == [None, None]
    assert WARMUP_OBSERVATION["metrics"]["qa_count"] is None


def test_warmup_requires_actual_returned_model_identity() -> None:
    controller = Controller(_result(returned_model="other-model"))
    with pytest.raises(WarmupFailure) as raised:
        premeasure_warmup(controller, instructions="same controller instructions", actions=list(WARMUP_ACTIONS))
    assert raised.value.receipt.status == "failed"
    assert len(controller.calls) == 1


def test_warmup_restores_timeout_and_tracer_when_choose_raises() -> None:
    controller = Controller(error=RuntimeError("transport exploded"))
    with pytest.raises(RuntimeError, match="transport exploded"):
        premeasure_warmup(controller, instructions="same controller instructions", actions=list(WARMUP_ACTIONS),
                           tracer="warmup-tracer")
    assert len(controller.calls) == 1
    assert controller.timeout_seconds == 30 and controller.tracer == "measured-tracer"


def test_warmup_rejects_non_global_action_vocabulary_without_a_request() -> None:
    controller = Controller(_result())
    with pytest.raises(ValueError, match="warmup_requires_global_action_vocabulary"):
        premeasure_warmup(controller, instructions="same controller instructions", actions=["finish"])
    assert controller.calls == []
