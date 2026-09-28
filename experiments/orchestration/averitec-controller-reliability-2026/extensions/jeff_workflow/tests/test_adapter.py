from __future__ import annotations

import math
from pathlib import Path
import random
import sys

import pytest

EXT = Path(__file__).resolve().parents[1]
EXP = EXT.parents[1]
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(EXP))

import adapter
from engine import ACTIONS, INSTRUCTIONS, State
from warmup import premeasure_warmup


class Runtime:
    def __init__(self):
        self.calls = []
        self.preflights = []
        self.identity = {"model": adapter.MODEL, "evidence": {"test": True}}

    def preflight(self, body):
        self.preflights.append(body)
        return {"input_tokens": 41, "state_characters": 120}

    def execute(self, body):
        self.calls.append(body)
        actions = list(body["questions"][adapter.QUESTION_ID]["criteria"])
        probabilities = {action: 0.01 for action in actions}
        probabilities[actions[0]] = 0.92
        return {"model": adapter.RETURNED_MODEL, "answers": {adapter.QUESTION_ID: {
            "type": "choice", "choice": actions[0], "probabilities": probabilities,
            "confidence": 0.75}}, "usage": {"input_tokens": 41, "output_tokens": 0}}


def observation():
    state = State()
    state.completed = ["decompose", "queries"]
    state.attempts = 1
    state.calls = 4
    state.last_status = "timeout"
    state.metrics = {"facet_count": 2, "query_count": 4, "candidate_count": 0,
                     "qa_count": 0, "selected_count": 0}
    return state.observation()


def test_one_native_nine_way_forward_in_caller_order():
    runtime = Runtime()
    actions = list(ACTIONS)
    random.Random(11).shuffle(actions)
    controller = adapter.NativeJeffController("unused", _runtime=runtime)
    assert controller.timeout_seconds == 120
    result = controller.choose(observation(), INSTRUCTIONS, actions)
    assert result.outcome == "ok" and result.action == actions[0]
    assert result.returned_model == adapter.RETURNED_MODEL and result.output_tokens is None
    assert len(runtime.calls) == 1 and len(runtime.preflights) == 1
    body = runtime.calls[0]
    assert list(body["questions"][adapter.QUESTION_ID]["criteria"]) == actions
    assert body["state"]["schema"] == "averitec-reliability-presentation/v2"
    assert body["questions"][adapter.QUESTION_ID]["instructions"] == adapter.NATIVE_INSTRUCTIONS


def test_controller_satisfies_standard_warmup_contract():
    controller = adapter.NativeJeffController("unused", _runtime=Runtime())
    receipt = premeasure_warmup(controller, instructions=INSTRUCTIONS,
                                actions=list(ACTIONS))
    assert receipt["status"] == "ok"
    assert receipt["returned_model"] == adapter.RETURNED_MODEL


def test_four_decimal_nine_way_rounding_is_accepted():
    runtime = Runtime()
    original = runtime.execute
    def rounded(body):
        payload = original(body)
        probabilities = payload["answers"][adapter.QUESTION_ID]["probabilities"]
        probabilities[next(iter(probabilities))] -= 0.0003
        return payload
    runtime.execute = rounded
    result = adapter.NativeJeffController("unused", _runtime=runtime).choose(
        observation(), INSTRUCTIONS, list(ACTIONS))
    assert result.outcome == "ok"


@pytest.mark.parametrize("actions", [list(ACTIONS[:-1]), list(ACTIONS) + ["other"],
                                      list(ACTIONS[:-1]) + [ACTIONS[0]]])
def test_action_vocabulary_is_closed(actions):
    controller = adapter.NativeJeffController("unused", _runtime=Runtime())
    with pytest.raises(ValueError, match="invalid_actions"):
        controller.choose(observation(), INSTRUCTIONS, actions)


def test_preflight_failure_blocks_forward():
    runtime = Runtime()
    runtime.preflight = lambda _body: (_ for _ in ()).throw(ValueError("truncated"))
    controller = adapter.NativeJeffController("unused", _runtime=runtime)
    with pytest.raises(ValueError, match="truncated"):
        controller.choose(observation(), INSTRUCTIONS, list(ACTIONS))
    assert runtime.calls == []


@pytest.mark.parametrize("change", ["model", "choice", "probabilities", "nan", "usage"])
def test_invalid_native_response_is_retained_as_invalid_output(change):
    runtime = Runtime()
    original = runtime.execute

    def execute(body):
        payload = original(body)
        answer = payload["answers"][adapter.QUESTION_ID]
        if change == "model": payload["model"] = "jev"
        if change == "choice": answer["choice"] = "unknown"
        if change == "probabilities": answer["probabilities"].pop(next(iter(answer["probabilities"])))
        if change == "nan": answer["probabilities"][next(iter(answer["probabilities"]))] = math.nan
        if change == "usage": payload["usage"]["input_tokens"] = True
        return payload

    runtime.execute = execute
    result = adapter.NativeJeffController("unused", _runtime=runtime).choose(
        observation(), INSTRUCTIONS, list(ACTIONS))
    assert result.outcome != "ok" and len(runtime.calls) == 1


def test_native_exception_has_no_retry():
    runtime = Runtime()
    attempts = []
    def fail(body):
        attempts.append(body)
        raise RuntimeError("private detail")
    runtime.execute = fail
    result = adapter.NativeJeffController("unused", _runtime=runtime).choose(
        observation(), INSTRUCTIONS, list(ACTIONS))
    assert result.outcome == "transport_error" and len(attempts) == 1
