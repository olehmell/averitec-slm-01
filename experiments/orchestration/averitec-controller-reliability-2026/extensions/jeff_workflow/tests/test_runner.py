from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import pytest

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))

import run_jeff as runner
from engine import RecordingTools
from workers import SyntheticTools


def fixture_freeze():
    case = {"case_id": "averitec-dev-0001", "claim": "A test claim", "split": "dev"}
    tools = RecordingTools(SyntheticTools(case))
    runner.run_episode(runner.OracleController(), tools)
    return {"cases": [{**case, "group_id": "group-1", "role": "development",
                       "records": tools.records, "preparation_trace_id": "prep-1"}]}


class FakeController:
    def __init__(self, _checkpoint, **_kwargs):
        self.model = runner.MODEL
        self.expected_returned_model = runner.RETURNED_MODEL
        self.tracer = None
        self.timeout_seconds = 120

    def preflight(self, _observation, _actions):
        return {"lossless": True}

    def identity(self):
        return {"model": self.model}

    def choose(self, observation, instructions, actions):
        result = replace(runner.OracleController().choose(observation, instructions, actions),
                         model=runner.MODEL, returned_model=runner.RETURNED_MODEL)
        with self.tracer.span("controller.request", kind="generation",
                              input={"observation": observation}):
            return result

    def close(self):
        pass


def test_expected_matrices():
    freeze = fixture_freeze()
    freeze["cases"] *= 4
    assert len(runner.plan(freeze, "qualification", runner.settings())) == 60
    freeze["cases"] *= 25
    assert len(runner.plan(freeze, "evaluation", runner.settings())) == 300


def test_condition_is_complete_and_auditable(tmp_path, monkeypatch):
    freeze = fixture_freeze()
    cfg = runner.settings()
    cfg["phases"]["qualification"] = {
        "cases": 1, "orders": ["canonical"], "canonical_repetitions": 1,
        "gpu_wall_seconds": 60,
    }
    monkeypatch.setattr(runner, "NativeJeffController", FakeController)
    monkeypatch.setattr(runner, "runtime_identity", lambda: {"synthetic_test_only": True})
    monkeypatch.setattr(runner, "settings", lambda: cfg)
    tasks = runner.plan(freeze, "qualification", cfg)
    output = tmp_path / "jeff"
    result = runner.run_condition(output, tasks, freeze, tmp_path, {}, time.monotonic() + 60)
    rows = runner.audit_condition(output, freeze, "qualification", cfg)
    assert len(rows) == 3
    assert result["status"] == "complete"
    assert result["score"]["decision_compliance"] == 1
    assert json.loads((output / "warmup.json").read_text())["status"] == "ok"


def test_budget_checked_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "NativeJeffController",
                        lambda *_args, **_kwargs: pytest.fail("must not load"))
    with pytest.raises(TimeoutError):
        runner.run_condition(tmp_path / "jeff", [], fixture_freeze(), tmp_path,
                             {}, time.monotonic() - 1)


def test_persisted_four_decimal_response_can_be_recovered():
    probabilities = {action: 0.1 for action in runner.ACTIONS}
    probabilities[runner.ACTIONS[0]] = 0.1997
    payload = {"model": runner.RETURNED_MODEL, "answers": {"next_action": {
        "type": "choice", "choice": runner.ACTIONS[0],
        "probabilities": probabilities, "confidence": 0.04,
    }}, "usage": {"input_tokens": 321, "output_tokens": 6}}
    parsed = runner.parse_native_response(payload, list(runner.ACTIONS))
    assert parsed["action"] == runner.ACTIONS[0]
    assert parsed["input_tokens"] == 321


def test_persisted_response_outside_rounding_tolerance_is_rejected():
    probabilities = {action: 0.1 for action in runner.ACTIONS}
    probabilities[runner.ACTIONS[0]] = 0.1994
    payload = {"model": runner.RETURNED_MODEL, "answers": {"next_action": {
        "type": "choice", "choice": runner.ACTIONS[0],
        "probabilities": probabilities, "confidence": 0.04,
    }}, "usage": {"input_tokens": 321, "output_tokens": 6}}
    with pytest.raises(ValueError, match="choice_invalid"):
        runner.parse_native_response(payload, list(runner.ACTIONS))
