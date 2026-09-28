from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import pytest

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
import run_laya as r
from engine import RecordingTools, State
from workers import SyntheticTools


def fixture_freeze():
    case = {'case_id': 'averitec-dev-0001', 'claim': 'A test claim', 'split': 'dev'}
    tools = RecordingTools(SyntheticTools(case))
    r.run_episode(r.OracleController(), tools)
    return {'cases': [{**case, 'group_id': 'group-1', 'role': 'development',
                       'records': tools.records, 'preparation_trace_id': 'prep-1'}]}


class FakeController:
    calls = 0

    def __init__(self, model, checkpoint_dir, **kwargs):
        self.model = self.expected_returned_model = model
        self.tracer = None
        self.timeout_seconds = 120

    def preflight(self, observation, actions):
        return {'lossless': True}

    def identity(self):
        return {'model': self.model}

    def choose(self, observation, instructions, actions):
        self.calls += 1
        result = replace(r.OracleController().choose(observation, instructions, actions),
                         model=self.model, returned_model=self.model)
        with self.tracer.span('controller.request', kind='generation', input={'observation': observation}):
            return result

    def close(self):
        pass


@pytest.fixture
def completed(tmp_path, monkeypatch):
    freeze = fixture_freeze()
    monkeypatch.setattr(r, 'NativeLayaController', FakeController)
    monkeypatch.setattr(r, 'runtime_identity', lambda: {'synthetic_test_only': True})
    tasks = r.plan(freeze, 'screen', 'base', r.settings())
    output = tmp_path / 'base'
    result = r.run_condition(output, 'base', tasks, freeze, tmp_path, {}, time.monotonic() + 60)
    return output, freeze, result


def test_expected_matrix():
    freeze = fixture_freeze()
    freeze['cases'] *= 4
    assert len(r.plan(freeze, 'screen', 'base', r.settings())) == 12
    assert len(r.plan(freeze, 'qualification', 'typed', r.settings())) == 60
    freeze['cases'] *= 25
    assert len(r.plan(freeze, 'evaluation', 'typed', r.settings())) == 300


def test_selection_rule():
    def condition(compliance, latency):
        return {'status': 'complete', 'score': {'decision_compliance': compliance, 'median_native_inference_latency_ms': latency}}
    assert r.select_checkpoint({'base': condition(.8, 5), 'typed': condition(.9, 10)}) == 'typed'
    assert r.select_checkpoint({'base': condition(.9, 5), 'typed': condition(.9, 10)}) == 'base'
    assert r.select_checkpoint({'base': condition(.9, 5), 'typed': condition(.9, 5)}) == 'base'
    assert r.select_checkpoint({'base': {'status': 'failed'}, 'typed': condition(.1, 20)}) == 'typed'
    with pytest.raises(ValueError, match='no_operational_screen'):
        r.select_checkpoint({'base': {'status': 'failed'}})


def test_condition_complete(completed):
    output, freeze, result = completed
    rows = r.audit_condition(output, freeze, 'screen', 'base', r.settings())
    assert len(rows) == 3
    assert result['status'] == 'complete'
    assert result['score']['decision_compliance'] == 1
    assert json.loads((output / 'warmup.json').read_text())['status'] == 'ok'


def test_wrong_phase_rejected(completed):
    output, freeze, _ = completed
    with pytest.raises(ValueError, match='extension_plan_mismatch'):
        r.audit_condition(output, freeze, 'qualification', 'base', r.settings())


def test_empty_traces_rejected(completed):
    output, freeze, _ = completed
    (output / 'traces.jsonl').write_text('')
    with pytest.raises(ValueError, match='extension_trace_trial_mismatch'):
        r.audit_condition(output, freeze, 'screen', 'base', r.settings())


def test_partial_rows_rejected(completed):
    output, freeze, _ = completed
    lines = (output / 'results.jsonl').read_text().splitlines()
    (output / 'results.jsonl').write_text('\n'.join(lines[:-1]) + '\n')
    with pytest.raises(ValueError, match='extension_trial_completeness'):
        r.audit_condition(output, freeze, 'screen', 'base', r.settings())


def test_budget_before_load(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('must not load')
    monkeypatch.setattr(r, 'NativeLayaController', forbidden)
    with pytest.raises(TimeoutError):
        r.run_condition(tmp_path / 'base', 'base', [], fixture_freeze(), tmp_path, {}, time.monotonic() - 1)
