import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from data import prepare_selection
from providers import ControllerResult


def inputs(tmp_path, *, limit=1):
    args = argparse.Namespace(role='development', limit=limit, repetitions=1,
                              orders=['canonical'], scenarios=['nominal'], controller='lfm',
                              mode='checkpoints', prompt_only=False, live_tools=False,
                              plan=False, output=tmp_path/'rows.jsonl', worker_endpoint=None,
                              endpoint='http://mock/v1', prompt_variant='v2', generation_profile='lfm_native')
    settings = run.config()
    freeze = run.create_freeze(args, settings, prepare_selection(settings), synthetic=True)
    args.freeze = tmp_path/'freeze.json'
    return args, settings, freeze


def fake_deployment(monkeypatch, requests, *, warmup_fails=False):
    # The model boundary is mocked: these are integration tests, never evidence.
    monkeypatch.setattr(run, 'require_committed_config', lambda: None)
    monkeypatch.setattr(run, 'check_freeze', lambda *a, **kw: None)
    class Fake:
        def __init__(self, *, model, timeout_seconds, **kwargs):
            self.model, self.timeout_seconds, self.tracer = model, timeout_seconds, None
            self.resolved_profile = {'name':kwargs['generation_profile']}
        def choose(self, observation, instructions, actions):
            requests.append((observation, instructions, actions, self.timeout_seconds))
            failed = warmup_fails and len(requests) == 1
            return ControllerResult(action=None if failed else 'finish',
                                    outcome='transport_error' if failed else 'ok',
                                    latency_ms=1, input_tokens=10, output_tokens=3,
                                    model=self.model, returned_model=self.model)
    monkeypatch.setattr(run, 'OpenAIController', Fake)


def test_real_boundary_presentation_and_warmup_are_separate(tmp_path, monkeypatch):
    args, settings, freeze = inputs(tmp_path)
    requests = []
    fake_deployment(monkeypatch, requests)
    rows = run.execute(args, settings, freeze)
    assert len(requests) == 9  # 1 valid-but-wrong warmup, 8 measured checkpoints
    assert requests[0][0]['pending_tool'] == 'decompose'
    assert requests[0][3] == 120
    assert all(request[3] == 30 for request in requests[1:])
    assert all(len(request[2]) == 9 for request in requests)
    assert len(rows[0]['events']) == rows[0]['planned_decisions'] == 8
    assert rows[0]['prompt_variant'] == 'v2'
    assert rows[0]['generation_profile'] == 'lfm_native'
    assert rows[0]['events'][0]['observation']['stage'] == 'decompose'
    assert not rows[0]['events'][0]['compliant']  # no correction of finish
    receipt = json.loads(Path(str(args.output)+'.run.json').read_text())
    assert receipt['generation_settings']['request_parameters']['temperature'] == .1
    assert len(list(tmp_path.glob('rows.jsonl.warmup-*.json'))) == 1
    run.execute(args, settings, freeze)
    assert len(requests) == 9  # a complete resume makes zero requests


def test_failed_warmup_does_not_write_measured_intent(tmp_path, monkeypatch):
    args, settings, freeze = inputs(tmp_path)
    requests = []
    fake_deployment(monkeypatch, requests, warmup_fails=True)
    with pytest.raises(ValueError, match='controller_warmup_failed'):
        run.execute(args, settings, freeze)
    assert len(requests) == 1
    assert not args.output.exists()
    assert not Path(str(args.output)+'.intents.jsonl').exists()
    receipts = list(tmp_path.glob('rows.jsonl.warmup-*.json'))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())['status'] == 'failed'


def test_evaluation_requires_explicit_study_phase_before_requests(tmp_path, monkeypatch):
    args, settings, freeze = inputs(tmp_path)
    args.role = 'evaluation'
    freeze['cases'][0]['role'] = 'evaluation'
    requests = []
    fake_deployment(monkeypatch, requests)
    with pytest.raises(ValueError, match='evaluation_requires_explicit_study_phase'):
        run.execute(args, settings, freeze)
    assert requests == []


@pytest.mark.parametrize('controller,expected', [('jev',5),('lfm',5),('qwen',7)])
def test_predeclared_sweep_size_and_no_profile_merging(tmp_path, monkeypatch, controller, expected):
    args, settings, freeze = inputs(tmp_path)
    args.controller, args.scenarios, args.plan = controller, list(run.SCENARIOS), True
    calls = []
    monkeypatch.setattr(run, 'execute', lambda condition, *_args: calls.append(condition) or [])
    run.development_sweep(args, settings, freeze)
    assert len(calls) == expected
    assert [(c.prompt_variant, c.generation_profile) for c in calls[:3]] == [('v1','baseline'),('v2','baseline'),('v3','baseline')]
    assert len({c.output for c in calls}) == expected
    assert all(c.role == 'development' and c.repetitions == 1 for c in calls)


@pytest.mark.parametrize('field,value,message', [
    ('role','evaluation','sweep_requires_structured_frozen_development'),
    ('limit',20,'sweep_exceeds_screening_case_limit'),
    ('repetitions',3,'screening_requires_single_canonical_pass'),
    ('scenarios',['nominal'],'screening_requires_all_predeclared_scenarios'),
])
def test_sweep_cannot_expand_scope(tmp_path, monkeypatch, field, value, message):
    args, settings, freeze = inputs(tmp_path)
    args.scenarios = list(run.SCENARIOS)
    setattr(args, field, value)
    monkeypatch.setattr(run, 'execute', lambda *a, **kw: pytest.fail('should fail before dispatch'))
    with pytest.raises(ValueError, match=message):
        run.development_sweep(args, settings, freeze)
