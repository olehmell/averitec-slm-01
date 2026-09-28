"""Offline contract tests for the six original native workflow passes."""
from __future__ import annotations

import importlib.util
from argparse import Namespace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'native_workflow_passes.py'
spec = importlib.util.spec_from_file_location('native_workflow_passes_test', MODULE)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def fake_freeze():
    return {'cases': [{'case_id': f'averitec-dev-{n:04}', 'group_id': f'group-{n}',
                       'role': 'evaluation', 'records': [{'number': n}]} for n in range(100)]}


def fake_episode(_controller, tools):
    number = tools.tools.records[0]['number']
    scenario = runner.SCENARIOS.index(tools.scenario)
    count = 8 if number * 3 + scenario < 199 else 7
    events = []
    for step in range(count):
        observation = {'case_number': number, 'scenario': tools.scenario, 'step': step}
        events.append({'step': step, 'observation': observation,
                       'observation_sha256': runner.object_digest(observation),
                       'expected_action': runner.ACTIONS[step % len(runner.ACTIONS)]})
    return {'events': events}


class FakeReplay:
    def __init__(self, records):
        self.records = records


class FakeFault:
    def __init__(self, tools, scenario):
        self.tools, self.scenario = tools, scenario


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(runner, 'ReplayTools', FakeReplay)
    monkeypatch.setattr(runner, 'FaultTools', FakeFault)
    monkeypatch.setattr(runner, 'run_episode', fake_episode)
    return fake_freeze()


def test_exact_plan_and_new_pass_identity(offline):
    plans = {(arm, number): runner.make_plan(offline, arm, number)
             for arm in runner.ARM_INFO for number in (1, 2, 3)}
    assert len({p['plan_sha256'] for p in plans.values()}) == 6
    for (arm, number), plan in plans.items():
        assert len(plan['trials']) == 300
        assert len(plan['checkpoints']) == 2299
        assert len({x['trial_id'] for x in plan['trials']}) == 300
        assert len({x['checkpoint_id'] for x in plan['checkpoints']}) == 2299
        assert all(t['arm'] == arm and t['pass'] == number for t in plan['trials'])
    assert plans['jeff', 1]['trials'][0]['case_id'] == plans['jeff', 3]['trials'][0]['case_id']
    assert set(runner.ARM_INFO['jeff']) == {'knowledgator/gliformer-large-v1', 'gliformer-large-v1', 'jeff_native_workflow_v1'}
    assert runner.ARM_INFO['laya_typed'][2] == 'laya_native_v1'


def test_adapter_isolation_and_native_profiles():
    jeff = runner.load_adapter('jeff')
    laya = runner.load_adapter('laya_typed')
    assert jeff is not laya
    assert jeff.PROFILE == 'jeff_native_workflow_v1'
    assert jeff.PROBABILITY_SUM_TOLERANCE == 0.0005
    assert laya.INSTRUCTION_PROFILE == 'laya_native_v1'
    assert laya.PRESENTATION_NAME == 'v2'  # historical constant; _present uses v4
    assert laya.EXPECTED_BUDGETS[laya.TYPED_MODEL_ID] == (1024, 256)


def test_lossless_preflight_and_token_cap(offline):
    plan = runner.make_plan(offline, 'jeff', 1)
    class Controller:
        def preflight(self, _observation, _actions):
            return {'lossless': True, 'input_tokens': 12}
    caps = {'input_tokens_per_request': 20, 'context_tokens': 20}
    report = runner.preflight(Controller(), plan, offline, 'jeff', caps)
    assert report['checks'][-1]['warmup'] is True
    class Truncated(Controller):
        def preflight(self, _observation, _actions):
            return {'lossless': False, 'input_tokens': 12}
    with pytest.raises(ValueError, match='preflight_loss'):
        runner.preflight(Truncated(), plan, offline, 'jeff', caps)
    with pytest.raises(ValueError, match='token_cap'):
        runner.preflight(Controller(), plan, offline, 'jeff', {'input_tokens_per_request': 8, 'context_tokens': 20})


def test_gate_is_exact_and_scoped(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    gate = tmp_path / 'gate.json'
    payload = {'schema': runner.GATE_SCHEMA, 'manifest_sha256': runner.sha(manifest),
               'decision': 'approved', 'reviewer': 'gpt-6-astra', 'arm': 'jeff', 'pass': 1}
    gate.write_text(json.dumps(payload))
    runner.check_gate(gate, manifest, 'jeff', 1)
    with pytest.raises(ValueError, match='astra_gate'):
        runner.check_gate(gate, manifest, 'jeff', 2)
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='astra_gate'):
        runner.check_gate(gate, manifest, 'jeff', 1)


def test_intent_order_and_unknown_failure_accounting(offline):
    plan = runner.make_plan(offline, 'laya_typed', 2)
    first = plan['checkpoints'][0]
    response = {'kind': 'response', **first, 'provider_outcome': 'ok', 'compliant': True}
    entries = [
        {'kind': 'request_intent', 'checkpoint_id': 'warmup'},
        {'kind': 'warmup_response', 'checkpoint_id': 'warmup', 'outcome': 'ok'},
        {'kind': 'request_intent', 'checkpoint_id': first['checkpoint_id']},
        response,
    ]
    summary = runner.audit(plan, entries)
    assert summary['status'] == 'incomplete'
    assert summary['missing_checkpoints'] == 2298
    assert summary['failed_checkpoints'] == 2298
    entries += [{'kind': 'request_intent', 'checkpoint_id': plan['checkpoints'][1]['checkpoint_id']},
                {'kind': 'uncertain', 'checkpoint_id': plan['checkpoints'][1]['checkpoint_id']}]
    assert runner.audit(plan, entries)['status'] == 'incomplete'
    with pytest.raises(ValueError, match='prior_intent'):
        runner.audit(plan, [response])
    with pytest.raises(ValueError, match='intent_order'):
        runner.audit(plan, [entries[2]])


def test_no_evaluator_gold_argument_or_import():
    source = MODULE.read_text()
    assert '--gold' not in source
    assert 'reference_evaluator_only' not in source
    assert 'verdict_gold_evaluator_only' not in source
    assert 'open_evaluator' not in source


def test_execute_ledger_and_terminal_receipts(tmp_path, monkeypatch, offline):
    plan = runner.make_plan(offline, 'jeff', 1)
    manifest = tmp_path / 'manifest.json'
    gate = tmp_path / 'gate.json'
    manifest.write_text('{}')
    gate.write_text('{}')
    class Controller:
        calls = 0
        fail = False
        model = runner.ARM_INFO['jeff'][0]
        def identity(self):
            return {'model': self.model, 'expected_returned_model': runner.ARM_INFO['jeff'][1],
                    'instruction_profile': runner.ARM_INFO['jeff'][2]}
        def preflight(self, _observation, _actions):
            return {'lossless': True, 'input_tokens': 10}
        def choose(self, _observation, _instructions, _actions):
            self.calls += 1
            outcome = 'transport_error' if self.fail and self.calls == 2 else 'ok'
            return SimpleNamespace(action='finish', outcome=outcome, model=self.model,
                returned_model=runner.ARM_INFO['jeff'][1], latency_ms=1.0,
                input_tokens=10, output_tokens=None, error_code=None,
                probabilities=None, confidence=None)
        def close(self):
            pass
    controller = Controller()
    caps = {'wall_seconds': 1020, 'gpu_seconds': 1020, 'gpu_jobs': 1,
            'requests': 2300, 'input_tokens_total': 30000,
            'input_tokens_per_request': 20, 'context_tokens': 20}
    report = runner.preflight(controller, plan, offline, 'jeff', caps)
    value = {'output_namespaces': {'jeff:1': 'jeff-pass-1'}, 'caps': {'jeff:1': caps},
             'preflight_sha256': {'jeff': report['sha256']}}
    monkeypatch.setattr(runner, 'validate_manifest', lambda *_args: (value, offline, plan))
    monkeypatch.setattr(runner, 'check_gate', lambda *_args: None)
    monkeypatch.setattr(runner, 'verify_staged', lambda *_args: None)
    monkeypatch.setattr(runner, '_controller', lambda *_args: controller)
    args = Namespace(arm='jeff', pass_number=1, manifest=manifest, gate=gate,
                     freeze=tmp_path / 'freeze.json', preparation=tmp_path / 'preparation.jsonl',
                     output=tmp_path / 'jeff-pass-1', checkpoint=tmp_path / 'checkpoint')
    receipt = runner.execute(args)
    assert receipt['summary']['status'] == 'complete'
    assert receipt['request_intents'] == 2300
    ledger = [json.loads(line) for line in (args.output / 'ledger.jsonl').read_text().splitlines()]
    assert ledger[0]['kind'] == 'request_intent' and ledger[0]['checkpoint_id'] == 'warmup'
    assert ledger[1]['kind'] == 'warmup_response'
    assert ledger[2]['kind'] == 'request_intent' and ledger[3]['kind'] == 'response'
    with pytest.raises(ValueError, match='already_exists'):
        runner.execute(args)

    controller.calls = 0
    controller.fail = True
    value['output_namespaces']['jeff:1'] = 'jeff-pass-1-unknown'
    args.output = tmp_path / 'jeff-pass-1-unknown'
    with pytest.raises(RuntimeError, match='native_uncertain_result'):
        runner.execute(args)
    failed = json.loads((args.output / 'receipt.json').read_text())
    assert failed['summary']['status'] == 'incomplete'
    assert failed['summary']['missing_checkpoints'] == 2299
    ledger = [json.loads(line) for line in (args.output / 'ledger.jsonl').read_text().splitlines()]
    assert ledger[-2]['kind'] == 'request_intent' and ledger[-1]['kind'] == 'uncertain'
