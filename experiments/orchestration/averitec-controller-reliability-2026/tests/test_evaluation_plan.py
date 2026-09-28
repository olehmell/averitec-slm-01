from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation_plan
import run
from data import code_binding, file_hash, prepare_selection, selected_cases, write_new
from engine import FaultTools, OracleController, ReplayTools, canonical, digest, run_episode


def _args(tmp_path, **changes):
    values = dict(role='development', limit=4, case_offset=0, repetitions=3,
                  orders=['canonical', 'reversed', 'seeded'], scenarios=list(run.SCENARIOS),
                  controller='qwen', mode='checkpoints', prompt_only=False, live_tools=False,
                  plan=False, output=tmp_path/'results.jsonl', worker_endpoint=None,
                  endpoint='http://unused/v1', prompt_variant='v2', generation_profile='baseline',
                  study_phase='qualification', qualification_receipt=None)
    values.update(changes)
    return argparse.Namespace(**values)


def _qualification_evidence(tmp_path):
    settings = run.config()
    selection = prepare_selection(settings)
    freeze_args = _args(tmp_path, case_offset=2, output=tmp_path/'qualification-dev4.json')
    freeze = run.create_freeze(freeze_args, settings, selection, synthetic=True)
    freeze['origin'] = 'real_tools'
    freeze['code_binding'] = code_binding()
    write_new(freeze_args.output, freeze)
    freeze_sha = digest(freeze)
    run_paths = []
    for controller, (variant, profile) in evaluation_plan.CONDITIONS.items():
        run_path = tmp_path/controller/'results.jsonl'
        run_path.parent.mkdir()
        tasks = evaluation_plan._expected_qualification_tasks(selection, controller)
        run_id = digest({'controller': controller})
        generation = evaluation_plan._expected_generation(settings, controller, variant, profile)
        metadata = {
            'schema': 'averitec-controller-run/v1', 'freeze_sha256': freeze_sha,
            'config_sha256': file_hash(Path(run.EXPERIMENT)/'config.yaml'),
            'code_binding': code_binding(), 'origin': 'real_controller_frozen_tools',
            'output_mode': 'structured', 'prompt_variant': variant,
            'generation_profile': profile, 'model': settings['models'][controller],
            'planned_tasks': tasks, 'run_id': run_id, 'generation_settings': generation,
            'warmup_policy': 'one_synthetic_request_before_pending_measurements',
            **evaluation_plan.presentation_identity(variant),
        }
        write_new(Path(str(run_path)+'.run.json'), metadata)
        result_lines, intent_lines, trace_lines = [], [], []
        by_case = {case['case_id']: case for case in freeze['cases']}
        for task in tasks:
            trace_id = evaluation_plan._trace_id(run_id, task['trial_id'])
            reference = run_episode(OracleController(), FaultTools(
                ReplayTools(by_case[task['case_id']]['records']), task['scenario']))
            events = [{
                'step': item['step'], 'observation': item['observation'],
                'observation_sha256': item['observation_sha256'],
                'expected_action': item['expected_action'], 'action': None,
                'provider_outcome': 'invalid_output',
                'model': settings['models'][controller]['model'], 'returned_model': None,
                'latency_ms': 1, 'input_tokens': 1, 'output_tokens': 0,
                'error_code': 'invalid_json', 'confidence': None, 'probabilities': None,
                'compliant': False,
            } for item in reference['events']]
            result = {'outcome': 'checkpoints_evaluated', 'events': events,
                      'full_pipeline_completed': None, 'protocol_correct_termination': None,
                      'controller_calls': len(events)}
            row = {**task, **result, 'origin': metadata['origin'], 'trace_id': trace_id,
                   'preparation_trace_id': by_case[task['case_id']]['preparation_trace_id'],
                   'freeze_sha256': freeze_sha,
                   'shared_worker_preparation_outcome': by_case[task['case_id']]['preparation_outcome'],
                   'planned_decisions': len(events)}
            row['row_sha256'] = digest(row)
            result_lines.append(canonical(row))
            intent_lines.append(canonical({'trial_id': task['trial_id'], 'trace_id': trace_id}))
            root_span = task['trial_id'][:16]
            trace_lines.append(canonical({'event': 'span_start', 'trace_id': trace_id,
                                          'span_id': root_span, 'name': 'controller.trial',
                                          'parent_span_id': None, 'input': task}))
            for index, event in enumerate(events):
                span_id = f'{index:016x}'
                trace_lines.append(canonical({'event': 'span_start', 'trace_id': trace_id,
                                              'span_id': span_id, 'name': 'controller.decision',
                                              'parent_span_id': root_span}))
                trace_lines.append(canonical({'event': 'span_end', 'trace_id': trace_id,
                                              'span_id': span_id, 'output': event}))
            trace_lines.append(canonical({'event': 'span_end', 'trace_id': trace_id,
                                          'span_id': root_span, 'output': result}))
        run_path.write_text('\n'.join(result_lines)+'\n')
        Path(str(run_path)+'.intents.jsonl').write_text('\n'.join(intent_lines)+'\n')
        Path(str(run_path)+'.traces.jsonl').write_text('\n'.join(trace_lines)+'\n')
        warmup_id = digest({'warmup': controller})[:32]
        warm_trace = evaluation_plan._trace_id(run_id, 'warmup_' + warmup_id)
        warmup_trace_path = Path(str(run_path)+'.warmup.traces.jsonl')
        warmup_receipt = {
            'status': 'ok', 'duration_ms': 1, 'usage': {'input_tokens': 1, 'output_tokens': 1},
            'profile': None, 'model': settings['models'][controller]['model'],
            'expected_returned_model': settings['models'][controller].get(
                'expected_version', settings['models'][controller]['model']),
            'returned_model': settings['models'][controller].get(
                'expected_version', settings['models'][controller]['model']),
            'outcome': 'ok', 'error_code': None,
        }
        warmup_trace_path.write_text(canonical({
            'event': 'span_start', 'trace_id': warm_trace, 'span_id': warmup_id[:16],
            'name': 'controller.warmup', 'parent_span_id': None})+'\n'+canonical({
            'event': 'span_end', 'trace_id': warm_trace, 'span_id': warmup_id[:16],
            'output': warmup_receipt})+'\n')
        write_new(Path(str(run_path)+'.warmup-'+warmup_id+'.json'), {
            'run_id': run_id, 'prompt_variant': variant, 'generation_profile': profile,
            'generation_settings': generation, **warmup_receipt,
        })
        run_paths.append(run_path)
    receipt_path = tmp_path/'gate.json'
    receipt = evaluation_plan.create_qualification_receipt(
        run_paths, freeze_args.output, receipt_path, settings)
    write_new(receipt_path, receipt)
    return settings, receipt_path, run_paths


def test_selected_cases_offset_is_canonical_and_nonnegative(tmp_path):
    settings = run.config()
    selection = prepare_selection(settings)
    expected = [row['case_id'] for row in selection['cases'] if row['role'] == 'development'][2:6]
    assert [row['case_id'] for row in selected_cases(selection, 'development', limit=4, offset=2)] == expected
    with pytest.raises(ValueError, match='case_offset_must_be_nonnegative'):
        selected_cases(selection, 'development', limit=4, offset=-1)


def test_freeze_plan_returns_exact_slice_without_constructing_worker(tmp_path, monkeypatch):
    settings = run.config()
    selection = prepare_selection(settings)
    args = _args(tmp_path, case_offset=2, plan=True)
    monkeypatch.setattr(run, 'create_freeze', lambda *a, **k: pytest.fail('worker freeze entered'))
    planned = run.plan_freeze(args, settings, selection)
    expected = [row['case_id'] for row in selection['cases'] if row['role'] == 'development'][2:6]
    assert planned['case_ids'] == expected
    assert planned['case_count'] == 4
    assert planned['model_calls'] == 0 and planned['ready_for_worker_calls'] is False


@pytest.mark.parametrize('change,message', [
    ({'role': 'evaluation'}, 'qualification_requires_development_role'),
    ({'case_offset': 2}, 'qualification_requires_approved_case_slice'),
    ({'limit': 3}, 'qualification_requires_approved_case_slice'),
    ({'orders': ['canonical']}, 'qualification_requires_approved_execution_matrix'),
    ({'repetitions': 1}, 'qualification_requires_approved_execution_matrix'),
    ({'prompt_variant': 'v3'}, 'qualification_requires_approved_controller_profile'),
    ({'prompt_only': True}, 'study_phase_requires_structured_frozen_tools'),
])
def test_qualification_execution_contract_is_exact(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        evaluation_plan.enforce_study_phase(_args(tmp_path, **change), run.config(), operation='execute')


def test_receipt_is_revalidated_and_not_a_self_reported_boolean(tmp_path):
    settings, receipt_path, run_paths = _qualification_evidence(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt['passed'] = False
    receipt_path.write_text(json.dumps(receipt))
    evaluation_plan.validate_qualification_receipt(receipt_path, settings)

    run_paths[0].write_text(run_paths[0].read_text().replace('checkpoints_evaluated', 'tampered', 1))
    with pytest.raises(ValueError, match='qualification_evidence_hash_mismatch'):
        evaluation_plan.validate_qualification_receipt(receipt_path, settings)


def test_evaluation_execution_contract_is_exact(tmp_path):
    settings, receipt_path, _run_paths = _qualification_evidence(tmp_path)
    approved = _args(tmp_path, role='evaluation', study_phase='evaluation', limit=100,
                     repetitions=1, orders=['canonical'], controller='lfm', prompt_variant='v3',
                     generation_profile='lfm_native', qualification_receipt=receipt_path)
    evaluation_plan.enforce_study_phase(approved, settings, operation='execute')
    for changes, message in [
        ({'role': 'development'}, 'evaluation_requires_evaluation_role'),
        ({'limit': 99}, 'evaluation_requires_approved_case_slice'),
        ({'repetitions': 2}, 'evaluation_requires_approved_execution_matrix'),
        ({'prompt_variant': 'v2'}, 'evaluation_requires_approved_controller_profile'),
    ]:
        values = vars(approved) | changes
        with pytest.raises(ValueError, match=message):
            evaluation_plan.enforce_study_phase(argparse.Namespace(**values), settings,
                                                operation='execute')


def test_study_execution_rejects_a_partial_freeze(tmp_path):
    settings = run.config()
    selection = prepare_selection(settings)
    args = _args(tmp_path, case_offset=2, output=tmp_path/'freeze.json')
    freeze = run.create_freeze(args, settings, selection, synthetic=True)
    freeze['cases'].pop()
    with pytest.raises(ValueError, match='qualification_freeze_case_slice_mismatch'):
        evaluation_plan.validate_study_freeze(_args(tmp_path), settings, freeze)


def test_gate_rejects_self_reported_warmup_returned_identity(tmp_path):
    settings, _receipt_path, run_paths = _qualification_evidence(tmp_path)
    warmup_path = next(run_paths[0].parent.glob(run_paths[0].name + '.warmup-*.json'))
    warmup = json.loads(warmup_path.read_text())
    warmup['expected_returned_model'] = warmup['returned_model'] = 'self-reported-other-model'
    warmup_path.write_text(json.dumps(warmup))
    with pytest.raises(ValueError, match='qualification_warmup_failed_or_mismatched'):
        evaluation_plan._validate_evidence(
            run_paths, tmp_path/'qualification-dev4.json', settings)


def test_gate_rejects_truncated_preparation_trace(tmp_path):
    settings, _receipt_path, run_paths = _qualification_evidence(tmp_path)
    freeze_path = tmp_path/'qualification-dev4.json'
    freeze = json.loads(freeze_path.read_text())
    trace_path = tmp_path/freeze['preparation_trace_journal']
    lines = trace_path.read_text().splitlines()
    trace_path.write_text('\n'.join(lines[:-1])+'\n')
    freeze['preparation_trace_sha256'] = file_hash(trace_path)
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match='qualification_trace_incomplete'):
        evaluation_plan._validate_evidence(run_paths, freeze_path, settings)


def test_evaluation_gate_fails_before_controller_construction(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, 'OpenAIController', lambda **kwargs: calls.append(kwargs))
    args = _args(tmp_path, role='evaluation', study_phase='evaluation', limit=100,
                 repetitions=1, orders=['canonical'], qualification_receipt=tmp_path/'missing.json')
    with pytest.raises(ValueError, match='qualification_receipt_missing'):
        run.execute(args, run.config(), {'selection': {}})
    assert calls == []
