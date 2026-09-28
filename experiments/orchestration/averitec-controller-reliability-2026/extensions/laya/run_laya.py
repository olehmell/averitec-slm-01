#!/usr/bin/env python3
"""Bounded, append-only native Laya extension; no outcome resume/retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from importlib import metadata
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time

EXT = Path(__file__).resolve().parent
EXP = EXT.parents[1]
sys.path.insert(0, str(EXP))
import yaml
from data import ROOT, check_freeze, code_binding, file_hash, verify_preparation_traces, write_new
from engine import ACTIONS, INSTRUCTIONS, FaultTools, OracleController, ReplayTools, digest, replay_checkpoints, run_episode
from run import action_order, _append
from tracing import TraceRecorder
from warmup import WARMUP_OBSERVATION, premeasure_warmup
from adapter import BASE_MODEL_ID, TYPED_MODEL_ID, NativeLayaController

MODELS = {'base': BASE_MODEL_ID, 'typed': TYPED_MODEL_ID}


class BoundedController:
    """Never continue to another checkpoint after an infrastructure failure."""
    def __init__(self, delegate, deadline):
        self.delegate, self.deadline, self.model = delegate, deadline, delegate.model

    def choose(self, observation, instructions, actions):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('extension_phase_wall_budget')
        result = self.delegate.choose(observation, instructions, actions)
        if result.outcome not in ('ok', 'invalid_output'):
            raise RuntimeError('extension_infrastructure_failure')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('extension_phase_wall_budget')
        return result


def settings():
    return yaml.safe_load((EXT / 'config.yaml').read_text())


def extension_binding():
    paths = [p for p in EXT.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
             and p.suffix in ('.py', '.json', '.yaml', '.txt', '.sh', '.md')]
    return {str(p.relative_to(EXT)): file_hash(p) for p in sorted(paths)}


def require_committed():
    for relative in extension_binding():
        path = EXT / relative
        result = subprocess.run(['git', 'show', 'HEAD:' + str(path.relative_to(ROOT))],
                                cwd=ROOT, capture_output=True, check=False)
        if result.returncode or result.stdout != path.read_bytes():
            raise ValueError('extension_source_must_be_committed:' + relative)


def validate_freeze(freeze, path, phase, cfg):
    check_freeze(freeze, freeze['selection'])
    verify_preparation_traces(freeze, path.parent)
    if freeze['selection_sha256'] != cfg['selection_sha256']:
        raise ValueError('extension_selection_mismatch')
    role = 'evaluation' if phase == 'evaluation' else 'development'
    cases = freeze['cases']
    expected = ([r['case_id'] for r in freeze['selection']['cases'] if r['role'] == role]
                if phase == 'evaluation' else cfg['development_case_ids'])
    if ([r['case_id'] for r in cases] != expected or len(cases) != cfg['phases'][phase]['cases']
            or any(r['role'] != role for r in cases)):
        raise ValueError('extension_freeze_case_set_mismatch')


def plan(freeze, phase, checkpoint, cfg):
    rows = []
    policy = cfg['phases'][phase]
    for case in freeze['cases']:
        for scenario in cfg['scenarios']:
            for order in policy['orders']:
                repeats = policy['canonical_repetitions'] if order == 'canonical' else 1
                for repeat in range(repeats):
                    row = dict(controller='laya_' + checkpoint, model=MODELS[checkpoint],
                               case_id=case['case_id'], group_id=case['group_id'], role=case['role'],
                               scenario=scenario, order=order, repetition=repeat, phase=phase,
                               mode='checkpoints', prompt_variant='laya_native_v1', tools_mode='frozen')
                    row['trial_id'] = digest(row)
                    rows.append(row)
    return rows


def reference(case, scenario):
    return run_episode(OracleController(), FaultTools(ReplayTools(case['records']), scenario))


def runtime_identity():
    versions = {}
    for name in ('torch', 'transformers', 'safetensors', 'huggingface_hub', 'numpy', 'laya'):
        versions[name] = metadata.version(name)
    return {'distributions': versions, 'python': sys.version, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'image_sha256': os.environ.get('LAYA_IMAGE_SHA256')}


def score(rows):
    events = [event for row in rows for event in row['events']]
    if not events:
        raise ValueError('empty_screen')
    return {'decisions': len(events), 'compliant': sum(e['compliant'] for e in events),
            'decision_compliance': sum(e['compliant'] for e in events) / len(events),
            'valid_action_rate': sum(e['provider_outcome'] == 'ok' for e in events) / len(events),
            'median_native_inference_latency_ms': statistics.median(e['latency_ms'] for e in events)}


def select_checkpoint(screens):
    eligible = {name: item for name, item in screens.items() if item['status'] == 'complete'}
    if not eligible:
        raise ValueError('no_operational_screen')
    return min(eligible, key=lambda name: (-eligible[name]['score']['decision_compliance'],
               eligible[name]['score']['median_native_inference_latency_ms'], name))


def evidence_files(output):
    return {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p != output / 'receipt.json'}


def verify_receipt(path, phase, cfg):
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != 'laya-extension-receipt/v1' or receipt.get('phase') != phase
            or receipt.get('status') != 'complete' or receipt.get('extension_binding') != extension_binding()
            or receipt.get('core_binding') != code_binding()
            or receipt.get('selection_sha256') != cfg['selection_sha256']):
        raise ValueError('extension_receipt_identity')
    if receipt.get('evidence') != evidence_files(path.parent):
        raise ValueError('extension_receipt_evidence_drift')
    if receipt.get('receipt_sha256') != digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'}):
        raise ValueError('extension_receipt_digest')
    freeze_path = path.parent / 'freeze.json'
    freeze = json.loads(freeze_path.read_text())
    validate_freeze(freeze, freeze_path, phase, cfg)
    if receipt['freeze_sha256'] != digest(freeze):
        raise ValueError('extension_receipt_freeze_mismatch')
    expected_names = set(MODELS) if phase == 'screen' else {receipt['selected_checkpoint']}
    if set(receipt['conditions']) != expected_names:
        raise ValueError('extension_receipt_conditions')
    if any(c['status'] != 'complete' for c in receipt['conditions'].values()) and phase != 'screen':
        raise ValueError('extension_qualification_incomplete')
    if phase == 'screen' and receipt['selected_checkpoint'] != select_checkpoint(receipt['conditions']):
        raise ValueError('extension_selection_rule_mismatch')
    for name, condition in receipt['conditions'].items():
        if condition['status'] != 'complete':
            continue
        rows = audit_condition(path.parent / name, freeze, phase, name, cfg)
        if condition['score'] != score(rows):
            raise ValueError('extension_receipt_score_mismatch')
    return receipt


def audit_condition(output, freeze, phase, checkpoint, cfg):
    meta = json.loads((output / 'metadata.json').read_text())
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    tasks = meta['planned_tasks']
    if tasks != plan(freeze, phase, checkpoint, cfg) or meta['freeze_sha256'] != digest(freeze):
        raise ValueError('extension_plan_mismatch')
    if [r['trial_id'] for r in rows] != [t['trial_id'] for t in tasks]:
        raise ValueError('extension_trial_completeness')
    intents = [json.loads(line) for line in (output / 'intents.jsonl').read_text().splitlines()]
    if intents != tasks:
        raise ValueError('extension_intent_completeness')
    traces = [json.loads(line) for line in (output / 'traces.jsonl').read_text().splitlines()]
    starts = [r['span_id'] for r in traces if r['event'] == 'span_start']
    ends = [r['span_id'] for r in traces if r['event'] == 'span_end']
    if len(set(starts)) != len(starts) or sorted(starts) != sorted(ends):
        raise ValueError('extension_trace_completeness')
    roots = [r for r in traces if r['event'] == 'span_start' and r.get('parent_span_id') is None]
    if [r['input'] for r in roots] != tasks or any(r['name'] != 'controller.trial' for r in roots):
        raise ValueError('extension_trace_trial_mismatch')
    decisions = [r for r in traces if r['event'] == 'span_start' and r.get('name') == 'controller.decision']
    requests = [r for r in traces if r['event'] == 'span_start' and r.get('name') == 'controller.request']
    if len(decisions) != sum(len(r['events']) for r in rows) or len(requests) != len(decisions):
        raise ValueError('extension_trace_decision_count')
    by_id = {case['case_id']: case for case in freeze['cases']}
    for row, task in zip(rows, tasks):
        if row['row_sha256'] != digest({k: v for k, v in row.items() if k != 'row_sha256'}):
            raise ValueError('extension_row_digest')
        if any(row[k] != v for k, v in task.items()) or row['freeze_sha256'] != meta['freeze_sha256']:
            raise ValueError('extension_row_identity')
        if len(row['events']) != row['planned_decisions'] or not row['events']:
            raise ValueError('extension_decision_completeness')
        expected_events = reference(by_id[task['case_id']], task['scenario'])['events']
        if len(expected_events) != row['planned_decisions']:
            raise ValueError('extension_reference_length')
        for event, expected in zip(row['events'], expected_events):
            if any(event[k] != expected[k] for k in ('step', 'observation', 'observation_sha256', 'expected_action')):
                raise ValueError('extension_reference_mismatch')
            if event['provider_outcome'] not in ('ok', 'invalid_output'):
                raise ValueError('extension_infrastructure_failure')
            if event['model'] != task['model'] or event['returned_model'] != task['model']:
                raise ValueError('extension_model_identity')
            if event['compliant'] != (event['provider_outcome'] == 'ok' and event['action'] == event['expected_action']):
                raise ValueError('extension_compliance_mismatch')
    return rows


def run_condition(output, checkpoint, tasks, freeze, assets_root, asset_receipt, deadline):
    from assets import checkpoint_path
    if time.monotonic() >= deadline:
        raise TimeoutError('extension_phase_wall_budget')
    output.mkdir()
    controller = NativeLayaController(MODELS[checkpoint], checkpoint_path(assets_root, checkpoint), device='cuda')
    controller.timeout_seconds = 120
    by_id = {r['case_id']: r for r in freeze['cases']}
    references = {(t['case_id'], t['scenario']): reference(by_id[t['case_id']], t['scenario']) for t in tasks}
    try:
        # All full inputs/orders are checked before even the synthetic warmup.
        preflights = []
        seen = set()
        for task in tasks:
            actions = action_order(task['order'], settings()['seed'])
            for event in references[task['case_id'], task['scenario']]['events']:
                key = digest([event['observation'], actions])
                if key not in seen:
                    preflights.append({'input_sha256': key, **controller.preflight(event['observation'], actions)})
                    seen.add(key)
        preflights.append({'warmup': True, **controller.preflight(WARMUP_OBSERVATION, list(ACTIONS))})
        write_new(output / 'preflight.json', {'checks': preflights})
        meta = {'planned_tasks': tasks, 'freeze_sha256': digest(freeze), 'identity': controller.identity(),
                'asset_receipt': asset_receipt, 'runtime': runtime_identity(),
                'latency_boundary': 'synchronized_native_forward_no_network'}
        write_new(output / 'metadata.json', meta)
        run_id = digest(meta)
        warm_tracer = TraceRecorder(output / 'warmup.traces.jsonl', run_id=run_id, trial_id='warmup')
        with warm_tracer.span('controller.warmup', input={'checkpoint': checkpoint}) as span:
            warmup = premeasure_warmup(controller, instructions=INSTRUCTIONS, actions=list(ACTIONS), tracer=warm_tracer)
            span.update(output=warmup)
        write_new(output / 'warmup.json', warmup)
        bounded = BoundedController(controller, deadline)
        with (output / 'intents.jsonl').open('x') as intents, (output / 'results.jsonl').open('x') as results:
            for task in tasks:
                if time.monotonic() >= deadline:
                    raise TimeoutError('extension_phase_wall_budget')
                _append(intents, task)
                tracer = TraceRecorder(output / 'traces.jsonl', run_id=run_id, trial_id=task['trial_id'], metadata=task)
                controller.tracer = tracer
                ref = references[task['case_id'], task['scenario']]
                with tracer.span('controller.trial', input=task) as span:
                    row = replay_checkpoints(bounded, ref, actions=action_order(task['order'], settings()['seed']), tracer=tracer)
                    span.update(output=row)
                row.update(task)
                row.update(freeze_sha256=digest(freeze), planned_decisions=len(ref['events']),
                           preparation_trace_id=by_id[task['case_id']]['preparation_trace_id'])
                row['row_sha256'] = digest(row)
                _append(results, row)
                if time.monotonic() >= deadline:
                    raise TimeoutError('extension_phase_wall_budget')
        rows = audit_condition(output, freeze, tasks[0]['phase'], checkpoint, settings())
        return {'status': 'complete', 'score': score(rows), 'trials': len(rows)}
    finally:
        controller.close()


def execute(args):
    cfg = settings()
    freeze = json.loads(args.freeze.read_text())
    validate_freeze(freeze, args.freeze, args.phase, cfg)
    selection = None
    qualification = None
    if args.phase != 'screen':
        if args.selection_receipt is None:
            raise ValueError('selection_receipt_required')
        selection = verify_receipt(args.selection_receipt, 'screen', cfg)
    if args.phase == 'qualification' and selection['freeze_sha256'] != digest(freeze):
        raise ValueError('development_freeze_drift')
    if args.phase == 'evaluation':
        if args.qualification_receipt is None:
            raise ValueError('qualification_receipt_required')
        qualification = verify_receipt(args.qualification_receipt, 'qualification', cfg)
        if (qualification['selection_receipt_sha256'] != file_hash(args.selection_receipt)
                or qualification['selected_checkpoint'] != selection['selected_checkpoint']):
            raise ValueError('qualification_selection_mismatch')
    names = list(MODELS) if selection is None else [selection['selected_checkpoint']]
    tasks = {name: plan(freeze, args.phase, name, cfg) for name in names}
    planned = {'phase': args.phase, 'conditions': tasks, 'freeze_sha256': digest(freeze),
               'gpu_wall_seconds': cfg['phases'][args.phase]['gpu_wall_seconds']}
    if args.plan:
        print(json.dumps(planned, indent=2))
        return
    require_committed()
    from assets import verify_assets
    asset_receipt = verify_assets(args.assets_root)
    if selection is not None and asset_receipt != selection['asset_receipt']:
        raise ValueError('selected_assets_drift')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / 'plan.json', planned)
    write_new(args.output / 'freeze.json', freeze)
    shutil.copyfile(args.freeze.parent / freeze['preparation_trace_journal'],
                    args.output / freeze['preparation_trace_journal'])
    deadline = time.monotonic() + planned['gpu_wall_seconds']
    conditions = {}
    for name in names:
        try:
            conditions[name] = run_condition(args.output / name, name, tasks[name], freeze,
                                             args.assets_root, asset_receipt, deadline)
        except Exception as exc:
            # Error type/code only; never serialize local credentials or huge state.
            conditions[name] = {'status': 'failed', 'error_type': type(exc).__name__, 'error_code': str(exc)[:240]}
            write_new(args.output / (name + '-failure.json'), conditions[name])
            known_structural_failure = isinstance(exc, ValueError) and str(exc).startswith('laya_preflight_')
            if args.phase != 'screen' or not known_structural_failure:
                raise
    selected = select_checkpoint(conditions) if selection is None else selection['selected_checkpoint']
    receipt = {'schema': 'laya-extension-receipt/v1', 'status': 'complete', 'phase': args.phase,
               'created_utc': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': digest(freeze),
               'selection_sha256': cfg['selection_sha256'], 'selected_checkpoint': selected,
               'extension_binding': extension_binding(), 'core_binding': code_binding(),
               'asset_receipt': asset_receipt, 'conditions': conditions,
               'selection_receipt_sha256': file_hash(args.selection_receipt) if selection else None,
               'qualification_receipt_sha256': file_hash(args.qualification_receipt) if qualification else None,
               'evidence': evidence_files(args.output)}
    receipt['receipt_sha256'] = digest(receipt)
    write_new(args.output / 'receipt.json', receipt)
    print(json.dumps({'phase': args.phase, 'selected': selected, 'conditions': conditions}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('screen', 'qualification', 'evaluation'))
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-receipt', type=Path)
    parser.add_argument('--qualification-receipt', type=Path)
    parser.add_argument('--plan', action='store_true')
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
