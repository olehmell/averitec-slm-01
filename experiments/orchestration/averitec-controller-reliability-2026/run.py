#!/usr/bin/env python3
"""Repository-root CLI for the separate controller reliability study."""
from __future__ import annotations

import argparse
from copy import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import uuid

import yaml

from analysis import summarize
from data import (ROOT, EXPERIMENT, check_freeze, code_binding, file_hash, prepare_selection,
                  selected_cases, verify_preparation_traces, write_new)
from engine import (ACTIONS, SCENARIOS, INSTRUCTIONS, FaultTools, OracleController,
                    RecordingTools, ReplayTools, canonical, digest, replay_checkpoints, run_episode)
from providers import JevController, OpenAIController
from gemini_provider import GeminiController, gemini_profile_identity
from generation_profiles import generation_profile_identity
from prompt_variants import (VARIANTS, PresentedController, instructions_for,
                             native_choice_criteria, presentation_identity)
from warmup import WarmupFailure, premeasure_warmup
from tracing import TraceExportError, TraceRecorder, export_traces
from evaluation_plan import (create_qualification_receipt, enforce_study_phase,
                             validate_qualification_receipt, validate_study_freeze)


def config():
    return yaml.safe_load((EXPERIMENT / 'config.yaml').read_text())


def require_committed_config():
    relative = str((EXPERIMENT / 'config.yaml').relative_to(ROOT))
    result = subprocess.run(['git', 'show', 'HEAD:' + relative], cwd=ROOT, capture_output=True)
    if result.returncode or result.stdout != (EXPERIMENT / 'config.yaml').read_bytes():
        raise ValueError('commit_experiment_config_before_model_compute')


def load_key(name='TYPESAFE_API_KEY'):
    if name not in ('TYPESAFE_API_KEY', 'GEMINI_API_KEY'):
        raise ValueError('unsupported_api_key_name')
    if os.environ.get(name):
        return
    dotenv = ROOT / '.env'
    for line in dotenv.read_text().splitlines() if dotenv.is_file() else []:
        found_name, separator, value = line.strip().partition('=')
        if separator and found_name.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if not value:
                raise ValueError('missing_api_key')
            os.environ[name] = value
            return
    raise ValueError('missing_api_key')


def create_freeze(args, settings, selection, *, synthetic=False):
    enforce_study_phase(args, settings, operation='freeze')
    from workers import PipelineTools, SyntheticTools
    cases = selected_cases(selection, args.role, limit=args.limit,
                           offset=getattr(args, 'case_offset', 0))
    if not cases:
        raise ValueError('no_selected_cases')
    if not synthetic:
        if args.output.exists():
            raise ValueError('freeze_output_already_exists')
        require_committed_config()
        if not args.worker_endpoint:
            raise ValueError('worker_endpoint_required')
    results = []
    trace_path = Path(str(args.output) + '.preparation.traces.jsonl')
    preparation_id = uuid.uuid4().hex
    for case in cases:
        runtime_case = {key: case[key] for key in ('case_id', 'claim', 'split')}
        tracer = TraceRecorder(trace_path, run_id=preparation_id, trial_id=case['case_id'],
                               metadata={'experiment': settings['id'], 'phase': 'tool_preparation',
                                         'origin': 'synthetic' if synthetic else 'real_tools',
                                         'case_id': case['case_id'], 'group_id': case['group_id'],
                                         'selection_sha256': digest(selection)})
        with tracer.span('shared_worker.preparation', input=runtime_case) as span:
            tools = (SyntheticTools(runtime_case, tracer=tracer) if synthetic else
                     PipelineTools(runtime_case, endpoint=args.worker_endpoint,
                                   model=settings['models']['qwen']['model'],
                                   source_corpora=ROOT / settings['source_corpora'],
                                   exclusions=ROOT / settings['exclusions'], root=ROOT,
                                   timeout_seconds=settings['workflow']['tool_timeout_seconds'], tracer=tracer))
            recorder = RecordingTools(tools)
            reference = run_episode(OracleController(), recorder, tracer=tracer)
            receipt = tools.preparation_receipt()
            span.update(output={'outcome': reference['outcome'], 'worker_receipt': receipt})
        results.append({'case_id': case['case_id'], 'group_id': case['group_id'], 'role': args.role,
                        'preparation_trace_id': tracer.trace_id,
                        'preparation_outcome': reference['outcome'],
                        'worker_receipt': receipt, 'worker_receipt_sha256': digest(receipt),
                        'records': recorder.records, 'records_sha256': digest(recorder.records)})
    return {
        'schema': 'averitec-controller-tool-freeze/v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'origin': 'synthetic' if synthetic else 'real_tools',
        'selection_sha256': digest(selection), 'selection': selection,
        'code_binding': code_binding(), 'shared_worker': settings['models']['qwen'],
        'corpus_registry_sha256': file_hash(ROOT / settings['source_corpora']),
        'exclusions_sha256': file_hash(ROOT / settings['exclusions']),
        'preparation_trace_journal': trace_path.name,
        'preparation_trace_sha256': file_hash(trace_path),
        'cases': results,
    }


def plan_freeze(args, settings, selection):
    """Return the exact bounded freeze selection without constructing a worker."""
    enforce_study_phase(args, settings, operation='freeze')
    cases = selected_cases(selection, args.role, limit=args.limit,
                           offset=getattr(args, 'case_offset', 0))
    if not cases:
        raise ValueError('no_selected_cases')
    return {
        'phase': getattr(args, 'study_phase', None),
        'role': args.role,
        'case_offset': getattr(args, 'case_offset', 0),
        'case_limit': args.limit,
        'case_count': len(cases),
        'case_ids': [case['case_id'] for case in cases],
        'group_ids': [case['group_id'] for case in cases],
        'selection_sha256': digest(selection),
        'model_calls': 0,
        'ready_for_worker_calls': False,
    }


def action_order(name, seed):
    actions = list(ACTIONS)
    if name == 'reversed':
        actions.reverse()
    elif name == 'seeded':
        random.Random(seed).shuffle(actions)
    return actions


def planned_tasks(args, freeze):
    cases = [row for row in freeze['cases'] if row['role'] == args.role]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError('freeze_has_no_requested_role')
    tasks = []
    for case in cases:
        for scenario in args.scenarios:
            for order in args.orders:
                # Three canonical repeats; order perturbations each once, matched at repeat 0.
                for repetition in range(args.repetitions if order == 'canonical' else 1):
                    task = {'controller': args.controller, 'case_id': case['case_id'],
                            'group_id': case['group_id'], 'role': args.role,
                            'scenario': scenario, 'order': order, 'repetition': repetition,
                            'mode': args.mode, 'output_mode': 'prompt_only' if args.prompt_only else 'structured',
                            'prompt_variant': getattr(args, 'prompt_variant', 'v1'),
                            'generation_profile': getattr(args, 'generation_profile', 'baseline'),
                            'tools_mode': 'live' if args.live_tools else 'frozen'}
                    task['trial_id'] = digest(task)
                    tasks.append(task)
    return tasks


def _append(handle, row):
    handle.write(canonical(row) + '\n')
    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def run_lock(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    for row in rows:
        if row['row_sha256'] != digest({k: v for k, v in row.items() if k != 'row_sha256'}):
            raise ValueError('result_row_digest')
    if len({row['trial_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate_result_rows')
    return rows


def read_complete_run(path):
    rows = read_rows(path)
    identity = json.loads(Path(str(path) + '.run.json').read_text())
    expected = {task['trial_id']: task for task in identity['planned_tasks']}
    if {row['trial_id'] for row in rows} != set(expected):
        raise ValueError('incomplete_run_resume_before_summary')
    for row in rows:
        if any(row.get(key) != value for key, value in expected[row['trial_id']].items()):
            raise ValueError('result_task_identity_mismatch')
        if row['freeze_sha256'] != identity['freeze_sha256'] or row['origin'] != identity['origin']:
            raise ValueError('result_run_identity_mismatch')
    return rows


def execute(args, settings, freeze, *, synthetic=False):
    enforce_study_phase(args, settings, operation='execute')
    check_freeze(freeze, freeze['selection'], allow_synthetic=synthetic)
    validate_study_freeze(args, settings, freeze)
    verify_preparation_traces(freeze, args.output.parent if synthetic else args.freeze.parent)
    tasks = planned_tasks(args, freeze)
    if getattr(args, 'study_phase', None):
        expected_trials = 60 if args.study_phase == 'qualification' else 300
        maximum = settings['evaluation_run'][f'max_{args.study_phase}_decisions']
        if len(tasks) != expected_trials or len(tasks) * 12 * 5 != maximum:
            raise ValueError(f'{args.study_phase}_trial_or_decision_budget_mismatch')
    order_seed = settings['seed']
    variant = getattr(args, 'prompt_variant', 'v1')
    profile = getattr(args, 'generation_profile', 'baseline')
    presentation = presentation_identity(variant)
    if synthetic:
        generation = {'name': 'baseline', 'model': 'deterministic_oracle'}
    elif args.controller == 'jev':
        if profile not in ('baseline', 'jev_native'):
            raise ValueError('generation_profile_model_incompatible')
        criteria = native_choice_criteria(variant) if profile == 'jev_native' else None
        generation = {'name': profile, 'model': settings['models']['jev']['model'],
                      'choice_criteria': criteria,
                      'timeout_seconds': settings['workflow']['decision_timeout_seconds']}
    elif args.controller == 'gemini':
        if args.prompt_only:
            raise ValueError('gemini_requires_native_structured_output')
        model_config = settings['models']['gemini']
        generation = gemini_profile_identity(profile, model_config['model'], model_config['expected_version'])
        if generation['timeout_seconds'] is None:
            generation['timeout_seconds'] = settings['workflow']['decision_timeout_seconds']
    else:
        generation = generation_profile_identity(profile, settings['models'][args.controller]['model'])
        if generation['timeout_seconds'] is None:
            generation['timeout_seconds'] = settings['workflow']['decision_timeout_seconds']
    if (not synthetic and args.role == 'evaluation'
            and getattr(args, 'study_phase', None) != 'evaluation'):
        raise ValueError('evaluation_requires_explicit_study_phase')
    identity = {'schema': 'averitec-controller-run/v1', 'freeze_sha256': digest(freeze),
                'config_sha256': file_hash(EXPERIMENT / 'config.yaml'),
                **presentation, 'generation_profile': profile, 'generation_settings': generation,
                'warmup_policy': 'one_synthetic_request_before_pending_measurements' if not synthetic else 'not_applicable',
                'code_binding': code_binding(),
                'model': 'deterministic_oracle' if synthetic else settings['models'][args.controller],
                'output_mode': 'structured' if not args.prompt_only else 'prompt_only',
                'origin': ('synthetic' if synthetic else 'real_controller_live_tools' if args.live_tools else 'real_controller_frozen_tools'),
                'planned_tasks': tasks, 'python': platform.python_version(),
                'endpoint_location': {'jev': 'typesafe_api', 'gemini': 'gemini_api'}.get(args.controller, 'local_or_explicit_endpoint')}
    if args.plan:
        by_id = {row['case_id']: row for row in freeze['cases']}
        planned = sum(len(run_episode(OracleController(), FaultTools(
            ReplayTools(by_id[task['case_id']]['records']), task['scenario']))['events'])
                      for task in tasks) if args.mode == 'checkpoints' else None
        print(json.dumps({'trials': len(tasks), 'planned_decision_calls': planned,
                          'max_decision_calls': len(tasks)*12, **presentation,
                          'generation_profile': profile,
                          'origin': identity['origin'], 'ready_for_model_calls': False,
                          'requires': 'committed config and reachable pinned endpoint'}, indent=2))
        return []
    identity['run_id'] = digest({'identity': identity, 'output_namespace': str(args.output.resolve())})
    if not synthetic:
        require_committed_config()
        if args.controller == 'jev':
            if args.prompt_only:
                raise ValueError('jev_has_native_choices_only')
            load_key()
            controller = JevController(model=settings['models']['jev']['model'],
                                       timeout_seconds=settings['workflow']['decision_timeout_seconds'],
                                       choice_criteria=criteria)
        elif args.controller == 'gemini':
            load_key('GEMINI_API_KEY')
            controller = GeminiController(model=model_config['model'],
                                          expected_version=model_config['expected_version'],
                                          seed=order_seed, generation_profile=profile,
                                          timeout_seconds=settings['workflow']['decision_timeout_seconds'])
        else:
            if not args.endpoint:
                raise ValueError('controller_endpoint_required')
            controller = OpenAIController(endpoint=args.endpoint,
                                          model=settings['models'][args.controller]['model'],
                                          seed=order_seed, structured=not args.prompt_only,
                                          generation_profile=profile,
                                          timeout_seconds=settings['workflow']['decision_timeout_seconds'])
        controller = PresentedController(controller, variant)
    else:
        controller = OracleController()
    if args.live_tools and (args.mode != 'trajectory' or not args.worker_endpoint):
        raise ValueError('live_tools_require_trajectory_and_worker_endpoint')
    runtime_cases = {case['case_id']: {k: case[k] for k in ('case_id','claim','split')}
                     for case in selected_cases(freeze['selection'], args.role)} if args.live_tools else {}
    output = args.output
    metadata_path = Path(str(output) + '.run.json')
    intent_path = Path(str(output) + '.intents.jsonl')
    by_id = {row['case_id']: row for row in freeze['cases']}
    with run_lock(output):
        if metadata_path.exists():
            if json.loads(metadata_path.read_text()) != identity:
                raise ValueError('resume_identity_mismatch')
        else:
            if output.exists() or intent_path.exists():
                raise ValueError('orphan_run_files')
            write_new(metadata_path, identity)
        rows = read_rows(output)
        planned_ids = {task['trial_id'] for task in tasks}
        if any(row['trial_id'] not in planned_ids for row in rows):
            raise ValueError('unexpected_result_trial')
        done = {row['trial_id'] for row in rows}
        intent_records = [json.loads(line) for line in intent_path.read_text().splitlines()] if intent_path.exists() else []
        intents = [record['trial_id'] for record in intent_records]
        intent_trace_ids = {record['trial_id']: record['trace_id'] for record in intent_records}
        if len(set(intents)) != len(intents) or not set(intents) <= planned_ids:
            raise ValueError('invalid_intent_journal')
        # Completed or intent-only interrupted trials never trigger warmup or
        # additional model calls on resume. Warmup is deployment telemetry only.
        if not synthetic and any(task['trial_id'] not in done and task['trial_id'] not in intents for task in tasks):
            warmup_id = uuid.uuid4().hex
            warm_tracer = TraceRecorder(Path(str(output) + '.warmup.traces.jsonl'),
                                       run_id=identity['run_id'], trial_id='warmup_' + warmup_id,
                                       metadata={'origin': 'synthetic_deployment_check',
                                                 'excluded_from_measurements': True,
                                                 **presentation, 'generation_profile': profile})
            try:
                with warm_tracer.span('controller.warmup', input={'dataset_used': False}) as warm_span:
                    warm_receipt = premeasure_warmup(controller, instructions=instructions_for(variant),
                                                    actions=list(ACTIONS), tracer=warm_tracer)
                    warm_span.update(output=warm_receipt)
            except WarmupFailure as failure:
                write_new(Path(str(output) + '.warmup-' + warmup_id + '.json'),
                          {'run_id': identity['run_id'], **presentation, 'generation_profile': profile,
                           'generation_settings': generation, **failure.receipt.as_dict()})
                raise ValueError('controller_warmup_failed') from None
            write_new(Path(str(output) + '.warmup-' + warmup_id + '.json'),
                      {'run_id': identity['run_id'], **presentation, 'generation_profile': profile,
                       'generation_settings': generation, **warm_receipt})
        with output.open('a', encoding='utf8') as target, intent_path.open('a', encoding='utf8') as journal:
            for task in tasks:
                if task['trial_id'] in done:
                    continue
                case = by_id[task['case_id']]
                tracer = TraceRecorder(Path(str(output) + '.traces.jsonl'), run_id=identity['run_id'],
                                       trial_id=task['trial_id'], metadata={
                                           **task, 'experiment': settings['id'],
                                           'origin': identity['origin'],
                                           'freeze_sha256': identity['freeze_sha256'],
                                           'preparation_trace_id': case['preparation_trace_id'],
                                           'instructions_sha256': identity['instructions_sha256']})
                reference = run_episode(OracleController(), FaultTools(ReplayTools(case['records']), task['scenario']))
                if task['trial_id'] in intents:
                    if intent_trace_ids[task['trial_id']] != tracer.trace_id:
                        raise ValueError('intent_trace_identity_mismatch')
                    result = {'outcome': 'interrupted_unknown_outcome', 'events': [],
                              'full_pipeline_completed': False, 'protocol_correct_termination': False,
                              'controller_calls': None}
                else:
                    _append(journal, {'trial_id': task['trial_id'], 'trace_id': tracer.trace_id})
                    with tracer.span('controller.trial', input=task) as span:
                        controller.tracer = tracer
                        if args.live_tools:
                            from workers import PipelineTools
                            raw_tools = PipelineTools(runtime_cases[case['case_id']], endpoint=args.worker_endpoint,
                                                      model=settings['models']['qwen']['model'],
                                                      source_corpora=ROOT/settings['source_corpora'],
                                                      exclusions=ROOT/settings['exclusions'],root=ROOT,
                                                      timeout_seconds=settings['workflow']['tool_timeout_seconds'],
                                                      tracer=tracer)
                        else:
                            raw_tools = ReplayTools(case['records'])
                        tools = FaultTools(raw_tools, task['scenario'])
                        actions = action_order(task['order'], order_seed)
                        if args.mode == 'trajectory':
                            result = run_episode(controller, tools, actions=actions, tracer=tracer)
                        else:
                            result = replay_checkpoints(controller, reference, actions=actions, tracer=tracer)
                        if args.live_tools:
                            result['worker_receipt'] = raw_tools.preparation_receipt()
                        span.update(output=result)
                row = {**task, **result, 'origin': identity['origin'],
                       'trace_id': tracer.trace_id, 'preparation_trace_id': case['preparation_trace_id'],
                       'freeze_sha256': identity['freeze_sha256'],
                       'shared_worker_preparation_outcome': case['preparation_outcome'],
                       'planned_decisions': len(reference['events']) if args.mode == 'checkpoints' else None}
                row['row_sha256'] = digest(row)
                _append(target, row)
                rows.append(row)
    return rows


def development_sweep(args, settings, freeze):
    """Finite screening matrix, not an optimizer or permission to enter evaluation."""
    if args.role != 'development' or args.live_tools or args.prompt_only:
        raise ValueError('sweep_requires_structured_frozen_development')
    expansion = getattr(args, 'expansion_screen', False)
    tuning = settings['expansion_screening'] if expansion else settings['development_tuning']
    if args.mode != 'checkpoints':
        raise ValueError('screening_requires_checkpoints')
    if args.limit is None or args.limit > tuning['screening_case_limit']:
        raise ValueError('sweep_exceeds_screening_case_limit')
    if expansion and (args.limit != 2 or len([c for c in freeze['cases'] if c['role'] == 'development']) < 2):
        raise ValueError('expansion_requires_two_development_cases')
    if args.repetitions != 1 or args.orders != ['canonical']:
        raise ValueError('screening_requires_single_canonical_pass')
    if args.scenarios != tuning['screening_scenarios']:
        raise ValueError('screening_requires_all_predeclared_scenarios')
    if expansion:
        variants = expansion_conditions(settings)[args.controller]
    else:
        if args.controller not in tuning['native_profiles']:
            raise ValueError('controller_requires_expansion_screen')
        variants = [(v, 'baseline') for v in tuning['prompt_variants']]
        for profile in tuning['native_profiles'][args.controller]:
            variants.extend((v, profile) for v in tuning['native_prompt_variants'])
    all_rows = []
    for variant, profile in variants:
        condition = copy(args)
        condition.prompt_variant, condition.generation_profile = variant, profile
        condition.output = args.output / f'{args.controller}-{variant}-{profile}.jsonl' if args.output else None
        print(json.dumps({'screening_condition': f'{args.controller}/{variant}/{profile}',
                          'plan_only': args.plan}), flush=True)
        all_rows.extend(execute(condition, settings, freeze))
    if not args.plan:
        summary_path = args.output / (args.controller + '-screening-summary.json')
        if summary_path.exists():
            if json.loads(summary_path.read_text()) != summarize(all_rows):
                raise ValueError('screening_summary_drift')
        else:
            write_new(summary_path, summarize(all_rows))
        print(json.dumps({'screening_completed': args.controller, 'conditions': len(variants),
                          'trials': len(all_rows), 'summary': str(summary_path)}))
    return all_rows


def expansion_conditions(settings):
    """Validate the finite approved matrix before any condition can run."""
    tuning = settings['expansion_screening']
    expected = {
        'jev': [('v2', 'baseline'), ('v3', 'baseline')],
        'qwen': [('v2', 'baseline'), ('v3', 'baseline')],
        'lfm': [('v2', 'lfm_native'), ('v3', 'lfm_native'), ('v4', 'lfm_native')],
        'lfm26': [('v2', 'lfm26_native'), ('v3', 'lfm26_native'), ('v4', 'lfm26_native')],
        'gemini': [('v2', 'gemini_native'), ('v3', 'gemini_native')],
    }
    actual = {key: [tuple(pair) for pair in pairs] for key, pairs in tuning['conditions'].items()}
    if actual != expected or tuning['max_conditions'] != 12 or tuning['max_warmup_calls'] != 12:
        raise ValueError('expansion_matrix_not_predeclared')
    if (tuning['screening_case_limit'] != 2 or tuning['screening_repetitions'] != 1
            or tuning['screening_orders'] != ['canonical']
            or tuning['screening_scenarios'] != list(SCENARIOS)
            or tuning['max_measured_decision_calls'] != 864
            or tuning['fresh_shared_tool_freeze_required'] is not True):
        raise ValueError('expansion_budget_not_predeclared')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--synthetic-smoke', action='store_true')
    mode.add_argument('--trace-smoke', action='store_true', help='Create a two-span synthetic tracing fixture, without network or model calls')
    mode.add_argument('--freeze-tools', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--development-sweep', action='store_true')
    mode.add_argument('--expansion-screen', action='store_true', help='Bounded five-controller development matrix')
    mode.add_argument('--qualify-runs', nargs='+', type=Path,
                      help='Validate five complete qualification runs and create a portable receipt')
    mode.add_argument('--summarize', nargs='+', type=Path)
    mode.add_argument('--export-traces', type=Path, help='Export an existing local trace journal to Langfuse; no model calls')
    parser.add_argument('--selection', type=Path, default=EXPERIMENT/'manifests/selection.json')
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--controller', choices=('jev','lfm','qwen','lfm26','gemini'), default='jev')
    parser.add_argument('--endpoint')
    parser.add_argument('--worker-endpoint')
    parser.add_argument('--role', choices=('development','evaluation'), default='development')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--case-offset', type=int, default=0)
    parser.add_argument('--study-phase', choices=('qualification', 'evaluation'))
    parser.add_argument('--qualification-receipt', type=Path)
    parser.add_argument('--mode', choices=('checkpoints','trajectory'), default='checkpoints')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--orders', nargs='+', choices=('canonical','reversed','seeded'), default=['canonical'])
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--prompt-only', action='store_true')
    parser.add_argument('--prompt-variant', choices=VARIANTS, default='v1')
    parser.add_argument('--generation-profile', choices=('baseline','jev_native','lfm_native','qwen_native','qwen_thinking','lfm26_native','gemini_native'), default='baseline')
    parser.add_argument('--live-tools', action='store_true', help='Actual shared-worker execution; trajectory mode only')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.repetitions < 1 or (args.limit is not None and args.limit < 1) or args.case_offset < 0:
        parser.error('limit/repetitions must be positive and case-offset nonnegative')
    if len(set(args.orders)) != len(args.orders) or len(set(args.scenarios)) != len(args.scenarios):
        parser.error('duplicate orders/scenarios')
    settings = config()
    if args.trace_smoke:
        if not args.output or args.output.exists():
            parser.error('--trace-smoke requires a new --output journal path')
        tracer = TraceRecorder(args.output, run_id=uuid.uuid4().hex, trial_id='observability_smoke',
                               metadata={'experiment': settings['id'], 'origin': 'synthetic',
                                         'purpose': 'langfuse_connectivity_smoke', 'model_calls': 0})
        with tracer.span('observability.smoke', input={'fixture': 'no_dataset_content'}) as root:
            with tracer.span('synthetic.generation', kind='generation', model='deterministic-smoke-fixture',
                             input={'prompt': 'Synthetic telemetry check; no model request.'}) as generation:
                generation.update(output={'completion': 'synthetic_ok'},
                                  usage={'input_tokens': 0, 'output_tokens': 0})
            root.update(output={'status': 'synthetic_ok', 'model_calls': 0})
        print(json.dumps({'trace_id': tracer.trace_id, 'journal': str(args.output), 'model_calls': 0}))
        return
    if args.export_traces:
        if not args.output or args.output.exists():
            parser.error('--export-traces requires a new --output receipt path')
        receipt = export_traces(args.export_traces, dotenv_path=ROOT/'.env')
        write_new(args.output, receipt)
        print(json.dumps(receipt, indent=2))
        return
    if args.check:
        selection = prepare_selection(settings)
        assert settings['workflow']['stages'] == list(ACTIONS[:-2])
        assert settings['workflow']['max_controller_calls'] == 12
        assert settings['workflow']['max_tool_retries_per_stage'] == 1
        assert file_hash(ROOT/settings['model_assets_manifest']) == settings['model_assets_manifest_sha256']
        expansion_conditions(settings)
        print(json.dumps({'protocol_valid': True, 'selected_cases': len(selection['cases']),
                          'models': list(settings['models']), 'model_calls': 0,
                          'vega_readiness': 'not_remotely_verified', 'quality_metrics': []}, indent=2))
        return
    if args.prepare:
        if not args.output:
            parser.error('--prepare requires --output')
        write_new(args.output, prepare_selection(settings))
        print(str(args.output))
        return
    if args.qualify_runs:
        if not args.output or not args.freeze:
            parser.error('--qualify-runs requires --freeze and a new --output receipt path')
        if args.output.exists():
            parser.error('--qualify-runs requires a new --output receipt path')
        receipt = create_qualification_receipt(args.qualify_runs, args.freeze, args.output, settings)
        write_new(args.output, receipt)
        validate_qualification_receipt(args.output, settings)
        print(json.dumps({'qualification_passed': True, 'runs': len(receipt['runs']),
                          'output': str(args.output)}))
        return
    if args.synthetic_smoke:
        args.controller = 'oracle'
        args.limit, args.repetitions = 2, 1
        args.orders, args.scenarios = ['canonical','reversed','seeded'], list(SCENARIOS)
        target = Path(tempfile.mkdtemp(prefix='averitec-controller-smoke-'))
        args.output = target/'synthetic-freeze.json'
        selection = prepare_selection(settings)
        freeze = create_freeze(args, settings, selection, synthetic=True)
        write_new(target/'synthetic-freeze.json', freeze)
        all_rows = []
        for run_mode in ('checkpoints', 'trajectory'):
            args.mode = run_mode
            args.output = target/(run_mode+'.jsonl')
            all_rows.extend(execute(args, settings, freeze, synthetic=True))
        assert all(row['outcome'] in ('checkpoints_evaluated','completed','correct_abort') for row in all_rows)
        assert all(e['compliant'] for row in all_rows for e in row['events'])
        write_new(target/'summary.json', summarize(all_rows))
        print(json.dumps({'synthetic_trials': len(all_rows), 'model_calls': 0,
                          'passed': True, 'output_directory': str(target)}, indent=2))
        return
    if not args.output and not args.plan:
        parser.error('--output is required')
    if args.freeze_tools:
        selection = json.loads(args.selection.read_text())
        if args.plan:
            print(json.dumps(plan_freeze(args, settings, selection), indent=2))
            return
        freeze = create_freeze(args, settings, selection)
        write_new(args.output, freeze)
        check_freeze(freeze, selection)
        verify_preparation_traces(freeze, args.output.parent)
        print(json.dumps({'frozen_cases': len(freeze['cases']), 'output': str(args.output)}))
    elif args.execute or args.development_sweep or args.expansion_screen:
        if not args.freeze:
            parser.error('--execute requires --freeze')
        runner = development_sweep if (args.development_sweep or args.expansion_screen) else execute
        rows = runner(args, settings, json.loads(args.freeze.read_text()))
        if not args.plan:
            print(json.dumps({'completed_trial_records': len(rows), 'output': str(args.output)}))
    else:
        rows = [row for path in args.summarize for row in read_complete_run(path)]
        if not rows or len({row['trial_id'] for row in rows}) != len(rows):
            raise ValueError('empty_or_duplicate_summary_trials')
        if len({row['freeze_sha256'] for row in rows}) != 1 or len({row['origin'] for row in rows}) != 1:
            raise ValueError('unmatched_freezes_or_origins')
        write_new(args.output, summarize(rows))
        print(str(args.output))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, TraceExportError) as error:
        # Deliberately omit arbitrary exception text (provider keys, claims, URLs).
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                          'reason': str(error) if isinstance(error, (ValueError, TraceExportError)) and str(error).replace('_','').isalnum() else 'input_or_runtime_error'}))
        raise SystemExit(2) from None
