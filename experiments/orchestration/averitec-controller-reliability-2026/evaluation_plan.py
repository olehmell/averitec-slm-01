"""Fixed qualification/evaluation contract and portable evidence gate."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from analysis import summarize
from data import (EXPERIMENT, check_freeze, code_binding, file_hash, prepare_selection,
                  verify_preparation_traces)
from engine import (ACTIONS, SCENARIOS, FaultTools, OracleController, ReplayTools,
                    digest, run_episode)
from gemini_provider import gemini_profile_identity
from generation_profiles import generation_profile_identity
from prompt_variants import native_choice_criteria, presentation_identity


CONDITIONS = {
    'jev': ('v2', 'baseline'),
    'qwen': ('v2', 'baseline'),
    'lfm': ('v3', 'lfm_native'),
    'lfm26': ('v3', 'lfm26_native'),
    'gemini': ('v2', 'gemini_native'),
}
QUALIFICATION_ORDERS = ['canonical', 'reversed', 'seeded']
RECEIPT_SCHEMA = 'averitec-controller-qualification-receipt/v1'


def _evaluation_config(settings: dict) -> dict:
    expected = {
        'purpose': 'fixed_five_controller_checkpoint_evaluation_without_adaptive_tuning',
        'qualification_case_offset': 2,
        'qualification_case_limit': 4,
        'qualification_repetitions': 3,
        'qualification_orders': QUALIFICATION_ORDERS,
        'evaluation_case_limit': 100,
        'evaluation_repetitions': 1,
        'evaluation_orders': ['canonical'],
        'scenarios': list(SCENARIOS),
        'conditions': {name: {'prompt_variant': pair[0], 'generation_profile': pair[1]}
                       for name, pair in CONDITIONS.items()},
        'max_qualification_decisions': 3600,
        'max_evaluation_decisions': 18000,
        'qualification_gate': 'complete_matched_runs_valid_warmups_no_interruption_or_transport_identity_failures',
        'accuracy_threshold': None,
        'semantic_failure_retries': 0,
        'qualification_warmup_calls': 5,
        'evaluation_warmup_calls': 5,
        'limits': {
            'qualification_freeze_wall_seconds': 3600,
            'qualification_lfm_wall_seconds': 900,
            'qualification_qwen_wall_seconds': 900,
            'qualification_lfm26_wall_seconds': 1800,
            'evaluation_freeze_wall_seconds': 21600,
            'evaluation_lfm_wall_seconds': 1800,
            'evaluation_qwen_wall_seconds': 1800,
            'evaluation_lfm26_wall_seconds': 10800,
            'maximum_allocated_gpu_seconds': 43200,
            'api_phase_wall_seconds': 14400,
        },
        'previous_screening_cases_excluded_from_qualification': True,
        'main_study_holdout_access': 'prohibited',
        'adaptive_tuning': False,
    }
    plan = settings.get('evaluation_run')
    if not isinstance(plan, dict):
        raise ValueError('evaluation_run_config_missing')
    if str(plan.get('approved_date')) != '2026-09-19':
        raise ValueError('evaluation_run_config_not_approved')
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError('evaluation_run_config_not_approved')
    if settings.get('development_tuning', {}).get('evaluation_settings_frozen') is not True:
        raise ValueError('evaluation_settings_not_frozen')
    return plan


def enforce_study_phase(args, settings: dict, *, operation: str) -> None:
    """Reject an expanded or implicit study matrix before any model/tool call."""
    phase = getattr(args, 'study_phase', None)
    if phase is None:
        if getattr(args, 'role', None) == 'evaluation':
            raise ValueError('evaluation_requires_explicit_study_phase')
        return
    plan = _evaluation_config(settings)
    if getattr(args, 'prompt_only', False) or getattr(args, 'live_tools', False):
        raise ValueError('study_phase_requires_structured_frozen_tools')
    if getattr(args, 'mode', 'checkpoints') != 'checkpoints':
        raise ValueError('study_phase_requires_checkpoints')
    offset = getattr(args, 'case_offset', 0)
    if phase == 'qualification':
        if getattr(args, 'role', None) != 'development':
            raise ValueError('qualification_requires_development_role')
        expected_offset = plan['qualification_case_offset'] if operation == 'freeze' else 0
        if offset != expected_offset or getattr(args, 'limit', None) != plan['qualification_case_limit']:
            raise ValueError('qualification_requires_approved_case_slice')
        if operation == 'execute':
            _require_execution_settings(args, plan, phase)
    elif phase == 'evaluation':
        if getattr(args, 'role', None) != 'evaluation':
            raise ValueError('evaluation_requires_evaluation_role')
        if offset != 0 or getattr(args, 'limit', None) != plan['evaluation_case_limit']:
            raise ValueError('evaluation_requires_approved_case_slice')
        receipt = getattr(args, 'qualification_receipt', None)
        if receipt is None:
            raise ValueError('evaluation_requires_qualification_receipt')
        # This is deliberately before worker/controller construction.
        validate_qualification_receipt(receipt, settings)
        if operation == 'execute':
            _require_execution_settings(args, plan, phase)
    else:  # pragma: no cover - argparse owns the public vocabulary
        raise ValueError('unknown_study_phase')


def _require_execution_settings(args, plan: dict, phase: str) -> None:
    repetitions = plan[f'{phase}_repetitions']
    orders = plan[f'{phase}_orders']
    if (getattr(args, 'repetitions', None) != repetitions
            or getattr(args, 'orders', None) != orders
            or getattr(args, 'scenarios', None) != plan['scenarios']):
        raise ValueError(f'{phase}_requires_approved_execution_matrix')
    condition = CONDITIONS.get(getattr(args, 'controller', None))
    if condition is None or (getattr(args, 'prompt_variant', None),
                             getattr(args, 'generation_profile', None)) != condition:
        raise ValueError(f'{phase}_requires_approved_controller_profile')


def validate_study_freeze(args, settings: dict, freeze: dict) -> None:
    """Require the exact canonical phase cases, not merely a compatible role."""
    phase = getattr(args, 'study_phase', None)
    if phase is None:
        return
    selection = prepare_selection(settings)
    if freeze.get('selection') != selection:
        raise ValueError(f'{phase}_freeze_selection_mismatch')
    role = 'development' if phase == 'qualification' else 'evaluation'
    offset = 2 if phase == 'qualification' else 0
    limit = 4 if phase == 'qualification' else 100
    expected = [row['case_id'] for row in selection['cases'] if row['role'] == role][offset:offset + limit]
    actual = [row.get('case_id') for row in freeze.get('cases', [])]
    if actual != expected or any(row.get('role') != role for row in freeze.get('cases', [])):
        raise ValueError(f'{phase}_freeze_case_slice_mismatch')


def _contained_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        raise ValueError('qualification_evidence_outside_receipt_directory') from None


def _reference(path: Path, root: Path) -> dict:
    if not path.is_file():
        raise ValueError('qualification_evidence_file_missing')
    return {'path': _contained_relative(path, root), 'sha256': file_hash(path)}


def _resolve_reference(reference: dict, root: Path) -> Path:
    if set(reference) != {'path', 'sha256'} or not isinstance(reference['path'], str):
        raise ValueError('qualification_evidence_reference_schema')
    path = root / reference['path']
    if _contained_relative(path, root) != reference['path']:
        raise ValueError('qualification_evidence_unsafe_path')
    if not path.is_file() or file_hash(path) != reference['sha256']:
        raise ValueError('qualification_evidence_hash_mismatch')
    return path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_spans(path: Path) -> tuple[dict[str, dict], dict[tuple[str, str], tuple[dict, dict]]]:
    rows = _jsonl(path)
    starts = {(row.get('trace_id'), row.get('span_id')): row for row in rows
              if row.get('event') == 'span_start'}
    ends = {(row.get('trace_id'), row.get('span_id')): row for row in rows
            if row.get('event') == 'span_end'}
    roots = {row['trace_id']: row for row in starts.values() if row.get('parent_span_id') is None}
    if not roots:
        raise ValueError('qualification_trace_roots_missing')
    if len(starts) != sum(row.get('event') == 'span_start' for row in rows):
        raise ValueError('qualification_trace_duplicate_span')
    if len(ends) != sum(row.get('event') == 'span_end' for row in rows):
        raise ValueError('qualification_trace_duplicate_span')
    if set(starts) != set(ends):
        raise ValueError('qualification_trace_incomplete')
    return roots, {key: (starts[key], ends[key]) for key in starts}


def _trace_id(run_id: str, trial_id: str) -> str:
    return hashlib.sha256(f'langfuse-trace:{run_id}:{trial_id}'.encode()).hexdigest()[:32]


def _expected_generation(settings: dict, controller: str, variant: str, profile: str) -> dict:
    if controller == 'jev':
        return {'name': profile, 'model': settings['models']['jev']['model'],
                'choice_criteria': native_choice_criteria(variant) if profile == 'jev_native' else None,
                'timeout_seconds': settings['workflow']['decision_timeout_seconds']}
    if controller == 'gemini':
        model = settings['models']['gemini']
        generation = gemini_profile_identity(profile, model['model'], model['expected_version'])
    else:
        generation = generation_profile_identity(profile, settings['models'][controller]['model'])
    if generation['timeout_seconds'] is None:
        generation['timeout_seconds'] = settings['workflow']['decision_timeout_seconds']
    return generation


def _expected_qualification_tasks(selection: dict, controller: str) -> list[dict]:
    variant, profile = CONDITIONS[controller]
    cases = [row for row in selection['cases'] if row['role'] == 'development'][2:6]
    tasks = []
    for case in cases:
        for scenario in SCENARIOS:
            for order in QUALIFICATION_ORDERS:
                for repetition in range(3 if order == 'canonical' else 1):
                    task = {
                        'controller': controller, 'case_id': case['case_id'],
                        'group_id': case['group_id'], 'role': 'development',
                        'scenario': scenario, 'order': order, 'repetition': repetition,
                        'mode': 'checkpoints', 'output_mode': 'structured',
                        'prompt_variant': variant, 'generation_profile': profile,
                        'tools_mode': 'frozen',
                    }
                    task['trial_id'] = digest(task)
                    tasks.append(task)
    return tasks


def _validate_measured_row(row: dict, task: dict, case: dict, model: str) -> dict:
    reference = run_episode(OracleController(), FaultTools(ReplayTools(case['records']), task['scenario']))
    expected_events = reference['events']
    if (row.get('outcome') != 'checkpoints_evaluated'
            or row.get('preparation_trace_id') != case['preparation_trace_id']
            or row.get('shared_worker_preparation_outcome') != case['preparation_outcome']
            or row.get('planned_decisions') != len(expected_events)
            or row.get('controller_calls') != len(expected_events)
            or len(row.get('events', [])) != len(expected_events)
            or row.get('full_pipeline_completed') is not None
            or row.get('protocol_correct_termination') is not None):
        raise ValueError('qualification_checkpoint_result_incomplete')
    for event, expected in zip(row['events'], expected_events):
        identity = {
            'step': expected['step'], 'observation': expected['observation'],
            'observation_sha256': expected['observation_sha256'],
            'expected_action': expected['expected_action'],
        }
        if any(event.get(key) != value for key, value in identity.items()):
            raise ValueError('qualification_checkpoint_identity_mismatch')
        outcome, action = event.get('provider_outcome'), event.get('action')
        if outcome not in ('ok', 'invalid_output', 'transport_error', 'version_mismatch'):
            raise ValueError('qualification_provider_outcome_invalid')
        if event.get('model') != model:
            raise ValueError('qualification_event_model_mismatch')
        compliant = outcome == 'ok' and action == expected['expected_action']
        if event.get('compliant') is not compliant:
            raise ValueError('qualification_compliance_self_report_mismatch')
        if outcome == 'ok':
            if action not in ACTIONS:
                raise ValueError('qualification_valid_action_invalid')
            violation = None if compliant else ('early_finish' if action == 'finish' else
                        'retry_rule' if expected['observation']['attempts_on_stage'] else 'wrong_stage')
            if event.get('violation') != violation:
                raise ValueError('qualification_violation_self_report_mismatch')
        elif action is not None or 'violation' in event:
            raise ValueError('qualification_invalid_output_action_mismatch')
    return {
        'outcome': row['outcome'], 'events': row['events'],
        'full_pipeline_completed': row['full_pipeline_completed'],
        'protocol_correct_termination': row['protocol_correct_termination'],
        'controller_calls': row['controller_calls'],
    }


def _validate_run(run_path: Path, freeze: dict, settings: dict) -> tuple[str, list[dict], dict]:
    # Imported lazily because run.py imports this module to enforce the gate.
    from run import read_complete_run

    metadata_path = Path(str(run_path) + '.run.json')
    intents_path = Path(str(run_path) + '.intents.jsonl')
    traces_path = Path(str(run_path) + '.traces.jsonl')
    warmup_traces_path = Path(str(run_path) + '.warmup.traces.jsonl')
    required = (run_path, metadata_path, intents_path, traces_path, warmup_traces_path)
    if not all(path.is_file() for path in required):
        raise ValueError('qualification_run_evidence_missing')
    rows = read_complete_run(run_path)
    metadata = json.loads(metadata_path.read_text())
    controllers = {task.get('controller') for task in metadata.get('planned_tasks', [])}
    if len(controllers) != 1:
        raise ValueError('qualification_run_controller_identity')
    controller = controllers.pop()
    if controller not in CONDITIONS:
        raise ValueError('qualification_run_controller_identity')
    expected = _expected_qualification_tasks(freeze['selection'], controller)
    if metadata.get('planned_tasks') != expected or len(rows) != 60:
        raise ValueError('qualification_run_plan_mismatch')
    if (metadata.get('schema') != 'averitec-controller-run/v1'
            or metadata.get('freeze_sha256') != digest(freeze)
            or metadata.get('config_sha256') != file_hash(EXPERIMENT / 'config.yaml')
            or metadata.get('code_binding') != code_binding()
            or metadata.get('origin') != 'real_controller_frozen_tools'
            or metadata.get('output_mode') != 'structured'):
        raise ValueError('qualification_run_binding_mismatch')
    variant, profile = CONDITIONS[controller]
    presentation = presentation_identity(variant)
    generation = _expected_generation(settings, controller, variant, profile)
    if (metadata.get('prompt_variant'), metadata.get('generation_profile')) != (variant, profile):
        raise ValueError('qualification_run_profile_mismatch')
    if (any(metadata.get(key) != value for key, value in presentation.items())
            or metadata.get('generation_settings') != generation
            or metadata.get('warmup_policy') != 'one_synthetic_request_before_pending_measurements'):
        raise ValueError('qualification_run_settings_mismatch')
    if metadata.get('model') != settings['models'][controller]:
        raise ValueError('qualification_run_model_mismatch')

    intents = _jsonl(intents_path)
    expected_ids = {task['trial_id'] for task in expected}
    if (len(intents) != 60 or {row.get('trial_id') for row in intents} != expected_ids
            or len({row.get('trace_id') for row in intents}) != 60):
        raise ValueError('qualification_intent_journal_incomplete')
    intent_traces = {row['trial_id']: row['trace_id'] for row in intents}
    if any(row.get('trace_id') != intent_traces.get(row['trial_id']) for row in rows):
        raise ValueError('qualification_intent_trace_mismatch')
    if any(trace_id != _trace_id(metadata['run_id'], trial_id)
           for trial_id, trace_id in intent_traces.items()):
        raise ValueError('qualification_intent_trace_mismatch')
    by_case = {case['case_id']: case for case in freeze['cases']}
    expected_by_id = {task['trial_id']: task for task in expected}
    model_name = settings['models'][controller]['model']
    results = {row['trial_id']: _validate_measured_row(
        row, expected_by_id[row['trial_id']], by_case[row['case_id']], model_name) for row in rows}
    trace_roots, trace_spans = _trace_spans(traces_path)
    if set(intent_traces.values()) != set(trace_roots):
        raise ValueError('qualification_trial_trace_mismatch')
    for row in rows:
        trace_id = row['trace_id']
        root = trace_roots[trace_id]
        root_end = trace_spans[(trace_id, root['span_id'])][1]
        if (root.get('name') != 'controller.trial'
                or root.get('input') != expected_by_id[row['trial_id']]
                or root_end.get('output') != results[row['trial_id']]):
            raise ValueError('qualification_trial_trace_output_mismatch')
        decisions = [(start, end) for (span_trace, _), (start, end) in trace_spans.items()
                     if span_trace == trace_id and start.get('name') == 'controller.decision']
        if (len(decisions) != len(row['events'])
                or [end.get('output') for _, end in decisions] != row['events']):
            raise ValueError('qualification_decision_trace_output_mismatch')
    if any(row.get('outcome') == 'interrupted_unknown_outcome' for row in rows):
        raise ValueError('qualification_interrupted_trial')
    if any(event.get('provider_outcome') in ('transport_error', 'version_mismatch')
           for row in rows for event in row.get('events', [])):
        raise ValueError('qualification_infrastructure_or_identity_failure')
    # invalid_output and valid-but-wrong actions intentionally remain measured outcomes.

    warmups = list(run_path.parent.glob(run_path.name + '.warmup-*.json'))
    if len(warmups) != 1:
        raise ValueError('qualification_requires_exactly_one_warmup')
    warmup = json.loads(warmups[0].read_text())
    expected_returned = settings['models'][controller].get('expected_version', model_name)
    if (warmup.get('status') != 'ok' or warmup.get('run_id') != metadata.get('run_id')
            or (warmup.get('prompt_variant'), warmup.get('generation_profile')) != (variant, profile)
            or warmup.get('generation_settings') != generation
            or warmup.get('outcome') != 'ok' or warmup.get('error_code') is not None
            or warmup.get('model') != model_name
            or warmup.get('expected_returned_model') != expected_returned
            or warmup.get('returned_model') != expected_returned):
        raise ValueError('qualification_warmup_failed_or_mismatched')
    warmup_roots, warmup_spans = _trace_spans(warmup_traces_path)
    marker = run_path.name + '.warmup-'
    warmup_id = warmups[0].name[len(marker):-len('.json')]
    if (len(warmup_roots) != 1 or len(warmup_id) != 32
            or set(warmup_roots) != {_trace_id(metadata['run_id'], 'warmup_' + warmup_id)}):
        raise ValueError('qualification_warmup_trace_mismatch')
    warmup_root = next(iter(warmup_roots.values()))
    warmup_end = warmup_spans[(warmup_root['trace_id'], warmup_root['span_id'])][1]
    wrapper_keys = {'run_id', *presentation.keys(), 'generation_profile', 'generation_settings'}
    warmup_payload = {key: value for key, value in warmup.items() if key not in wrapper_keys}
    if warmup_root.get('name') != 'controller.warmup' or warmup_end.get('output') != warmup_payload:
        raise ValueError('qualification_warmup_trace_output_mismatch')
    evidence = {
        'controller': controller,
        'result': run_path,
        'metadata': metadata_path,
        'intents': intents_path,
        'traces': traces_path,
        'warmup_receipt': warmups[0],
        'warmup_traces': warmup_traces_path,
    }
    return controller, rows, evidence


def _validate_evidence(run_paths: list[Path], freeze_path: Path, settings: dict) -> tuple[list[dict], dict]:
    _evaluation_config(settings)
    selection = prepare_selection(settings)
    freeze = json.loads(freeze_path.read_text())
    check_freeze(freeze, selection)
    verify_preparation_traces(freeze, freeze_path.parent)
    preparation_path = freeze_path.parent / freeze['preparation_trace_journal']
    preparation_roots, preparation_spans = _trace_spans(preparation_path)
    if set(preparation_roots) != {case['preparation_trace_id'] for case in freeze['cases']}:
        raise ValueError('qualification_preparation_trace_identity_mismatch')
    for case in freeze['cases']:
        root = preparation_roots[case['preparation_trace_id']]
        end = preparation_spans[(root['trace_id'], root['span_id'])][1]
        expected_output = {'outcome': case['preparation_outcome'],
                           'worker_receipt': case['worker_receipt']}
        if root.get('name') != 'shared_worker.preparation' or end.get('output') != expected_output:
            raise ValueError('qualification_preparation_trace_output_mismatch')
    expected_cases = [row['case_id'] for row in selection['cases'] if row['role'] == 'development'][2:6]
    if ([row.get('case_id') for row in freeze.get('cases', [])] != expected_cases
            or any(row.get('role') != 'development' for row in freeze.get('cases', []))):
        raise ValueError('qualification_freeze_case_slice_mismatch')
    all_rows, run_evidence, names = [], [], []
    for path in run_paths:
        controller, rows, evidence = _validate_run(path, freeze, settings)
        names.append(controller)
        all_rows.extend(rows)
        run_evidence.append(evidence)
    if Counter(names) != Counter(CONDITIONS.keys()):
        raise ValueError('qualification_requires_five_controller_runs')
    if sum(row['planned_decisions'] for row in all_rows) > settings['evaluation_run']['max_qualification_decisions']:
        raise ValueError('qualification_decision_budget_exceeded')
    summary = summarize(all_rows)
    stability = {
        'arms': [{key: arm[key] for key in ('controller', 'scenario', 'order',
                                             'repeated_checkpoint_groups', 'decision_repeatability')}
                 for arm in summary['arms']],
        'option_order_comparable_checkpoints': summary['option_order_comparable_checkpoints'],
        'option_order_changed_fraction': summary['option_order_changed_fraction'],
    }
    return run_evidence, {'full_summary': summary, 'repeatability_and_order_sensitivity': stability}


def create_qualification_receipt(run_paths: list[Path], freeze_path: Path,
                                 output: Path, settings: dict) -> dict:
    root = output.parent
    _contained_relative(output, root)
    evidence, summary = _validate_evidence(run_paths, freeze_path, settings)
    freeze = json.loads(freeze_path.read_text())
    preparation = freeze_path.parent / freeze['preparation_trace_journal']
    receipt = {
        'schema': RECEIPT_SCHEMA,
        'passed': True,
        'config_sha256': file_hash(EXPERIMENT / 'config.yaml'),
        'code_binding': code_binding(),
        'freeze': _reference(freeze_path, root),
        'preparation_traces': _reference(preparation, root),
        'runs': [{key: value if key == 'controller' else _reference(value, root)
                  for key, value in item.items()} for item in evidence],
        'summary': summary,
    }
    return receipt


def validate_qualification_receipt(path: Path, settings: dict) -> dict:
    if not path.is_file():
        raise ValueError('qualification_receipt_missing')
    receipt = json.loads(path.read_text())
    if receipt.get('schema') != RECEIPT_SCHEMA:
        raise ValueError('qualification_receipt_schema')
    if (receipt.get('config_sha256') != file_hash(EXPERIMENT / 'config.yaml')
            or receipt.get('code_binding') != code_binding()):
        raise ValueError('qualification_receipt_current_binding_mismatch')
    root = path.parent
    freeze_path = _resolve_reference(receipt.get('freeze', {}), root)
    preparation_path = _resolve_reference(receipt.get('preparation_traces', {}), root)
    freeze = json.loads(freeze_path.read_text())
    if preparation_path != freeze_path.parent / freeze.get('preparation_trace_journal', ''):
        raise ValueError('qualification_preparation_trace_path_mismatch')
    run_paths = []
    for item in receipt.get('runs', []):
        if set(item) != {'controller', 'result', 'metadata', 'intents', 'traces',
                         'warmup_receipt', 'warmup_traces'}:
            raise ValueError('qualification_receipt_run_schema')
        resolved = {key: _resolve_reference(value, root) for key, value in item.items()
                    if key != 'controller'}
        run_path = resolved['result']
        expected_paths = {
            'metadata': Path(str(run_path) + '.run.json'),
            'intents': Path(str(run_path) + '.intents.jsonl'),
            'traces': Path(str(run_path) + '.traces.jsonl'),
            'warmup_traces': Path(str(run_path) + '.warmup.traces.jsonl'),
        }
        if any(resolved[key] != value for key, value in expected_paths.items()):
            raise ValueError('qualification_receipt_run_path_mismatch')
        warmups = list(run_path.parent.glob(run_path.name + '.warmup-*.json'))
        if len(warmups) != 1 or resolved['warmup_receipt'] != warmups[0]:
            raise ValueError('qualification_receipt_run_path_mismatch')
        run_paths.append(run_path)
    evidence, summary = _validate_evidence(run_paths, freeze_path, settings)
    if [item['controller'] for item in receipt['runs']] != [item['controller'] for item in evidence]:
        raise ValueError('qualification_receipt_controller_mismatch')
    if receipt.get('summary') != summary:
        raise ValueError('qualification_receipt_summary_mismatch')
    return receipt
