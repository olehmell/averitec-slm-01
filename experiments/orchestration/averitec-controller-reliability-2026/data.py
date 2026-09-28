"""Gold-free selection and frozen, shared tool responses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from engine import STAGES, canonical, digest

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parent


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare_selection(config: dict, *, root=ROOT) -> dict:
    runtime_path = root / config['dataset_refs'][0]
    split_path = root / config['source_split_manifest']
    runtime = [json.loads(line) for line in runtime_path.read_text().splitlines() if line.strip()]
    if any(set(row) != {'case_id', 'claim', 'split'} for row in runtime):
        raise ValueError('runtime_must_be_gold_free')
    if len({r['case_id'] for r in runtime}) != len(runtime):
        raise ValueError('duplicate_runtime_case')
    by_id = {row['case_id']: row for row in runtime}
    split = json.loads(split_path.read_text())
    candidates = []
    for group in split['groups']:
        if group['role'] != 'study_fit':
            continue
        eligible = sorted(case_id for case_id in group['case_ids']
                          if case_id in by_id and by_id[case_id]['split'] == config['selection']['source_split']
                          and case_id != 'averitec-dev-0002')
        if eligible:
            candidates.append({'case_id': eligible[0], 'group_id': group['group_id'],
                               'split': by_id[eligible[0]]['split']})
    candidates.sort(key=lambda row: row['group_id'])
    random.Random(config['seed']).shuffle(candidates)
    development = config['selection']['development_cases']
    evaluation = config['selection']['evaluation_cases']
    if len(candidates) < development + evaluation:
        raise ValueError('insufficient_disjoint_groups')
    rows = [{**row, 'role': 'development' if i < development else 'evaluation'}
            for i, row in enumerate(candidates[:development + evaluation])]
    return {'schema': 'averitec-controller-selection/v1', 'seed': config['seed'],
            'runtime_path': config['dataset_refs'][0], 'runtime_sha256': file_hash(runtime_path),
            'source_split_path': config['source_split_manifest'], 'source_split_sha256': file_hash(split_path),
            'selection_config_sha256': digest(config['selection']), 'cases': rows,
            'exclusions': {'prior_exploratory_probe': ['averitec-dev-0002'],
                           'main_study_holdout': 'all groups', 'source_split': 'train'},
            'gold_used': False}


def selected_cases(selection: dict, role: str, *, root=ROOT, limit=None, offset=0) -> list[dict]:
    import yaml
    settings = yaml.safe_load((EXPERIMENT / 'config.yaml').read_text())
    if selection != prepare_selection(settings, root=root):
        raise ValueError('selection_not_canonical')
    runtime = root / selection['runtime_path']
    split = root / selection['source_split_path']
    if file_hash(runtime) != selection['runtime_sha256'] or file_hash(split) != selection['source_split_sha256']:
        raise ValueError('selection_input_drift')
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError('case_offset_must_be_nonnegative')
    metadata = [row for row in selection['cases'] if row['role'] == role]
    metadata = metadata[offset:offset + limit if limit is not None else None]
    by_id = {row['case_id']: row for row in (json.loads(line) for line in runtime.read_text().splitlines())}
    return [{**by_id[row['case_id']], 'group_id': row['group_id']} for row in metadata]


def code_binding() -> dict:
    # Shared tools are imported, so freeze their actual bytes as well as this experiment.
    paths = list(EXPERIMENT.glob('*.py')) + [EXPERIMENT / 'config.yaml']
    sibling = EXPERIMENT / 'support'
    paths += [sibling / name for name in ('averitec_fixed.py', 'fixed_inference.py', 'fixed_runner.py', 'source_corpora.py')]
    paths += list((sibling / 'inference').glob('*.py'))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def verify_preparation_traces(freeze: dict, trace_root: Path) -> None:
    """A freeze travels with its preparation journal, never with a guessed path."""
    name = freeze.get('preparation_trace_journal')
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('unsafe_preparation_trace_path')
    journal = trace_root / name
    if not journal.is_file() or file_hash(journal) != freeze.get('preparation_trace_sha256'):
        raise ValueError('preparation_trace_journal_drift')
    roots = {}
    with journal.open() as source:
        for line in source:
            row = json.loads(line)
            if row.get('event') == 'span_start' and row.get('parent_span_id') is None:
                roots[row['trace_id']] = row
    for case in freeze['cases']:
        root = roots.get(case['preparation_trace_id'])
        if not root or root.get('input', {}).get('case_id') != case['case_id']:
            raise ValueError('preparation_trace_case_mismatch')


def check_freeze(freeze: dict, selection: dict, *, allow_synthetic=False) -> None:
    import yaml
    from engine import FaultTools, OracleController, ReplayTools, SCENARIOS, checked_tool_result, run_episode
    if freeze.get('schema') != 'averitec-controller-tool-freeze/v1':
        raise ValueError('freeze_schema')
    if freeze.get('origin') not in ('real_tools', 'synthetic'):
        raise ValueError('freeze_origin')
    if not allow_synthetic and freeze['origin'] != 'real_tools':
        raise ValueError('synthetic_freeze_not_model_evidence')
    if freeze['selection_sha256'] != digest(selection):
        raise ValueError('freeze_selection_mismatch')
    if selection != prepare_selection(yaml.safe_load((EXPERIMENT / 'config.yaml').read_text())):
        raise ValueError('selection_not_canonical')
    if freeze['code_binding'] != code_binding():
        raise ValueError('freeze_code_drift')
    if not isinstance(freeze.get('preparation_trace_sha256'), str) or len(freeze['preparation_trace_sha256']) != 64:
        raise ValueError('freeze_trace_binding_missing')
    records = freeze['cases']
    if not records or len({r['case_id'] for r in records}) != len(records):
        raise ValueError('freeze_case_cardinality')
    selected = {r['case_id']: r for r in selection['cases']}
    for case in records:
        if not isinstance(case.get('preparation_trace_id'), str) or len(case['preparation_trace_id']) != 32:
            raise ValueError('case_trace_binding_missing')
        if case['case_id'] not in selected or case['group_id'] != selected[case['case_id']]['group_id']:
            raise ValueError('freeze_case_identity')
        if case['role'] != selected[case['case_id']]['role']:
            raise ValueError('freeze_case_role')
        if case['records_sha256'] != digest(case['records']):
            raise ValueError('freeze_records_drift')
        for record in case['records']:
            if set(record) != {'action', 'result'} or record['action'] not in STAGES:
                raise ValueError('freeze_record_schema')
            checked_tool_result(record['result'])
        tools = ReplayTools(case['records'])
        reference = run_episode(OracleController(), tools)
        if tools.position != len(case['records']):
            raise ValueError('unused_frozen_tool_results')
        if reference['outcome'] not in ('completed', 'correct_abort'):
            raise ValueError('freeze_reference_invalid')
        if reference['outcome'] != case['preparation_outcome']:
            raise ValueError('shared_worker_preparation_outcome_drift')
        if 'worker_receipt' not in case or case['worker_receipt_sha256'] != digest(case['worker_receipt']):
            raise ValueError('worker_receipt_missing_or_changed')
        for scenario in SCENARIOS:
            run_episode(OracleController(), FaultTools(ReplayTools(case['records']), scenario))
