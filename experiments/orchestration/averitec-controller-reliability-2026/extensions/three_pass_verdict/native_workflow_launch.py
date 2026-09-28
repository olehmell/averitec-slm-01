#!/usr/bin/env python3
"""Build and stage exact native workflow jobs; submit one approved pass once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import native_workflow_passes as workflow  # noqa: E402

EXT = Path('experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict')
VEGA_BASE = Path(os.environ.get('AVERITEC_COMPUTE_ROOT', '.private-artifacts/averitec-compute'))
INTENT_SCHEMA = 'averitec-native-workflow-submit-intent/v1'
ACCEPTED_SCHEMA = 'averitec-native-workflow-submit-accepted/v1'
BUNDLE_SCHEMA = 'averitec-native-workflow-bundle/v1'
SUBMISSION_GATE_SCHEMA = 'averitec-native-workflow-submission-gate/v1'
HEX = re.compile(r'[0-9a-f]{64}\Z')
ASSET_KEYS = {'jeff': {'checkpoint', 'asset_manifest', 'image', 'runtime'},
              'laya_typed': {'checkpoint', 'asset_manifest', 'wheel', 'image', 'runtime'}}


def _new_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _new_json(path: Path, value: dict) -> None:
    _new_bytes(path, (workflow.canonical(value) + '\n').encode())


def _head_blob(name: str) -> bytes:
    return subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=REPO)


def _committed_file(name: str) -> bytes:
    path = REPO / name
    blob = _head_blob(name)
    if not path.is_file() or path.is_symlink() or path.read_bytes() != blob:
        raise ValueError('native_uncommitted_source:' + name)
    return blob


def asset_index(path: Path) -> dict:
    value = workflow.read(path)
    if not isinstance(value, dict) or set(value) != {'schema', 'arms', 'receipts', 'historical_preflights'} or value['schema'] != 'averitec-native-workflow-asset-index/v1':
        raise ValueError('native_asset_index_shape')
    if (set(value['arms']) != set(workflow.ARM_INFO) or set(value['receipts']) != set(workflow.ARM_INFO)
            or set(value['historical_preflights']) != set(workflow.ARM_INFO)):
        raise ValueError('native_asset_index_arms')
    for arm in workflow.ARM_INFO:
        item = value['arms'][arm]
        if not isinstance(item, dict) or set(item) != ASSET_KEYS[arm] | {'python_paths'}:
            raise ValueError('native_asset_index_entry')
        python_paths = item['python_paths']
        if not isinstance(python_paths, list) or not python_paths or len(set(python_paths)) != len(python_paths):
            raise ValueError('native_python_paths')
        all_paths = [item[key] for key in ASSET_KEYS[arm]] + python_paths
        if any(not isinstance(p, str) or not p.startswith('/') or '\n' in p or ':' in p or ',' in p
               or not Path(p).exists() or Path(p).is_symlink() for p in all_paths):
            raise ValueError('native_asset_path')
        if (not Path(item['checkpoint']).is_dir() or not Path(item['runtime']).is_dir()
                or any(not Path(p).is_dir() for p in python_paths)
                or any(not Path(item[k]).is_file() for k in ASSET_KEYS[arm] - {'checkpoint', 'runtime'})):
            raise ValueError('native_asset_path_type')
        receipts = value['receipts'][arm]
        expected = {'qualification'} if arm == 'jeff' else {'selection', 'qualification'}
        if not isinstance(receipts, dict) or set(receipts) != expected:
            raise ValueError('native_receipt_index')
        if any(not isinstance(p, str) or not p.startswith('/') or not Path(p).is_file() or Path(p).is_symlink()
               for p in receipts.values()):
            raise ValueError('native_receipt_path')
        preflight = value['historical_preflights'][arm]
        if not isinstance(preflight, str) or not preflight.startswith('/') or not Path(preflight).is_file() or Path(preflight).is_symlink():
            raise ValueError('native_historical_preflight_path')
    return value


def _hash_index(index: dict) -> tuple[dict, dict]:
    if workflow.jeff_checkpoint_digest(Path(index['arms']['jeff']['checkpoint'])) != '64bc03d73b133461134468ad22b5ff2315579419d534018561742624a1a262df':
        raise ValueError('native_historical_jeff_checkpoint_drift')
    workflow.verify_laya_assets(Path(index['arms']['laya_typed']['checkpoint']),
                                Path(index['arms']['laya_typed']['asset_manifest']))
    assets = {arm: {key: workflow.sha(Path(item[key])) for key in ASSET_KEYS[arm]}
              for arm, item in index['arms'].items()}
    receipts = {arm: {key: workflow.sha(Path(path)) for key, path in item.items()}
                for arm, item in index['receipts'].items()}
    return assets, receipts


def _caps() -> dict:
    return {f'{arm}:{number}': {'wall_seconds': 1020, 'gpu_seconds': 1020, 'gpu_jobs': 1,
              'requests': 2300, 'input_tokens_total': 2_300_000,
              'input_tokens_per_request': 1000, 'context_tokens': 1024 if arm == 'laya_typed' else 4096}
            for number in (1, 2, 3) for arm in workflow.ARM_INFO}


def draft(freeze_path: Path, preparation_path: Path, index_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('native_manifest_destination_exists')
    index = asset_index(index_path)
    assets, receipts = _hash_index(index)
    freeze = workflow.read(freeze_path)
    code = {}
    for name in workflow.REQUIRED_CODE:
        relative = (workflow.EXP / name).relative_to(REPO).as_posix()
        code[name] = hashlib.sha256(_committed_file(relative)).hexdigest()
    namespaces = {f'{arm}:{number}': f'{arm}-workflow-pass-{number}'
                  for number in (1, 2, 3) for arm in workflow.ARM_INFO}
    value = {'schema': workflow.SCHEMA, 'protocol_version': 'three-pass-verdict-v1',
             'source_commit': workflow.commit(), 'historical_core_commit': workflow.HISTORICAL_CORE_COMMIT,
             'code_sha256': code,
             'freeze_sha256': workflow.sha(freeze_path),
             'freeze_content_sha256': workflow.object_digest(freeze),
             'preparation_sha256': workflow.sha(preparation_path),
             'case_ids': [row['case_id'] for row in freeze['cases']],
             'source_groups': [row['group_id'] for row in freeze['cases']],
             'scenarios': list(workflow.SCENARIOS), 'action_order': list(workflow.ACTIONS),
             'retries': 0, 'passes': [1, 2, 3],
             'arms': {arm: {'model': model, 'returned_model': returned, 'profile': profile,
                            'revision': workflow.ARM_REVISIONS[arm]}
                      for arm, (model, returned, profile) in workflow.ARM_INFO.items()},
             'assets': assets, 'historical_receipts': receipts,
             'historical_preflight_sha256': {arm: workflow.sha(Path(path)) for arm, path in index['historical_preflights'].items()},
             'preflight_sha256': {arm: '0' * 64 for arm in workflow.ARM_INFO},
             'output_namespaces': namespaces, 'caps': _caps()}
    value['job_plan_sha256'] = workflow.object_digest(workflow.job_plan(value))
    # This validates the gold-free freeze and exact 300/2,299 plan on both arms.
    _new_json(output, value)
    try:
        for arm in workflow.ARM_INFO:
            workflow.validate_manifest(output, arm, 1, freeze_path, preparation_path)
    except BaseException:
        output.unlink()
        raise
    return value


def reconstruct_preflight(draft_path: Path, freeze_path: Path, preparation_path: Path,
                          index_path: Path, arm: str, output: Path | None) -> dict:
    if (output is not None and output.exists()) or arm not in workflow.ARM_INFO:
        raise ValueError('native_preflight_destination_or_arm')
    manifest, freeze, plan = workflow.validate_manifest(draft_path, arm, 1, freeze_path, preparation_path)
    index = asset_index(index_path)
    old_path = Path(index['historical_preflights'][arm])
    if workflow.sha(old_path) != manifest['historical_preflight_sha256'][arm]:
        raise ValueError('native_historical_preflight_drift')
    old = workflow.read(old_path)
    if not isinstance(old, dict) or set(old) != {'checks'} or not isinstance(old['checks'], list):
        raise ValueError('native_historical_preflight_shape')
    historical = old['checks']
    by_case = {row['case_id']: row for row in freeze['cases']}
    checks, expected_old_keys, seen = [], [], set()
    for trial in plan['trials']:
        case = by_case[trial['case_id']]
        reference = workflow.run_episode(workflow.OracleController(),
                    workflow.FaultTools(workflow.ReplayTools(case['records']), trial['scenario']))
        for event in reference['events']:
            state_hash = event['observation_sha256']
            if state_hash in seen:
                continue
            expected_old_keys.append(workflow.object_digest([event['observation'], list(workflow.ACTIONS)]))
            seen.add(state_hash)
            checks.append({'observation_sha256': state_hash})
    if (len(checks) != 1148 or len(historical) != 1149
            or [row.get('input_sha256') for row in historical[:-1]] != expected_old_keys
            or historical[-1].get('warmup') is not True):
        raise ValueError('native_historical_preflight_state_join')
    caps = manifest['caps'][f'{arm}:1']
    for row, old_row in zip(checks, historical[:-1]):
        report = {key: value for key, value in old_row.items() if key != 'input_sha256'}
        tokens = report.get('total_tokens', report.get('input_tokens'))
        if (report.get('lossless') is not True or type(tokens) is not int or tokens < 0
                or tokens > caps['input_tokens_per_request'] or tokens > caps['context_tokens']):
            raise ValueError('native_historical_preflight_loss_or_cap')
        row['report'] = report
    warm = {key: value for key, value in historical[-1].items() if key != 'warmup'}
    warm_tokens = warm.get('total_tokens', warm.get('input_tokens'))
    if (warm.get('lossless') is not True or type(warm_tokens) is not int or warm_tokens < 0
            or warm_tokens > caps['input_tokens_per_request'] or warm_tokens > caps['context_tokens']):
        raise ValueError('native_historical_warmup_loss_or_cap')
    checks.append({'warmup': True, 'observation_sha256': workflow.object_digest(workflow.WARMUP_OBSERVATION), 'report': warm})
    report = {'schema': 'averitec-native-workflow-reconstructed-preflight/v1',
              'mode': 'historical_reconstruction_live_recheck_required',
              'arm': arm, 'profile': workflow.ARM_INFO[arm][2],
              'manifest_sha256': workflow.sha(draft_path), 'plan_sha256': plan['plan_sha256'],
              'historical_preflight_sha256': workflow.sha(old_path),
              'historical_qualification_sha256': manifest['historical_receipts'][arm]['qualification'],
              'checks': checks, 'sha256': workflow.object_digest(checks),
              'historical_parser_note': ('Jeff historical parser differed in probability tolerance; preflight itself is unchanged.'
                                         if arm == 'jeff' else 'Laya historical native preflight profile is unchanged.')}
    if output is not None:
        _new_json(output, report)
    return report


def finalize(draft_path: Path, freeze_path: Path, preparation_path: Path,
             reports: dict[str, Path], index_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('native_manifest_destination_exists')
    value = workflow.read(draft_path)
    for arm in workflow.ARM_INFO:
        workflow.validate_manifest(draft_path, arm, 1, freeze_path, preparation_path)
    index = asset_index(index_path)
    assets, receipts = _hash_index(index)
    if assets != value['assets'] or receipts != value['historical_receipts']:
        raise ValueError('native_final_asset_drift')
    for arm in workflow.ARM_INFO:
        report = workflow.read(reports[arm])
        if (report.get('arm') != arm or report.get('profile') != workflow.ARM_INFO[arm][2]
                or report.get('manifest_sha256') != workflow.sha(draft_path)
                or report.get('plan_sha256') != workflow.make_plan(workflow.read(freeze_path), arm, 1)['plan_sha256']
                or report.get('sha256') != workflow.object_digest(report.get('checks'))
                or not isinstance(report.get('checks'), list) or len(report['checks']) != 1149
                or report['checks'][-1].get('warmup') is not True):
            raise ValueError('native_preflight_report_binding')
        if report.get('mode') == 'historical_reconstruction_live_recheck_required':
            if (report.get('historical_preflight_sha256') != value['historical_preflight_sha256'][arm]
                    or report.get('historical_qualification_sha256') != value['historical_receipts'][arm]['qualification']):
                raise ValueError('native_preflight_provenance')
            expected = reconstruct_preflight(draft_path, freeze_path, preparation_path, index_path, arm, None)
            if report != expected:
                raise ValueError('native_preflight_reconstruction_drift')
        else:
            identity = report.get('identity')
            if (not isinstance(identity, dict) or identity.get('model') != workflow.ARM_INFO[arm][0]
                    or identity.get('instruction_profile') != workflow.ARM_INFO[arm][2]
                    or identity.get('expected_returned_model', identity.get('returned_model')) != workflow.ARM_INFO[arm][1]):
                raise ValueError('native_preflight_identity')
        value['preflight_sha256'][arm] = report['sha256']
    _new_json(output, value)
    try:
        for arm in workflow.ARM_INFO:
            workflow.validate_manifest(output, arm, 1, freeze_path, preparation_path)
    except BaseException:
        output.unlink()
        raise
    return value


def _relative_inputs(freeze_path: Path, preparation_path: Path, index_path: Path,
                     manifest_path: Path, gates: dict[str, Path]) -> dict[str, bytes]:
    if freeze_path.name == preparation_path.name:
        raise ValueError('native_freeze_preparation_names')
    result = {freeze_path.name: freeze_path.read_bytes(), preparation_path.name: preparation_path.read_bytes(),
              'asset-index.json': index_path.read_bytes(), 'launch-manifest.json': manifest_path.read_bytes()}
    for key, path in gates.items():
        result[f'gate-{key.replace(":", "-pass-")}.json'] = path.read_bytes()
    return result


def check_budget(path: Path) -> dict:
    value = workflow.read(path)
    required = {'schema', 'as_of_utc', 'prior_gpu_seconds_reserved', 'prior_jobs',
                'this_workflow_gpu_seconds', 'this_workflow_jobs',
                'maximum_gpu_seconds_total', 'maximum_jobs_total', 'source_records_sha256'}
    if (not isinstance(value, dict) or set(value) != required
            or value['schema'] != 'averitec-native-workflow-budget-reconciliation/v1'
            or not isinstance(value['as_of_utc'], str) or not value['as_of_utc']
            or any(type(value[key]) is not int or value[key] < 0
                   for key in ('prior_gpu_seconds_reserved', 'prior_jobs'))
            or value['this_workflow_gpu_seconds'] != 6120 or value['this_workflow_jobs'] != 6
            or value['maximum_gpu_seconds_total'] != 91140 or value['maximum_jobs_total'] != 48
            or value['prior_gpu_seconds_reserved'] + 6120 > 91140
            or value['prior_jobs'] + 6 > 48
            or not isinstance(value['source_records_sha256'], dict) or not value['source_records_sha256']
            or any(not isinstance(name, str) or not HEX.fullmatch(str(digest))
                   for name, digest in value['source_records_sha256'].items())):
        raise ValueError('native_budget_reconciliation_invalid')
    return value


def check_smoke(path: Path, arm: str) -> dict:
    value = workflow.read(path)
    if (not isinstance(value, dict) or value.get('schema') != 'averitec-native-workflow-development-smoke/v1'
            or value.get('arm') != arm or value.get('phase') != 'development_smoke'
            or value.get('status') != 'complete' or value.get('model_calls') != 1
            or not isinstance(value.get('evidence_sha256'), dict) or not value['evidence_sha256']
            or any(not HEX.fullmatch(str(digest)) for digest in value['evidence_sha256'].values())):
        raise ValueError('native_smoke_receipt_invalid:' + arm)
    return value


def _source_files(freeze: dict, manifest: dict) -> dict[str, bytes]:
    names = {str((workflow.EXP / name).relative_to(REPO)) for name in manifest['code_sha256']}
    names |= set(freeze['code_binding'])
    cfg = workflow.EXP / 'config.yaml'
    names.add(str(cfg.relative_to(REPO)))
    result = {}
    for name in sorted(names):
        if Path(name).is_absolute() or '..' in Path(name).parts or name.startswith('datasets/'):
            raise ValueError('native_source_path')
        expected = freeze['code_binding'].get(name)
        if expected is not None:
            # The freeze binds older adaptive-core files; stage their exact
            # committed versions while retaining the new runner files.
            blob = subprocess.check_output(['git', 'show', f'{workflow.HISTORICAL_CORE_COMMIT}:{name}'], cwd=REPO)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError('native_historical_core_hash_mismatch:' + name)
            if name in {str((workflow.EXP / item).relative_to(REPO)) for item in manifest['code_sha256']}:
                if blob != _committed_file(name):
                    raise ValueError('native_direct_code_differs_from_freeze:' + name)
        else:
            blob = _committed_file(name)
        result[name] = blob
    return result


def _submission_command(bundle: Path, arm: str, number: int) -> list[str]:
    script = bundle / 'source' / EXT / 'gpu/native_workflow.slurm'
    export = f'ALL,NW_BUNDLE={bundle},NW_ARM={arm},NW_PASS={number}'
    return ['sbatch', '--parsable', '--no-requeue', '--time', '00:17:00', '--chdir', str(bundle),
            '--export', export, '--output', str(bundle / 'logs' / f'{arm}-pass-{number}-%j.out'),
            '--error', str(bundle / 'logs' / f'{arm}-pass-{number}-%j.err'), str(script)]


def planned_intents(bundle: Path, manifest: dict) -> dict:
    jobs = []
    script = bundle / 'source' / EXT / 'gpu/native_workflow.slurm'
    for row in workflow.job_plan(manifest)['jobs']:
        arm, number = row['arm'], row['pass']
        gate = bundle / 'inputs' / f'gate-{arm}-pass-{number}.json'
        jobs.append({'arm': arm, 'pass': number, 'output': str(bundle / 'outputs' / row['output_namespace']),
                     'manifest_sha256': workflow.sha(bundle / 'inputs/launch-manifest.json'),
                     'gate_sha256': workflow.sha(gate), 'script_sha256': workflow.sha(script),
                     'command': _submission_command(bundle, arm, number),
                     'gpu_seconds': 1020, 'gpu_jobs': 1, 'retries': 0})
    return {'schema': 'averitec-native-workflow-planned-submissions/v1', 'jobs': jobs}


def prepare(bundle: Path, freeze_path: Path, preparation_path: Path, index_path: Path,
            manifest_path: Path, gates: dict[str, Path], budget_path: Path,
            smoke_paths: dict[str, Path]) -> dict:
    if bundle.exists() or not bundle.parent.is_dir() or not bundle.is_absolute():
        raise ValueError('native_bundle_destination_must_be_new')
    if bundle.parent != VEGA_BASE or not re.fullmatch(r'native-workflow-[a-z0-9-]+', bundle.name):
        raise ValueError('native_vega_bundle_namespace')
    if freeze_path.name != 'freeze.json' or preparation_path.name != 'evaluation100.json.preparation.traces.jsonl':
        raise ValueError('native_freeze_staged_names')
    keys = {f'{arm}:{number}' for number in (1, 2, 3) for arm in workflow.ARM_INFO}
    if set(gates) != keys:
        raise ValueError('native_six_gates_required')
    if set(smoke_paths) != set(workflow.ARM_INFO):
        raise ValueError('native_two_smokes_required')
    check_budget(budget_path)
    for arm, path in smoke_paths.items():
        check_smoke(path, arm)
    manifest = workflow.read(manifest_path)
    for arm in workflow.ARM_INFO:
        workflow.validate_manifest(manifest_path, arm, 1, freeze_path, preparation_path)
    if '0' * 64 in manifest['preflight_sha256'].values():
        raise ValueError('native_preflight_unsealed')
    for key, gate_path in gates.items():
        arm, number = key.split(':')
        workflow.check_gate(gate_path, manifest_path, arm, int(number))
    index = asset_index(index_path)
    if _hash_index(index) != (manifest['assets'], manifest['historical_receipts']):
        raise ValueError('native_asset_drift')
    freeze = workflow.read(freeze_path)
    sources = _source_files(freeze, manifest)
    inputs = _relative_inputs(freeze_path, preparation_path, index_path, manifest_path, gates)
    inputs['budget-reconciliation.json'] = budget_path.read_bytes()
    for arm, path in smoke_paths.items():
        inputs[f'smoke-{arm}.json'] = path.read_bytes()
    staged_index = json.loads(index_path.read_text(encoding='utf-8'))
    for arm, path in index['historical_preflights'].items():
        inputs[f'historical-preflight-{arm}.json'] = Path(path).read_bytes()
        staged_index['historical_preflights'][arm] = str(bundle / 'inputs' / f'historical-preflight-{arm}.json')
    for arm, row in index['receipts'].items():
        for name, path in row.items():
            inputs[f'historical-{arm}-{name}-receipt.json'] = Path(path).read_bytes()
            staged_index['receipts'][arm][name] = str(bundle / 'inputs' / f'historical-{arm}-{name}-receipt.json')
    inputs['asset-index.json'] = (workflow.canonical(staged_index) + '\n').encode()
    # check_freeze recomputes selection from the gold-free runtime file and split manifest.
    import yaml
    config = yaml.safe_load((workflow.EXP / 'config.yaml').read_text())
    for relative in (config['dataset_refs'][0], config['source_split_manifest']):
        path = REPO / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError('native_selection_source_missing:' + relative)
        inputs[Path(relative).name] = path.read_bytes()
    bundle.mkdir(mode=0o700)
    for name in ('source', 'inputs', 'outputs', 'logs', 'submissions'):
        (bundle / name).mkdir()
    _new_bytes(bundle / 'source/SOURCE_COMMIT', (manifest['source_commit'] + '\n').encode())
    for name, blob in sources.items():
        _new_bytes(bundle / 'source' / name, blob)
    for relative in (config['dataset_refs'][0], config['source_split_manifest']):
        _new_bytes(bundle / 'source' / relative, (REPO / relative).read_bytes())
    for name, blob in inputs.items():
        _new_bytes(bundle / 'inputs' / name, blob)
    _new_json(bundle / 'job-plan.json', workflow.job_plan(manifest))
    _new_json(bundle / 'planned-intents.json', planned_intents(bundle, manifest))
    receipt = {'schema': BUNDLE_SCHEMA, 'bundle_path': str(bundle),
               'source_commit': manifest['source_commit'],
               'historical_core_commit': manifest['historical_core_commit'],
               'manifest_sha256': workflow.sha(bundle / 'inputs/launch-manifest.json'),
               'job_plan_sha256': workflow.sha(bundle / 'job-plan.json'),
               'planned_intents_sha256': workflow.sha(bundle / 'planned-intents.json'),
               'source_sha256': {name: workflow.sha(bundle / 'source' / name) for name in sources},
               'selection_sources_sha256': {name: workflow.sha(bundle / 'source' / name)
                                            for name in (config['dataset_refs'][0], config['source_split_manifest'])},
               'inputs_sha256': {name: workflow.sha(bundle / 'inputs' / name) for name in inputs},
               'maximum_jobs': 6, 'maximum_gpu_seconds': 6120,
               'model_calls': 0, 'submitted_jobs': 0, 'gold_staged': False}
    _new_json(bundle / 'bundle-receipt.json', receipt)
    return receipt


def _check_namespace(output: Path, *, allow_existing_output: bool = False,
                     allow_runtime: bool = False) -> None:
    if not allow_existing_output and output.exists():
        raise ValueError('native_output_already_exists')
    runtime = Path(str(output) + '.runtime')
    if not allow_runtime and runtime.exists():
        raise ValueError('native_runtime_already_exists')
    if allow_runtime and runtime.exists() and (not runtime.is_dir() or runtime.is_symlink()):
        raise ValueError('native_runtime_invalid')


def verify_bundle(bundle: Path, arm: str, number: int, *, require_gate: bool = True,
                  allow_existing_output: bool = False,
                  allow_runtime: bool = False) -> tuple[dict, dict, Path, Path]:
    if arm not in workflow.ARM_INFO or number not in (1, 2, 3) or not bundle.is_dir() or bundle.is_symlink():
        raise ValueError('native_bundle_or_scope')
    receipt = workflow.read(bundle / 'bundle-receipt.json')
    manifest_path = bundle / 'inputs/launch-manifest.json'
    manifest = workflow.read(manifest_path)
    key = f'{arm}:{number}'
    gate_path = bundle / 'inputs' / f'gate-{arm}-pass-{number}.json'
    if (receipt.get('schema') != BUNDLE_SCHEMA or receipt.get('bundle_path') != str(bundle)
            or receipt.get('source_commit') != manifest['source_commit']
            or receipt.get('historical_core_commit') != manifest['historical_core_commit']
            or receipt.get('manifest_sha256') != workflow.sha(manifest_path)
            or receipt.get('job_plan_sha256') != workflow.sha(bundle / 'job-plan.json')
            or receipt.get('planned_intents_sha256') != workflow.sha(bundle / 'planned-intents.json')
            or receipt.get('gold_staged') is not False):
        raise ValueError('native_bundle_receipt_drift')
    if workflow.object_digest(workflow.job_plan(manifest)) != manifest['job_plan_sha256']:
        raise ValueError('native_job_plan_drift')
    if workflow.read(bundle / 'planned-intents.json') != planned_intents(bundle, manifest):
        raise ValueError('native_planned_intents_drift')
    check_budget(bundle / 'inputs/budget-reconciliation.json')
    for arm_name in workflow.ARM_INFO:
        check_smoke(bundle / 'inputs' / f'smoke-{arm_name}.json', arm_name)
    index = asset_index(bundle / 'inputs/asset-index.json')
    if _hash_index(index) != (manifest['assets'], manifest['historical_receipts']):
        raise ValueError('native_staged_asset_drift')
    for arm_name, path in index['historical_preflights'].items():
        if workflow.sha(Path(path)) != manifest['historical_preflight_sha256'][arm_name]:
            raise ValueError('native_staged_historical_preflight_drift')
    if require_gate:
        workflow.check_gate(gate_path, manifest_path, arm, number)
    for section, root in (('source_sha256', bundle / 'source'), ('selection_sources_sha256', bundle / 'source'),
                          ('inputs_sha256', bundle / 'inputs')):
        for name, expected in receipt[section].items():
            path = root / name
            if path.is_symlink() or workflow.sha(path) != expected:
                raise ValueError('native_bundle_member_drift:' + name)
    source_files = {p.relative_to(bundle / 'source').as_posix() for p in (bundle / 'source').rglob('*') if p.is_file()}
    if source_files != ({'SOURCE_COMMIT'} | set(receipt['source_sha256']) | set(receipt['selection_sources_sha256'])):
        raise ValueError('native_bundle_source_set')
    input_files = {p.name for p in (bundle / 'inputs').iterdir() if p.is_file()}
    if input_files != set(receipt['inputs_sha256']):
        raise ValueError('native_bundle_input_set')
    if (bundle / 'source/SOURCE_COMMIT').read_text().strip() != manifest['source_commit']:
        raise ValueError('native_source_commit_drift')
    output = bundle / 'outputs' / manifest['output_namespaces'][key]
    _check_namespace(output, allow_existing_output=allow_existing_output,
                     allow_runtime=allow_runtime)
    return receipt, manifest, gate_path, output


def check_submission_gate(bundle: Path) -> dict:
    gate = workflow.read(bundle / 'submission-gate.json')
    expected = {'schema': SUBMISSION_GATE_SCHEMA, 'decision': 'approved', 'reviewer': 'gpt-6-astra',
                'bundle_receipt_sha256': workflow.sha(bundle / 'bundle-receipt.json'),
                'planned_intents_sha256': workflow.sha(bundle / 'planned-intents.json'),
                'budget_reconciliation_sha256': workflow.sha(bundle / 'inputs/budget-reconciliation.json'),
                'smoke_sha256': {arm: workflow.sha(bundle / 'inputs' / f'smoke-{arm}.json')
                                 for arm in workflow.ARM_INFO}}
    if gate != expected:
        raise ValueError('native_submission_astra_gate')
    return gate


def submit(bundle: Path, arm: str, number: int) -> dict:
    receipt, manifest, gate_path, output = verify_bundle(bundle, arm, number)
    check_submission_gate(bundle)
    key = f'{arm}:{number}'
    path = bundle / 'submissions' / f'{arm}-pass-{number}.intent.json'
    accepted_path = bundle / 'submissions' / f'{arm}-pass-{number}.accepted.json'
    script = bundle / 'source' / EXT / 'gpu/native_workflow.slurm'
    intent = {'schema': INTENT_SCHEMA, 'arm': arm, 'pass': number, 'bundle': str(bundle),
              'manifest_sha256': workflow.sha(bundle / 'inputs/launch-manifest.json'),
              'gate_sha256': workflow.sha(gate_path), 'script_sha256': workflow.sha(script),
              'job_plan_sha256': manifest['job_plan_sha256'], 'output': str(output),
              'gpu_seconds': 1020, 'gpu_jobs': 1, 'retries': 0, 'created_unix': time.time()}
    _new_json(path, intent)
    command = _submission_command(bundle, arm, number)
    # One scheduler call. Unknown/failed submission leaves the durable intent and cannot retry.
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    raw = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r'[1-9][0-9]*(?:;[A-Za-z0-9._-]+)?', raw):
        raise RuntimeError('native_submission_unknown_no_retry')
    accepted = {'schema': ACCEPTED_SCHEMA, 'arm': arm, 'pass': number,
                'job_id': raw.split(';', 1)[0], 'intent_sha256': workflow.sha(path),
                'manifest_sha256': intent['manifest_sha256'], 'gate_sha256': intent['gate_sha256'],
                'job_plan_sha256': intent['job_plan_sha256'], 'gpu_seconds': 1020}
    _new_json(accepted_path, accepted)
    return accepted


def verify_runtime(bundle: Path, arm: str, number: int, job_id: str, *, allow_existing_output: bool = False) -> dict:
    _receipt, manifest, gate_path, _output = verify_bundle(bundle, arm, number,
                                                            allow_existing_output=allow_existing_output,
                                                            allow_runtime=True)
    check_submission_gate(bundle)
    prefix = f'{arm}-pass-{number}'
    intent_path = bundle / 'submissions' / (prefix + '.intent.json')
    accepted_path = bundle / 'submissions' / (prefix + '.accepted.json')
    intent, accepted = workflow.read(intent_path), workflow.read(accepted_path)
    if (intent.get('schema') != INTENT_SCHEMA or intent.get('arm') != arm or intent.get('pass') != number
            or intent.get('bundle') != str(bundle) or intent.get('gpu_seconds') != 1020
            or intent.get('manifest_sha256') != workflow.sha(bundle / 'inputs/launch-manifest.json')
            or intent.get('gate_sha256') != workflow.sha(gate_path)
            or intent.get('job_plan_sha256') != manifest['job_plan_sha256']
            or accepted != {'schema': ACCEPTED_SCHEMA, 'arm': arm, 'pass': number,
                            'job_id': job_id, 'intent_sha256': workflow.sha(intent_path),
                            'manifest_sha256': intent['manifest_sha256'], 'gate_sha256': intent['gate_sha256'],
                            'job_plan_sha256': intent['job_plan_sha256'], 'gpu_seconds': 1020}):
        raise ValueError('native_submission_binding')
    return accepted


def collect_sacct(bundle: Path, arm: str, number: int, job_id: str) -> dict:
    verify_runtime(bundle, arm, number, job_id, allow_existing_output=True)
    target = bundle / 'submissions' / f'{arm}-pass-{number}.sacct.json'
    if target.exists():
        raise ValueError('native_sacct_already_recorded')
    command = ['sacct', '-n', '-P', '-j', job_id, '--format', 'JobID,ElapsedRaw,AllocTRES,State']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError('native_sacct_unavailable')
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    roots = [row for row in rows if len(row) == 4 and row[0] == job_id]
    if len(roots) != 1 or not roots[0][1].isdigit():
        raise ValueError('native_sacct_job_missing')
    raw_state = roots[0][3].split()[0].split('+')[0]
    if raw_state not in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}:
        raise ValueError('native_sacct_not_terminal')
    record = {'schema': 'averitec-native-workflow-sacct/v1', 'arm': arm, 'pass': number,
              'job_id': job_id, 'elapsed_raw_seconds': int(roots[0][1]),
              'alloc_tres': roots[0][2], 'state': raw_state,
              'accepted_sha256': workflow.sha(bundle / 'submissions' / f'{arm}-pass-{number}.accepted.json'),
              'raw_lines': result.stdout.splitlines()}
    _new_json(target, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('draft', 'reconstruct-preflight', 'finalize', 'prepare', 'submit', 'verify-runtime', 'collect-sacct'):
        modes.add_argument('--' + name, action='store_true')
    for name in ('freeze', 'preparation', 'asset-index', 'draft-manifest', 'manifest',
                 'jeff-preflight', 'laya-preflight', 'gates-dir', 'bundle', 'output',
                 'budget-reconciliation', 'jeff-smoke', 'laya-smoke'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--arm', choices=tuple(workflow.ARM_INFO))
    parser.add_argument('--pass-number', type=int, choices=(1, 2, 3))
    parser.add_argument('--job-id')
    args = parser.parse_args()
    gates = {f'{arm}:{number}': args.gates_dir / f'gate-{arm}-pass-{number}.json'
             for number in (1, 2, 3) for arm in workflow.ARM_INFO} if args.gates_dir else {}
    if args.draft:
        result = draft(args.freeze, args.preparation, args.asset_index, args.output)
    elif args.reconstruct_preflight:
        result = reconstruct_preflight(args.draft_manifest, args.freeze, args.preparation,
                                       args.asset_index, args.arm, args.output)
    elif args.finalize:
        result = finalize(args.draft_manifest, args.freeze, args.preparation,
                          {'jeff': args.jeff_preflight, 'laya_typed': args.laya_preflight},
                          args.asset_index, args.output)
    elif args.prepare:
        result = prepare(args.bundle, args.freeze, args.preparation, args.asset_index, args.manifest, gates,
                         args.budget_reconciliation, {'jeff': args.jeff_smoke, 'laya_typed': args.laya_smoke})
    elif args.submit:
        result = submit(args.bundle, args.arm, args.pass_number)
    elif args.collect_sacct:
        result = collect_sacct(args.bundle, args.arm, args.pass_number, args.job_id)
    else:
        result = verify_runtime(args.bundle, args.arm, args.pass_number, args.job_id)
    print(workflow.canonical(result))


if __name__ == '__main__':
    main()
