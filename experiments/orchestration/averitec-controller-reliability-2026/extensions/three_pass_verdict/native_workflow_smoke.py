#!/usr/bin/env python3
"""One-shot, one-call development smoke for original Jeff/Laya workflow adapters."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import native_workflow_passes as workflow  # noqa: E402
import native_workflow_launch as launch  # noqa: E402

SCHEMA = 'averitec-native-workflow-smoke-manifest/v1'
BUNDLE_SCHEMA = 'averitec-native-workflow-smoke-bundle/v1'
GATE_SCHEMA = 'averitec-native-workflow-smoke-astra-gate/v1'
RECEIPT_SCHEMA = 'averitec-native-workflow-development-smoke/v1'
INTENT_SCHEMA = 'averitec-native-workflow-smoke-submit-intent/v1'
ACCEPTED_SCHEMA = 'averitec-native-workflow-smoke-submit-accepted/v1'
DEV_FREEZE_SHA = '7bf4384e6cab8ccc7a6412f0a48328482f68d47c59d93ba5521271b9bff0f77d'
DEV_PREPARATION_SHA = '30212f623efefb8612382692f8852c55c54cf2ea354c2f2c8e126928233d34b5'
DEV_PREFLIGHT_SHA = {
    'jeff': 'f88a1d671ea4bd0fb78ea42859cbfe5c5b2196b4c25cff863546fbef60bc1fb0',
    'laya_typed': 'cdda2b8d2e8411659c1131b5da8d213413a8a115f42ffbc20ca715c8cbbefd50',
}
# Six unsubmitted native workflow jobs are reduced from 1,200 to 1,020 seconds.
# Historical local workflow allocations, including failed attempts, stay charged.
TRANSFER_HEADROOM = {'jeff': 480, 'laya_typed': 480}
EXTRA_SOURCE = (
    'experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/native_workflow_smoke.py',
    'experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/gpu/native_workflow_smoke.slurm',
    'experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/gpu/run_native_workflow_smoke.sh',
)


def _new(path: Path, value: dict) -> None:
    launch._new_json(path, value)


def _observation(freeze: dict) -> dict:
    cases = freeze['cases']
    if len(cases) != 4 or cases[0]['case_id'] != 'averitec-dev-0023' or any(case['role'] != 'development' for case in cases):
        raise ValueError('native_smoke_development_case_set')
    reference = workflow.run_episode(workflow.OracleController(),
                    workflow.FaultTools(workflow.ReplayTools(cases[0]['records']), 'nominal'))
    first = reference['events'][0]
    return first['observation']


def _historical_report(path: Path, observation: dict) -> tuple[dict, dict]:
    old = workflow.read(path)
    if not isinstance(old, dict) or not isinstance(old.get('checks'), list):
        raise ValueError('native_smoke_historical_preflight_shape')
    key = workflow.object_digest([observation, list(workflow.ACTIONS)])
    matching = [row for row in old['checks'] if row.get('input_sha256') == key]
    if len(matching) != 1 or old['checks'][-1].get('warmup') is not True:
        raise ValueError('native_smoke_preflight_state_join')
    selected = {k: v for k, v in matching[0].items() if k != 'input_sha256'}
    warmup = {k: v for k, v in old['checks'][-1].items() if k != 'warmup'}
    if selected.get('lossless') is not True or warmup.get('lossless') is not True:
        raise ValueError('native_smoke_historical_preflight_loss')
    return selected, warmup


def check_budget_transfer(path: Path) -> dict:
    value = workflow.read(path)
    required = {'schema', 'as_of_utc', 'transfers_from_workflow',
                'smoke_and_infrastructure_before', 'smoke_and_infrastructure_after',
                'new_smoke_gpu_seconds', 'maximum_gpu_seconds_total', 'maximum_gpu_jobs',
                'prior_gpu_jobs', 'source_records_sha256'}
    if (not isinstance(value, dict) or set(value) != required
            or value['schema'] != 'averitec-native-workflow-smoke-budget-transfer/v1'
            or not isinstance(value['as_of_utc'], str) or not value['as_of_utc']
            or not isinstance(value['transfers_from_workflow'], dict)
            or not value['transfers_from_workflow']
            or not set(value['transfers_from_workflow']) <= set(TRANSFER_HEADROOM)
            or any(type(amount) is not int or amount <= 0 or amount > TRANSFER_HEADROOM[name]
                   for name, amount in value['transfers_from_workflow'].items())
            or sum(value['transfers_from_workflow'].values()) != 960
            or value['smoke_and_infrastructure_before'] != 12000
            or value['smoke_and_infrastructure_after'] != 12960
            or value['new_smoke_gpu_seconds'] != 960
            or value['maximum_gpu_seconds_total'] != 91140
            or value['maximum_gpu_jobs'] != 48
            or type(value['prior_gpu_jobs']) is not int or value['prior_gpu_jobs'] < 0
            or value['prior_gpu_jobs'] + 2 > 48
            or not isinstance(value['source_records_sha256'], dict) or not value['source_records_sha256']
            or any(not workflow.HEX.fullmatch(str(digest)) for digest in value['source_records_sha256'].values())):
        raise ValueError('native_smoke_budget_transfer_invalid')
    return value


def _copy_source(destination: Path, freeze: dict, full_manifest: dict) -> dict[str, str]:
    blobs = launch._source_files(freeze, full_manifest)
    for name in EXTRA_SOURCE:
        blobs[name] = launch._committed_file(name)
    for name, blob in blobs.items():
        launch._new_bytes(destination / name, blob)
    return {name: workflow.sha(destination / name) for name in blobs}


def prepare(bundle: Path, full_manifest_path: Path, asset_index_path: Path, dev_freeze_path: Path,
            dev_preparation_path: Path, dev_preflights: dict[str, Path], budget_transfer_path: Path) -> dict:
    if (bundle.exists() or bundle.parent != launch.VEGA_BASE
            or not re.fullmatch(r'native-workflow-smoke-[a-z0-9-]+', bundle.name)):
        raise ValueError('native_smoke_bundle_namespace')
    if set(dev_preflights) != set(workflow.ARM_INFO):
        raise ValueError('native_smoke_preflight_paths')
    check_budget_transfer(budget_transfer_path)
    full = workflow.read(full_manifest_path)
    if (full['source_commit'] != workflow.commit() or workflow.sha(dev_freeze_path) != DEV_FREEZE_SHA
            or workflow.sha(dev_preparation_path) != DEV_PREPARATION_SHA
            or full.get('preflight_sha256') != workflow.RECONSTRUCTED_PREFLIGHT_SHA256
            or full.get('historical_receipts') != workflow.HISTORICAL_RECEIPT_SHA256):
        raise ValueError('native_smoke_source_or_development_drift')
    if dev_freeze_path.parent != dev_preparation_path.parent:
        raise ValueError('native_smoke_preparation_location')
    index = launch.asset_index(asset_index_path)
    if launch._hash_index(index) != (full['assets'], full['historical_receipts']):
        raise ValueError('native_smoke_assets_drift')
    freeze = workflow.read(dev_freeze_path)
    observation = _observation(freeze)
    reports = {}
    for arm, path in dev_preflights.items():
        if workflow.sha(path) != DEV_PREFLIGHT_SHA[arm]:
            raise ValueError('native_smoke_historical_report_drift')
        selected, warmup = _historical_report(path, observation)
        for row in (selected, warmup):
            tokens = row.get('total_tokens', row.get('input_tokens'))
            if type(tokens) is not int or tokens < 0 or tokens > 1000 or tokens > (1024 if arm == 'laya_typed' else 4096):
                raise ValueError('native_smoke_historical_token_cap')
        reports[arm] = {'selected': selected, 'warmup': warmup}
    bundle.mkdir(mode=0o700)
    for name in ('source', 'inputs', 'smokes', 'logs', 'submissions'):
        (bundle / name).mkdir()
    launch._new_bytes(bundle / 'source/SOURCE_COMMIT', (full['source_commit'] + '\n').encode())
    source_hashes = _copy_source(bundle / 'source', freeze, full)
    for name, expected in full['code_sha256'].items():
        staged = bundle / 'source' / 'experiments/orchestration/averitec-controller-reliability-2026' / name
        if workflow.sha(staged) != expected:
            raise ValueError('native_smoke_full_code_drift:' + name)
    import yaml
    cfg = yaml.safe_load((workflow.EXP / 'config.yaml').read_text())
    selection_sources = {}
    for relative in (cfg['dataset_refs'][0], cfg['source_split_manifest']):
        source = REPO / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError('native_smoke_selection_source_missing')
        launch._new_bytes(bundle / 'source' / relative, source.read_bytes())
        selection_sources[relative] = workflow.sha(bundle / 'source' / relative)
    staged_index = deepcopy(index)
    inputs = {'full-manifest.json': full_manifest_path.read_bytes(),
              'development-freeze.json': dev_freeze_path.read_bytes(),
              dev_preparation_path.name: dev_preparation_path.read_bytes(),
              'budget-transfer.json': budget_transfer_path.read_bytes()}
    for arm, path in dev_preflights.items():
        inputs[f'development-preflight-{arm}.json'] = path.read_bytes()
    for arm, row in index['receipts'].items():
        for name, path in row.items():
            target = f'historical-{arm}-{name}-receipt.json'
            inputs[target] = Path(path).read_bytes()
            staged_index['receipts'][arm][name] = str(bundle / 'inputs' / target)
    for arm, path in index['historical_preflights'].items():
        target = f'evaluation-preflight-{arm}.json'
        inputs[target] = Path(path).read_bytes()
        staged_index['historical_preflights'][arm] = str(bundle / 'inputs' / target)
    inputs['asset-index.json'] = (workflow.canonical(staged_index) + '\n').encode()
    for name, blob in inputs.items():
        launch._new_bytes(bundle / 'inputs' / name, blob)
    smoke_manifest = {'schema': SCHEMA, 'source_commit': full['source_commit'],
                      'full_manifest_sha256': workflow.sha(bundle / 'inputs/full-manifest.json'),
                      'asset_index_sha256': workflow.sha(bundle / 'inputs/asset-index.json'),
                      'budget_transfer_sha256': workflow.sha(bundle / 'inputs/budget-transfer.json'),
                      'development_freeze_sha256': DEV_FREEZE_SHA,
                      'development_preparation_sha256': DEV_PREPARATION_SHA,
                      'case_id': 'averitec-dev-0023', 'scenario': 'nominal',
                      'observation_sha256': workflow.object_digest(observation),
                      'warmup_observation_sha256': workflow.object_digest(workflow.WARMUP_OBSERVATION),
                      'instructions_sha256': hashlib.sha256(workflow.INSTRUCTIONS.encode()).hexdigest(),
                      'actions': list(workflow.ACTIONS),
                      'historical_development_preflight_sha256': DEV_PREFLIGHT_SHA,
                      'preflight_reports': reports,
                      'preflight_manifest_sha256': full['preflight_sha256'],
                      'source_sha256': source_hashes,
                      'caps': {arm: {'gpu_seconds': 480, 'wall_seconds': 480, 'gpu_jobs': 1,
                                     'requests': 1, 'input_tokens_per_request': 1000,
                                     'context_tokens': 1024 if arm == 'laya_typed' else 4096,
                                     'output_tokens': 0} for arm in workflow.ARM_INFO},
                      'output_namespaces': {arm: f'{arm}-development-smoke' for arm in workflow.ARM_INFO}}
    _new(bundle / 'inputs/smoke-manifest.json', smoke_manifest)
    receipt = {'schema': BUNDLE_SCHEMA, 'bundle_path': str(bundle), 'source_commit': full['source_commit'],
               'smoke_manifest_sha256': workflow.sha(bundle / 'inputs/smoke-manifest.json'),
               'full_manifest_sha256': smoke_manifest['full_manifest_sha256'],
               'source_sha256': source_hashes, 'selection_sources_sha256': selection_sources,
               'inputs_sha256': {name: workflow.sha(bundle / 'inputs' / name)
                                 for name in [*inputs, 'smoke-manifest.json']},
               'model_calls': 0, 'submitted_jobs': 0, 'gold_staged': False,
               'maximum_jobs': 2, 'maximum_gpu_seconds': 960}
    _new(bundle / 'bundle-receipt.json', receipt)
    return receipt


def _check_namespace(output: Path, *, allow_output: bool = False, allow_runtime: bool = False) -> None:
    if not allow_output and output.exists():
        raise ValueError('native_smoke_output_exists')
    runtime = Path(str(output) + '.runtime')
    if not allow_runtime and runtime.exists():
        raise ValueError('native_smoke_runtime_exists')
    if allow_runtime and runtime.exists() and (not runtime.is_dir() or runtime.is_symlink()):
        raise ValueError('native_smoke_runtime_invalid')


def verify_bundle(bundle: Path, arm: str, *, allow_output: bool = False,
                  allow_runtime: bool = False) -> tuple[dict, dict, dict, Path]:
    if arm not in workflow.ARM_INFO or not bundle.is_dir() or bundle.is_symlink():
        raise ValueError('native_smoke_bundle_or_arm')
    receipt = workflow.read(bundle / 'bundle-receipt.json')
    manifest = workflow.read(bundle / 'inputs/smoke-manifest.json')
    full = workflow.read(bundle / 'inputs/full-manifest.json')
    index = launch.asset_index(bundle / 'inputs/asset-index.json')
    if (receipt.get('schema') != BUNDLE_SCHEMA or receipt.get('bundle_path') != str(bundle)
            or receipt.get('gold_staged') is not False or receipt.get('model_calls') != 0
            or receipt.get('smoke_manifest_sha256') != workflow.sha(bundle / 'inputs/smoke-manifest.json')
            or manifest.get('schema') != SCHEMA or manifest.get('source_commit') != full['source_commit']
            or manifest.get('full_manifest_sha256') != workflow.sha(bundle / 'inputs/full-manifest.json')
            or manifest.get('asset_index_sha256') != workflow.sha(bundle / 'inputs/asset-index.json')
            or manifest.get('budget_transfer_sha256') != workflow.sha(bundle / 'inputs/budget-transfer.json')
            or manifest.get('development_freeze_sha256') != DEV_FREEZE_SHA
            or manifest.get('development_preparation_sha256') != DEV_PREPARATION_SHA
            or manifest.get('historical_development_preflight_sha256') != DEV_PREFLIGHT_SHA
            or manifest.get('preflight_manifest_sha256') != full['preflight_sha256']
            or full.get('preflight_sha256') != workflow.RECONSTRUCTED_PREFLIGHT_SHA256
            or manifest.get('source_sha256') != receipt.get('source_sha256')
            or manifest.get('actions') != list(workflow.ACTIONS)
            or manifest.get('warmup_observation_sha256') != workflow.object_digest(workflow.WARMUP_OBSERVATION)
            or manifest.get('instructions_sha256') != hashlib.sha256(workflow.INSTRUCTIONS.encode()).hexdigest()
            or manifest.get('output_namespaces') != {candidate: f'{candidate}-development-smoke'
                                                      for candidate in workflow.ARM_INFO}
            or manifest.get('caps', {}).get(arm) != {'gpu_seconds': 480, 'wall_seconds': 480, 'gpu_jobs': 1,
                  'requests': 1, 'input_tokens_per_request': 1000,
                  'context_tokens': 1024 if arm == 'laya_typed' else 4096, 'output_tokens': 0}
            or workflow.sha(bundle / 'inputs/development-freeze.json') != DEV_FREEZE_SHA
            or workflow.sha(bundle / 'inputs/qualification-dev4.json.preparation.traces.jsonl') != DEV_PREPARATION_SHA):
        raise ValueError('native_smoke_bundle_drift')
    freeze = workflow.read(bundle / 'inputs/development-freeze.json')
    observation = _observation(freeze)
    if workflow.object_digest(observation) != manifest.get('observation_sha256'):
        raise ValueError('native_smoke_state_drift')
    for candidate in workflow.ARM_INFO:
        path = bundle / 'inputs' / f'development-preflight-{candidate}.json'
        if (workflow.sha(path) != DEV_PREFLIGHT_SHA[candidate]
                or _historical_report(path, observation) !=
                    (manifest['preflight_reports'][candidate]['selected'],
                     manifest['preflight_reports'][candidate]['warmup'])):
            raise ValueError('native_smoke_preflight_drift')
    check_budget_transfer(bundle / 'inputs/budget-transfer.json')
    if launch._hash_index(index) != (full['assets'], full['historical_receipts']):
        raise ValueError('native_smoke_asset_drift')
    for section, root in (('source_sha256', bundle / 'source'), ('selection_sources_sha256', bundle / 'source'),
                          ('inputs_sha256', bundle / 'inputs')):
        for name, expected in receipt[section].items():
            if workflow.sha(root / name) != expected:
                raise ValueError('native_smoke_member_drift:' + name)
    source_files = {p.relative_to(bundle / 'source').as_posix() for p in (bundle / 'source').rglob('*') if p.is_file()}
    if source_files != ({'SOURCE_COMMIT'} | set(receipt['source_sha256']) | set(receipt['selection_sources_sha256'])):
        raise ValueError('native_smoke_source_set')
    input_files = {p.name for p in (bundle / 'inputs').iterdir() if p.is_file()}
    if input_files != set(receipt['inputs_sha256']):
        raise ValueError('native_smoke_input_set')
    for name, expected in full['code_sha256'].items():
        staged = bundle / 'source' / 'experiments/orchestration/averitec-controller-reliability-2026' / name
        if workflow.sha(staged) != expected:
            raise ValueError('native_smoke_full_code_drift:' + name)
    if (bundle / 'source/SOURCE_COMMIT').read_text().strip() != full['source_commit']:
        raise ValueError('native_smoke_source_commit')
    output = bundle / 'smokes' / manifest['output_namespaces'][arm]
    _check_namespace(output, allow_output=allow_output, allow_runtime=allow_runtime)
    return receipt, manifest, full, output


def check_gate(bundle: Path, arm: str) -> dict:
    value = workflow.read(bundle / f'gate-{arm}.json')
    expected = {'schema': GATE_SCHEMA, 'arm': arm, 'reviewer': 'gpt-6-astra', 'decision': 'approved',
                'smoke_manifest_sha256': workflow.sha(bundle / 'inputs/smoke-manifest.json'),
                'bundle_receipt_sha256': workflow.sha(bundle / 'bundle-receipt.json'),
                'budget_transfer_sha256': workflow.sha(bundle / 'inputs/budget-transfer.json')}
    if value != expected:
        raise ValueError('native_smoke_astra_gate')
    return value


def submit(bundle: Path, arm: str) -> dict:
    _receipt, manifest, _full, output = verify_bundle(bundle, arm)
    check_gate(bundle, arm)
    script = bundle / 'source' / launch.EXT / 'gpu/native_workflow_smoke.slurm'
    intent_path = bundle / 'submissions' / f'{arm}.intent.json'
    intent = {'schema': INTENT_SCHEMA, 'arm': arm, 'bundle': str(bundle),
              'smoke_manifest_sha256': workflow.sha(bundle / 'inputs/smoke-manifest.json'),
              'gate_sha256': workflow.sha(bundle / f'gate-{arm}.json'),
              'script_sha256': workflow.sha(script), 'output': str(output),
              'gpu_seconds': 480, 'gpu_jobs': 1, 'requests': 1, 'retries': 0,
              'created_unix': time.time()}
    _new(intent_path, intent)
    command = ['sbatch', '--parsable', '--no-requeue', '--time', '00:08:00', '--chdir', str(bundle),
               '--export', f'ALL,NWS_BUNDLE={bundle},NWS_ARM={arm}',
               '--output', str(bundle / 'logs' / f'{arm}-%j.out'),
               '--error', str(bundle / 'logs' / f'{arm}-%j.err'), str(script)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    raw = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r'[1-9][0-9]*(?:;[A-Za-z0-9._-]+)?', raw):
        raise RuntimeError('native_smoke_submission_unknown_no_retry')
    accepted = {'schema': ACCEPTED_SCHEMA, 'arm': arm, 'job_id': raw.split(';', 1)[0],
                'intent_sha256': workflow.sha(intent_path), 'smoke_manifest_sha256': intent['smoke_manifest_sha256'],
                'gate_sha256': intent['gate_sha256'], 'gpu_seconds': 480}
    _new(bundle / 'submissions' / f'{arm}.accepted.json', accepted)
    return accepted


def verify_runtime(bundle: Path, arm: str, job_id: str, *, allow_output: bool = False) -> dict:
    _receipt, _manifest, _full, _output = verify_bundle(bundle, arm, allow_output=allow_output,
                                                        allow_runtime=True)
    check_gate(bundle, arm)
    intent_path = bundle / 'submissions' / f'{arm}.intent.json'
    intent = workflow.read(intent_path)
    accepted = workflow.read(bundle / 'submissions' / f'{arm}.accepted.json')
    if (intent.get('schema') != INTENT_SCHEMA or intent.get('arm') != arm or intent.get('bundle') != str(bundle)
            or intent.get('gpu_seconds') != 480 or intent.get('requests') != 1
            or intent.get('script_sha256') != workflow.sha(bundle / 'source' / launch.EXT / 'gpu/native_workflow_smoke.slurm')
            or accepted != {'schema': ACCEPTED_SCHEMA, 'arm': arm, 'job_id': job_id,
                            'intent_sha256': workflow.sha(intent_path),
                            'smoke_manifest_sha256': intent['smoke_manifest_sha256'],
                            'gate_sha256': intent['gate_sha256'], 'gpu_seconds': 480}):
        raise ValueError('native_smoke_submission_binding')
    return accepted


def collect_sacct(bundle: Path, arm: str, job_id: str) -> dict:
    verify_runtime(bundle, arm, job_id, allow_output=True)
    target = bundle / 'submissions' / f'{arm}.sacct.json'
    if target.exists():
        raise ValueError('native_smoke_sacct_already_recorded')
    command = ['sacct', '-n', '-P', '-j', job_id, '--format', 'JobID,ElapsedRaw,AllocTRES,State']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError('native_smoke_sacct_unavailable')
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    roots = [row for row in rows if len(row) == 4 and row[0] == job_id]
    if len(roots) != 1 or not roots[0][1].isdigit():
        raise ValueError('native_smoke_sacct_job_missing')
    state = roots[0][3].split()[0].split('+')[0]
    if state not in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}:
        raise ValueError('native_smoke_sacct_not_terminal')
    receipt = {'schema': 'averitec-native-workflow-smoke-sacct/v1', 'arm': arm,
               'job_id': job_id, 'elapsed_raw_seconds': int(roots[0][1]),
               'alloc_tres': roots[0][2], 'state': state,
               'accepted_sha256': workflow.sha(bundle / 'submissions' / f'{arm}.accepted.json'),
               'raw_lines': result.stdout.splitlines()}
    _new(target, receipt)
    return receipt


def execute(bundle: Path, arm: str) -> dict:
    _receipt, manifest, full, output = verify_bundle(bundle, arm, allow_runtime=True)
    check_gate(bundle, arm)
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('native_smoke_slurm_job_required')
    runtime = Path(str(output) + '.runtime')
    if not runtime.is_dir() or runtime.is_symlink():
        raise ValueError('native_smoke_wrapper_runtime_required')
    verify_runtime(bundle, arm, os.environ['SLURM_JOB_ID'])
    index = launch.asset_index(bundle / 'inputs/asset-index.json')
    freeze = workflow.read(bundle / 'inputs/development-freeze.json')
    workflow.check_freeze(freeze, freeze['selection'])
    workflow.verify_preparation_traces(freeze, bundle / 'inputs')
    observation = _observation(freeze)
    if workflow.object_digest(observation) != manifest['observation_sha256']:
        raise ValueError('native_smoke_observation_drift')
    report = manifest['preflight_reports'][arm]
    if _historical_report(bundle / 'inputs' / f'development-preflight-{arm}.json', observation) != (report['selected'], report['warmup']):
        raise ValueError('native_smoke_preflight_manifest_drift')
    args = argparse.Namespace(arm=arm, checkpoint=Path(index['arms'][arm]['checkpoint']),
        asset_manifest=Path(index['arms'][arm]['asset_manifest']),
        image=Path(index['arms'][arm]['image']), runtime=Path(index['arms'][arm]['runtime']),
        wheel=Path(index['arms'][arm]['wheel']) if arm == 'laya_typed' else None,
        qualification_receipt=Path(index['receipts'][arm]['qualification']),
        selection_receipt=Path(index['receipts'][arm]['selection']) if arm == 'laya_typed' else None)
    workflow.verify_staged(args, full)
    output.mkdir(mode=0o700)
    workflow.fsync_directory(output.parent)
    intent = {'schema': 'averitec-native-workflow-development-smoke-intent/v1', 'arm': arm,
              'smoke_manifest_sha256': workflow.sha(bundle / 'inputs/smoke-manifest.json'),
              'gate_sha256': workflow.sha(bundle / f'gate-{arm}.json'),
              'full_manifest_sha256': workflow.sha(bundle / 'inputs/full-manifest.json'),
              'observation_sha256': manifest['observation_sha256'],
              'source_commit': manifest['source_commit'],
              'created_utc': datetime.now(timezone.utc).isoformat()}
    workflow.write_new(output / 'intent.json', intent)
    deadline = time.monotonic() + 480
    controller = None
    calls = 0
    error = None
    response = None
    old_term, old_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('native_smoke_signal_' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        controller = workflow._controller(arm, args.checkpoint)
        identity = controller.identity()
        if (identity.get('model') != workflow.ARM_INFO[arm][0]
                or identity.get('instruction_profile') != workflow.ARM_INFO[arm][2]
                or identity.get('expected_returned_model', identity.get('returned_model')) != workflow.ARM_INFO[arm][1]):
            raise ValueError('native_smoke_runtime_identity')
        current = controller.preflight(observation, list(workflow.ACTIONS))
        warmup = controller.preflight(workflow.WARMUP_OBSERVATION, list(workflow.ACTIONS))
        if current != report['selected'] or warmup != report['warmup'] or current.get('lossless') is not True or warmup.get('lossless') is not True:
            raise ValueError('native_smoke_live_preflight_drift')
        for row in (current, warmup):
            input_tokens = row.get('total_tokens', row.get('input_tokens'))
            if (type(input_tokens) is not int or input_tokens < 0
                    or input_tokens > manifest['caps'][arm]['input_tokens_per_request']
                    or input_tokens > manifest['caps'][arm]['context_tokens']):
                raise ValueError('native_smoke_token_cap')
        workflow.write_new(output / 'preflight.json', {'selected': current, 'warmup': warmup, 'identity': identity})
        if time.monotonic() >= deadline:
            raise TimeoutError('native_smoke_wall_cap')
        workflow.append(output / 'ledger.jsonl', {'kind': 'request_intent', 'arm': arm,
             'observation_sha256': manifest['observation_sha256'], 'model': workflow.ARM_INFO[arm][0],
             'smoke_manifest_sha256': intent['smoke_manifest_sha256'], 'at': time.time()})
        calls = 1
        try:
            result = controller.choose(deepcopy(observation), workflow.INSTRUCTIONS, list(workflow.ACTIONS))
        except BaseException as exc:
            workflow.append(output / 'ledger.jsonl', {'kind': 'uncertain', 'reason': type(exc).__name__, 'at': time.time()})
            raise
        if (result.outcome != 'ok' or result.action not in workflow.ACTIONS
                or result.model != workflow.ARM_INFO[arm][0]
                or result.returned_model != workflow.ARM_INFO[arm][1]
                or type(result.input_tokens) is not int or result.input_tokens < 0
                or result.input_tokens > manifest['caps'][arm]['input_tokens_per_request']
                or result.output_tokens != (None if arm == 'jeff' else 0)
                or time.monotonic() >= deadline):
            workflow.append(output / 'ledger.jsonl', {'kind': 'uncertain', 'reason': 'identity_outcome_or_cap',
                                                       'outcome': result.outcome, 'at': time.time()})
            raise ValueError('native_smoke_unknown_response')
        response = {'kind': 'response', 'action': result.action, 'outcome': result.outcome,
                    'model': result.model, 'returned_model': result.returned_model,
                    'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens,
                    'latency_ms': result.latency_ms, 'probabilities': result.probabilities,
                    'confidence': result.confidence, 'at': time.time()}
        workflow.append(output / 'ledger.jsonl', response)
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        if controller is not None:
            controller.close()
        evidence = {p.name: workflow.sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'receipt.json'}
        receipt = {'schema': RECEIPT_SCHEMA, 'arm': arm, 'phase': 'development_smoke',
                   'status': 'complete' if error is None and response is not None and calls == 1 else 'incomplete',
                   'model_calls': calls, 'evidence_sha256': evidence,
                   'smoke_manifest_sha256': intent['smoke_manifest_sha256'],
                   'full_manifest_sha256': intent['full_manifest_sha256'],
                   'gate_sha256': intent['gate_sha256'], 'error': error,
                   'created_utc': datetime.now(timezone.utc).isoformat()}
        workflow.write_new(output / 'receipt.json', receipt)
    if error is not None:
        raise RuntimeError(error)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'submit', 'verify-runtime', 'execute', 'collect-sacct'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(workflow.ARM_INFO))
    parser.add_argument('--job-id')
    for name in ('full-manifest', 'asset-index', 'development-freeze', 'development-preparation',
                 'jeff-development-preflight', 'laya-development-preflight', 'budget-transfer'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.bundle, args.full_manifest, args.asset_index, args.development_freeze,
                         args.development_preparation,
                         {'jeff': args.jeff_development_preflight,
                          'laya_typed': args.laya_development_preflight}, args.budget_transfer)
    elif args.submit:
        result = submit(args.bundle, args.arm)
    elif args.verify_runtime:
        result = verify_runtime(args.bundle, args.arm, args.job_id)
    elif args.collect_sacct:
        result = collect_sacct(args.bundle, args.arm, args.job_id)
    else:
        result = execute(args.bundle, args.arm)
    print(workflow.canonical(result))


if __name__ == '__main__':
    main()
