#!/usr/bin/env python3
"""Sealed, one-shot Jeff/Laya original-workflow pass. Planning makes no model call."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
REPO = HERE.parents[4]
for directory in (EXP, EXP / 'extensions' / 'evidence_selection'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from data import check_freeze, verify_preparation_traces  # noqa: E402
from engine import ACTIONS, INSTRUCTIONS, FaultTools, OracleController, ReplayTools, digest as object_digest, run_episode  # noqa: E402
from warmup import WARMUP_OBSERVATION  # noqa: E402

SCHEMA = 'averitec-native-workflow-three-pass/v1'
GATE_SCHEMA = 'averitec-native-workflow-astra-gate/v1'
RECEIPT_SCHEMA = 'averitec-native-workflow-pass-receipt/v1'
SCENARIOS = ('nominal', 'retrieval_timeout_once', 'retrieval_timeout_persistent')
ARM_INFO = {
    'jeff': ('knowledgator/gliformer-large-v1', 'gliformer-large-v1', 'jeff_native_workflow_v1'),
    'laya_typed': ('convaiinnovations/laya-typed-decisions', 'convaiinnovations/laya-typed-decisions', 'laya_native_v1'),
}
ARM_REVISIONS = {
    'jeff': 'd0a4e53d09cebe6bc963dd9be319d4279084bb2d',
    'laya_typed': 'c5d78730f3493e4fe16d61507ef4b78eef7318cf',
}
HISTORICAL_CORE_COMMIT = 'd6dba7c374020c118fc6d15701bb1cf2a49ac892'
HISTORICAL_PREFLIGHT_SHA256 = {
    'jeff': 'f88a1d671ea4bd0fb78ea42859cbfe5c5b2196b4c25cff863546fbef60bc1fb0',
    'laya_typed': 'cdda2b8d2e8411659c1131b5da8d213413a8a115f42ffbc20ca715c8cbbefd50',
}
RECONSTRUCTED_PREFLIGHT_SHA256 = {
    'jeff': 'fd8ba03a0274a7dd19aedde4a6baaf236df35a319a05a8961f36a03385496e57',
    'laya_typed': 'eda555a971e09dad0844e263d65ae9da9469ca36bd55fda41526a551fd63e47b',
}
HISTORICAL_RECEIPT_SHA256 = {
    'jeff': {'qualification': 'de2dfbf4a0d447a184a42e438c7d73260a966e251126805fba6ba906bd895155'},
    'laya_typed': {'selection': '423051439c77557f7607882684045eefe436b918bbc66290b8c33471fd320e2e',
                   'qualification': '9602092588ff8c504a88cb0d7d0599cfd79a623d30a317c6197ae004fb84b917'},
}
REQUIRED_CODE = (
    'engine.py', 'data.py', 'providers.py', 'prompt_variants.py', 'warmup.py',
    'extensions/jeff_workflow/adapter.py',
    'extensions/evidence_selection/jeff_native.py',
    'extensions/laya/adapter.py', 'extensions/laya/assets.py', 'extensions/laya/model-assets.json',
    'extensions/three_pass_verdict/native_workflow_passes.py',
    'extensions/three_pass_verdict/native_workflow_launch.py',
    'extensions/three_pass_verdict/gpu/native_workflow.slurm',
    'extensions/three_pass_verdict/gpu/run_native_workflow.sh',
    'generation_profiles.py',
)
HEX = re.compile(r'[0-9a-f]{64}\Z')
GOLD_NAMES = ('reference', 'gold', 'answer_key', 'evaluator')


def sha(path: Path) -> str:
    if path.is_dir():
        files = sorted(p for p in path.rglob('*') if p.is_file())
        if not files or any(p.is_symlink() for p in path.rglob('*')):
            raise ValueError('native_asset_tree_invalid')
        return hashlib.sha256(canonical([[str(p.relative_to(path)), sha(p)] for p in files]).encode()).hexdigest()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jeff_checkpoint_digest(path: Path) -> str:
    if not path.is_dir() or path.is_symlink():
        raise ValueError('native_jeff_checkpoint_path')
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('native_jeff_checkpoint_tree')
    return hashlib.sha256(canonical({'checkpoint/' + p.relative_to(path).as_posix(): sha(p)
                                     for p in files}).encode()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def commit() -> str:
    source = REPO / 'SOURCE_COMMIT'
    return source.read_text().strip() if source.exists() else subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as f:
        f.write(canonical(value) + '\n')
        f.flush()
        os.fsync(f.fileno())
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append(path: Path, value: Any) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(canonical(value) + '\n')
        f.flush()
        os.fsync(f.fileno())


def load_adapter(arm: str):
    """Unique module names prevent the historical adapter.py collision."""
    name = 'jeff_workflow' if arm == 'jeff' else 'laya'
    path = EXP / 'extensions' / name / 'adapter.py'
    module_name = '_original_workflow_' + name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError('native_adapter_unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_laya_assets(checkpoint: Path, asset_manifest: Path) -> None:
    canonical_manifest = EXP / 'extensions/laya/model-assets.json'
    if sha(asset_manifest) != sha(canonical_manifest):
        raise ValueError('native_laya_asset_manifest_drift')
    path = EXP / 'extensions/laya/assets.py'
    spec = importlib.util.spec_from_file_location('_original_workflow_laya_assets', path)
    if spec is None or spec.loader is None:
        raise ValueError('native_laya_asset_verifier_unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # typed-decisions is under root/checkpoints/<repository>/<revision>/.
    root = checkpoint.parents[3]
    if module.checkpoint_path(root, 'typed') != checkpoint:
        raise ValueError('native_laya_typed_checkpoint_path')
    module.verify_assets(root, module.load_manifest(asset_manifest))


def load_freeze(path: Path, preparation: Path, manifest: dict) -> dict:
    if sha(path) != manifest['freeze_sha256'] or sha(preparation) != manifest['preparation_sha256']:
        raise ValueError('native_freeze_hash_drift')
    if (manifest['freeze_sha256'] != 'f1604b70494b6fb8a4595676f713fd2d5c2c112623a0ace03dafd98062346f1c'
            or manifest['preparation_sha256'] != 'd69086be92790bf3872f573a457faf3a48316772bcabcd81e2e7c3c3d92fef4a'
            or manifest['freeze_content_sha256'] != '27a79cf6780275bb419dcd2ef9fec5131a4d183ae2f2d69b9a32cf5e7dd362ff'):
        raise ValueError('native_protocol_freeze_binding')
    if preparation.parent != path.parent:
        raise ValueError('native_preparation_location')
    freeze = read(path)
    # The builder runs in a newer checkout. The staged source combines the
    # committed runner with the freeze's exact historical core binding and
    # performs the full check_freeze before any model load.
    if (REPO / 'SOURCE_COMMIT').exists():
        check_freeze(freeze, freeze['selection'])
    verify_preparation_traces(freeze, path.parent)
    cases = freeze['cases']
    if (len(cases) != 100 or any(row['role'] != 'evaluation' for row in cases)
            or [row['case_id'] for row in cases] != manifest['case_ids']
            or [row['group_id'] for row in cases] != manifest['source_groups']
            or object_digest(freeze) != manifest['freeze_content_sha256']):
        raise ValueError('native_freeze_case_drift')
    return freeze


def make_plan(freeze: dict, arm: str, pass_number: int) -> dict:
    if arm not in ARM_INFO or pass_number not in (1, 2, 3):
        raise ValueError('native_arm_pass_invalid')
    trials, checkpoints = [], []
    for case in freeze['cases']:
        for scenario in SCENARIOS:
            trial = {'arm': arm, 'pass': pass_number, 'case_id': case['case_id'],
                     'group_id': case['group_id'], 'scenario': scenario, 'order': 'canonical',
                     'mode': 'checkpoints', 'phase': 'evaluation'}
            trial['trial_id'] = object_digest(trial)
            trials.append(trial)
            reference = run_episode(OracleController(), FaultTools(ReplayTools(case['records']), scenario))
            for event in reference['events']:
                checkpoint = {'trial_id': trial['trial_id'], 'step': event['step'],
                              'observation_sha256': event['observation_sha256'],
                              'expected_action': event['expected_action']}
                checkpoint['checkpoint_id'] = object_digest(checkpoint)
                checkpoints.append(checkpoint)
    if (len(trials), len(checkpoints)) != (300, 2299):
        raise ValueError('native_plan_count_drift')
    if len({row['trial_id'] for row in trials}) != 300 or len({row['checkpoint_id'] for row in checkpoints}) != 2299:
        raise ValueError('native_plan_duplicate_id')
    return {'trials': trials, 'checkpoints': checkpoints, 'plan_sha256': object_digest([trials, checkpoints])}


def job_plan(value: dict) -> dict:
    jobs = []
    for number in (1, 2, 3):
        for arm in ARM_INFO:
            key = f'{arm}:{number}'
            cap = value['caps'][key]
            jobs.append({'arm': arm, 'pass': number,
                         'output_namespace': value['output_namespaces'][key],
                         'gpu_count': 1, 'gpu_seconds': cap['gpu_seconds'],
                         'slurm_time': '00:17:00', 'requeue': False, 'retries': 0})
    return {'schema': 'averitec-native-workflow-job-plan/v1', 'jobs': jobs,
            'maximum_jobs': 6, 'maximum_gpu_seconds': 6120}


def validate_manifest(path: Path, arm: str, pass_number: int, freeze_path: Path, preparation_path: Path) -> tuple[dict, dict, dict]:
    value = read(path)
    required = {'schema', 'protocol_version', 'source_commit', 'historical_core_commit', 'code_sha256', 'freeze_sha256',
                'freeze_content_sha256', 'preparation_sha256', 'case_ids', 'source_groups',
                'scenarios', 'action_order', 'retries', 'arms', 'passes', 'assets',
                'historical_receipts', 'historical_preflight_sha256', 'preflight_sha256', 'output_namespaces', 'caps',
                'job_plan_sha256'}
    if not isinstance(value, dict) or set(value) != required or value['schema'] != SCHEMA:
        raise ValueError('native_manifest_shape')
    if (value['source_commit'] != commit() or value['historical_core_commit'] != HISTORICAL_CORE_COMMIT
            or value['protocol_version'] != 'three-pass-verdict-v1'
            or value['scenarios'] != list(SCENARIOS) or value['action_order'] != list(ACTIONS)
            or value['retries'] != 0 or value['passes'] != [1, 2, 3]
            or value['arms'] != {a: {'model': m, 'returned_model': r, 'profile': p, 'revision': ARM_REVISIONS[a]}
                                 for a, (m, r, p) in ARM_INFO.items()}):
        raise ValueError('native_manifest_policy')
    code = value['code_sha256']
    if not isinstance(code, dict) or set(code) != set(REQUIRED_CODE):
        raise ValueError('native_code_binding')
    for relative, expected in code.items():
        file = EXP / relative
        if not isinstance(expected, str) or not HEX.fullmatch(expected) or sha(file) != expected:
            raise ValueError('native_code_drift:' + relative)
    namespaces = value['output_namespaces']
    keys = {f'{a}:{p}' for a in ARM_INFO for p in (1, 2, 3)}
    if (not isinstance(namespaces, dict) or set(namespaces) != keys
            or len(set(namespaces.values())) != 6
            or any(not isinstance(v, str) or not v or '/' in v or '\\' in v or v in ('.', '..')
                   for v in namespaces.values())):
        raise ValueError('native_namespace_policy')
    caps = value['caps']
    if not isinstance(caps, dict) or set(caps) != keys:
        raise ValueError('native_caps_missing')
    for row in caps.values():
        if (not isinstance(row, dict) or set(row) != {'wall_seconds', 'gpu_seconds', 'gpu_jobs',
                                                     'requests', 'input_tokens_total', 'input_tokens_per_request',
                                                     'context_tokens'}
                or any(type(v) is not int or v <= 0 for v in row.values())
                or row['wall_seconds'] > 1020 or row['gpu_seconds'] != 1020
                or row['wall_seconds'] > row['gpu_seconds'] or row['gpu_seconds'] % 60 != 0
                or row['gpu_jobs'] != 1 or row['requests'] != 2300
                or row['input_tokens_total'] < row['input_tokens_per_request']):
            raise ValueError('native_caps_invalid')
    if (sum(row['gpu_seconds'] for row in caps.values()) > 6120
            or value['job_plan_sha256'] != object_digest(job_plan(value))):
        raise ValueError('native_job_plan_binding')
    for name in ('assets', 'historical_receipts', 'historical_preflight_sha256', 'preflight_sha256'):
        section = value[name]
        if not isinstance(section, dict) or set(section) != set(ARM_INFO):
            raise ValueError('native_section_missing:' + name)
    for row in value['preflight_sha256'].values():
        if not isinstance(row, str) or not HEX.fullmatch(row):
            raise ValueError('native_preflight_binding')
    for row in value['historical_preflight_sha256'].values():
        if not isinstance(row, str) or not HEX.fullmatch(row):
            raise ValueError('native_historical_preflight_binding')
    if (value['historical_preflight_sha256'] != HISTORICAL_PREFLIGHT_SHA256
            or value['historical_receipts'] != HISTORICAL_RECEIPT_SHA256
            or any(value['preflight_sha256'][arm] not in ('0' * 64, RECONSTRUCTED_PREFLIGHT_SHA256[arm])
                   for arm in ARM_INFO)):
        raise ValueError('native_historical_evidence_binding')
    for arm_name in ARM_INFO:
        assets, receipts = value['assets'][arm_name], value['historical_receipts'][arm_name]
        asset_keys = ({'checkpoint', 'asset_manifest', 'image', 'runtime'} if arm_name == 'jeff'
                      else {'checkpoint', 'asset_manifest', 'wheel', 'image', 'runtime'})
        if (not isinstance(assets, dict) or set(assets) != asset_keys
                or not isinstance(receipts, dict) or not receipts):
            raise ValueError('native_asset_receipt_binding')
        required_receipts = {'qualification'} if arm_name == 'jeff' else {'selection', 'qualification'}
        if set(receipts) != required_receipts or any(not HEX.fullmatch(str(v)) for v in (*assets.values(), *receipts.values())):
            raise ValueError('native_asset_receipt_binding')
    if (value['assets']['jeff']['asset_manifest'] != '4fafa412268c169f1cfb8c56bb29bea774fa95947d60e0aa6cd36678740af9e3'
            or value['assets']['laya_typed']['wheel'] != '0f0fed09d04e0b9e05a54643e77c6bebb9e881bf3c8e0a5c536b58c71d6659fb'):
        raise ValueError('native_historical_asset_binding')
    freeze = load_freeze(freeze_path, preparation_path, value)
    plan = make_plan(freeze, arm, pass_number)
    return value, freeze, plan


def check_gate(path: Path, manifest_path: Path, arm: str, pass_number: int) -> None:
    gate = read(path)
    if gate != {'schema': GATE_SCHEMA, 'manifest_sha256': sha(manifest_path),
                'decision': 'approved', 'reviewer': 'gpt-6-astra',
                'arm': arm, 'pass': pass_number}:
        raise ValueError('native_astra_gate_required')


def check_file(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or not HEX.fullmatch(expected) or sha(path) != expected:
        raise ValueError('native_asset_or_receipt_drift:' + str(path))


def verify_staged(args: argparse.Namespace, value: dict) -> None:
    assets = {'checkpoint': args.checkpoint, 'asset_manifest': args.asset_manifest,
              'wheel': args.wheel, 'image': args.image, 'runtime': args.runtime}
    for name, expected in value['assets'][args.arm].items():
        if assets[name] is None:
            raise ValueError('native_asset_paths_incomplete')
        check_file(assets[name], expected)
    if args.arm == 'jeff' and jeff_checkpoint_digest(args.checkpoint) != '64bc03d73b133461134468ad22b5ff2315579419d534018561742624a1a262df':
        raise ValueError('native_jeff_historical_checkpoint_drift')
    if args.arm == 'laya_typed':
        verify_laya_assets(args.checkpoint, args.asset_manifest)
    receipts = {'qualification': args.qualification_receipt, 'selection': args.selection_receipt}
    if set(value['historical_receipts'][args.arm]) != {k for k, v in receipts.items() if v is not None}:
        raise ValueError('native_receipt_paths_incomplete')
    for name, expected in value['historical_receipts'][args.arm].items():
        check_file(receipts[name], expected)
        historical = read(receipts[name])
        phase = 'screen' if name == 'selection' else 'qualification'
        if historical.get('status') != 'complete' or historical.get('phase') != phase:
            raise ValueError('native_historical_receipt_incomplete')
    if args.arm == 'laya_typed':
        selected = read(args.selection_receipt)
        qualified = read(args.qualification_receipt)
        if (selected.get('selected_checkpoint') != 'typed'
                or qualified.get('selected_checkpoint') != 'typed'
                or qualified.get('selection_receipt_sha256') != sha(args.selection_receipt)):
            raise ValueError('native_laya_historical_link')


def preflight_only(args: argparse.Namespace) -> dict:
    value, freeze, plan = validate_manifest(args.manifest, args.arm, args.pass_number, args.freeze, args.preparation)
    verify_staged(args, value)
    controller = _controller(args.arm, args.checkpoint)
    try:
        identity = controller.identity()
        if (identity.get('model') != ARM_INFO[args.arm][0]
                or identity.get('instruction_profile') != ARM_INFO[args.arm][2]
                or identity.get('expected_returned_model', identity.get('returned_model')) != ARM_INFO[args.arm][1]):
            raise ValueError('native_runtime_identity_drift')
        result = preflight(controller, plan, freeze, args.arm, value['caps'][f'{args.arm}:{args.pass_number}'])
        result['identity'] = identity
        result['manifest_sha256'] = sha(args.manifest)
        result['plan_sha256'] = plan['plan_sha256']
        if args.output is not None:
            write_new(args.output, result)
        return result
    finally:
        controller.close()


def preflight(controller: Any, plan: dict, freeze: dict, arm: str, caps: dict) -> dict:
    by_case = {row['case_id']: row for row in freeze['cases']}
    rows, seen = [], set()
    for trial in plan['trials']:
        case = by_case[trial['case_id']]
        reference = run_episode(OracleController(), FaultTools(ReplayTools(case['records']), trial['scenario']))
        for event in reference['events']:
            key = event['observation_sha256']
            if key in seen:
                continue
            report = controller.preflight(event['observation'], list(ACTIONS))
            if report.get('lossless') is not True:
                raise ValueError('native_preflight_loss')
            token_count = report.get('total_tokens', report.get('input_tokens'))
            if (type(token_count) is not int or token_count < 0
                    or token_count > caps['input_tokens_per_request']
                    or token_count > caps['context_tokens']):
                raise ValueError('native_preflight_token_cap')
            rows.append({'observation_sha256': key, 'report': report})
            seen.add(key)
    warm = controller.preflight(WARMUP_OBSERVATION, list(ACTIONS))
    if warm.get('lossless') is not True:
        raise ValueError('native_warmup_preflight_loss')
    warm_tokens = warm.get('total_tokens', warm.get('input_tokens'))
    if (type(warm_tokens) is not int or warm_tokens < 0
            or warm_tokens > caps['input_tokens_per_request']
            or warm_tokens > caps['context_tokens']):
        raise ValueError('native_warmup_token_cap')
    rows.append({'warmup': True, 'observation_sha256': object_digest(WARMUP_OBSERVATION), 'report': warm})
    return {'arm': arm, 'profile': ARM_INFO[arm][2], 'checks': rows,
            'sha256': object_digest(rows)}


def _controller(arm: str, checkpoint: Path):
    adapter = load_adapter(arm)
    if arm == 'jeff':
        return adapter.NativeJeffController(checkpoint, device='cuda')
    return adapter.NativeLayaController(ARM_INFO[arm][0], checkpoint, device='cuda')


def _event(result: Any, planned: dict, observation: dict) -> dict:
    return {'checkpoint_id': planned['checkpoint_id'], 'trial_id': planned['trial_id'],
            'step': planned['step'], 'observation_sha256': planned['observation_sha256'],
            'expected_action': planned['expected_action'], 'action': result.action,
            'provider_outcome': result.outcome, 'model': result.model,
            'returned_model': result.returned_model, 'latency_ms': result.latency_ms,
            'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens,
            'error_code': result.error_code, 'probabilities': result.probabilities,
            'confidence': result.confidence,
            'compliant': result.outcome == 'ok' and result.action == planned['expected_action']}


def audit(plan: dict, entries: list[dict]) -> dict:
    planned_ids = [row['checkpoint_id'] for row in plan['checkpoints']]
    expected_sequence = ['warmup', *planned_ids]
    intents = [row['checkpoint_id'] for row in entries if row.get('kind') == 'request_intent']
    if intents != expected_sequence[:len(intents)]:
        raise ValueError('native_intent_order')
    if len(set(intents)) != len(intents):
        raise ValueError('native_duplicate_intent')
    for index, row in enumerate(entries):
        if row.get('kind') in ('response', 'warmup_response', 'uncertain'):
            if index == 0 or entries[index - 1].get('kind') != 'request_intent' or entries[index - 1]['checkpoint_id'] != row['checkpoint_id']:
                raise ValueError('native_response_without_prior_intent')
    outcomes = {row['checkpoint_id']: row for row in entries if row.get('kind') == 'response'}
    if len(outcomes) != sum(row.get('kind') == 'response' for row in entries):
        raise ValueError('native_duplicate_response')
    if not set(outcomes) <= set(planned_ids):
        raise ValueError('native_unplanned_response')
    complete = (len(outcomes) == 2299 and len(intents) == 2300
                and any(row.get('kind') == 'warmup_response' and row.get('outcome') == 'ok' for row in entries)
                and all(row['provider_outcome'] in ('ok', 'invalid_output') for row in outcomes.values()))
    trial_ids = {row['trial_id'] for row in plan['trials']}
    if {row['trial_id'] for row in outcomes.values()} - trial_ids:
        raise ValueError('native_trial_join')
    trial_recorded = {trial_id: 0 for trial_id in trial_ids}
    for row in outcomes.values():
        trial_recorded[row['trial_id']] += 1
    return {'status': 'complete' if complete else 'incomplete', 'planned_trials': 300,
            'planned_checkpoints': 2299, 'recorded_checkpoints': len(outcomes),
            'missing_checkpoints': 2299 - len(outcomes),
            'recorded_trials': sum(count > 0 for count in trial_recorded.values()),
            'missing_trials': sum(count == 0 for count in trial_recorded.values()),
            'compliant_checkpoints': sum(bool(row['compliant']) for row in outcomes.values()),
            'failed_checkpoints': 2299 - sum(bool(row['compliant']) for row in outcomes.values())}


def execute(args: argparse.Namespace) -> dict:
    value, freeze, plan = validate_manifest(args.manifest, args.arm, args.pass_number, args.freeze, args.preparation)
    check_gate(args.gate, args.manifest, args.arm, args.pass_number)
    key = f'{args.arm}:{args.pass_number}'
    if args.output.name != value['output_namespaces'][key]:
        raise ValueError('native_output_namespace_mismatch')
    if args.output.exists():
        raise ValueError('native_output_already_exists_no_resume')
    verify_staged(args, value)
    args.output.mkdir(parents=True, exist_ok=False)
    fsync_directory(args.output.parent)
    intent = {'schema': 'averitec-native-workflow-intent/v1', 'arm': args.arm, 'pass': args.pass_number,
              'manifest_sha256': sha(args.manifest), 'gate_sha256': sha(args.gate),
              'plan_sha256': plan['plan_sha256'], 'created_utc': datetime.now(timezone.utc).isoformat(),
              'namespace': args.output.name}
    write_new(args.output / 'intent.json', intent)
    write_new(args.output / 'plan.json', plan)
    ledger = args.output / 'ledger.jsonl'
    deadline = time.monotonic() + value['caps'][key]['wall_seconds']
    controller = None
    entries: list[dict] = []
    error = None
    tokens = 0
    calls = 0
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('native_signal_' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        controller = _controller(args.arm, args.checkpoint)
        identity = controller.identity()
        if (identity.get('model') != ARM_INFO[args.arm][0]
                or identity.get('instruction_profile') != ARM_INFO[args.arm][2]
                or identity.get('expected_returned_model', identity.get('returned_model')) != ARM_INFO[args.arm][1]):
            raise ValueError('native_runtime_identity_drift')
        report = preflight(controller, plan, freeze, args.arm, value['caps'][key])
        if report['sha256'] != value['preflight_sha256'][args.arm]:
            raise ValueError('native_preflight_manifest_drift')
        write_new(args.output / 'preflight.json', report)
        write_new(args.output / 'identity.json', identity)
        by_case = {row['case_id']: row for row in freeze['cases']}
        observations = {}
        for trial in plan['trials']:
            reference = run_episode(OracleController(), FaultTools(ReplayTools(by_case[trial['case_id']]['records']), trial['scenario']))
            observations.update({object_digest({'trial_id': trial['trial_id'], 'step': e['step'],
                'observation_sha256': e['observation_sha256'], 'expected_action': e['expected_action']}): e['observation'] for e in reference['events']})
        sequence = [{'checkpoint_id': 'warmup', 'observation_sha256': object_digest(WARMUP_OBSERVATION)}] + plan['checkpoints']
        caps = value['caps'][key]
        for item in sequence:
            if time.monotonic() >= deadline or calls >= caps['requests']:
                raise TimeoutError('native_pass_cap')
            if sha(args.manifest) != intent['manifest_sha256']:
                raise ValueError('native_manifest_changed_during_pass')
            obs = WARMUP_OBSERVATION if item['checkpoint_id'] == 'warmup' else observations[item['checkpoint_id']]
            request = {'kind': 'request_intent', 'checkpoint_id': item['checkpoint_id'],
                       'observation_sha256': item['observation_sha256'], 'model': ARM_INFO[args.arm][0],
                       'source_manifest_sha256': intent['manifest_sha256'], 'at': time.time()}
            append(ledger, request)
            entries.append(request)
            calls += 1
            try:
                result = controller.choose(deepcopy(obs), INSTRUCTIONS, list(ACTIONS))
            except BaseException as exc:
                uncertain = {'kind': 'uncertain', 'checkpoint_id': item['checkpoint_id'],
                             'reason': type(exc).__name__, 'at': time.time()}
                append(ledger, uncertain)
                entries.append(uncertain)
                raise
            if result.model != ARM_INFO[args.arm][0] or result.returned_model != ARM_INFO[args.arm][1] or result.outcome not in ('ok', 'invalid_output'):
                uncertain = {'kind': 'uncertain', 'checkpoint_id': item['checkpoint_id'],
                             'reason': 'identity_or_outcome', 'outcome': result.outcome, 'at': time.time()}
                append(ledger, uncertain)
                entries.append(uncertain)
                raise ValueError('native_uncertain_result')
            token_count = result.input_tokens
            if token_count is not None:
                if type(token_count) is not int or token_count < 0 or token_count > caps['input_tokens_per_request']:
                    uncertain = {'kind': 'uncertain', 'checkpoint_id': item['checkpoint_id'], 'reason': 'token_cap', 'at': time.time()}
                    append(ledger, uncertain)
                    entries.append(uncertain)
                    raise ValueError('native_token_cap')
                tokens += token_count
                if tokens > caps['input_tokens_total']:
                    uncertain = {'kind': 'uncertain', 'checkpoint_id': item['checkpoint_id'], 'reason': 'total_token_cap', 'at': time.time()}
                    append(ledger, uncertain)
                    entries.append(uncertain)
                    raise ValueError('native_total_token_cap')
            outcome = ({'kind': 'warmup_response', 'checkpoint_id': 'warmup', 'outcome': result.outcome,
                        'action': result.action, 'returned_model': result.returned_model,
                        'latency_ms': result.latency_ms, 'input_tokens': result.input_tokens,
                        'probabilities': result.probabilities} if item['checkpoint_id'] == 'warmup'
                       else {'kind': 'response', **_event(result, item, obs)})
            append(ledger, outcome)
            entries.append(outcome)
            if item['checkpoint_id'] == 'warmup' and result.outcome != 'ok':
                raise ValueError('native_warmup_failed')
            if time.monotonic() >= deadline:
                raise TimeoutError('native_pass_wall_cap')
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        if controller is not None:
            controller.close()
        summary = audit(plan, entries)
        if error is not None:
            summary['status'] = 'incomplete'
        receipt = {'schema': RECEIPT_SCHEMA, 'arm': args.arm, 'pass': args.pass_number,
                   'manifest_sha256': sha(args.manifest), 'gate_sha256': sha(args.gate),
                   'plan_sha256': plan['plan_sha256'], 'summary': summary,
                   'request_intents': calls, 'input_tokens_reported': tokens, 'error': error,
                   'ledger_sha256': sha(ledger) if ledger.exists() else None,
                   'created_utc': datetime.now(timezone.utc).isoformat()}
        write_new(args.output / 'receipt.json', receipt)
    if error:
        raise RuntimeError(error)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(ARM_INFO), required=True)
    parser.add_argument('--pass-number', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--asset-manifest', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--qualification-receipt', type=Path)
    parser.add_argument('--selection-receipt', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='load assets and prove every request fits; no forward pass')
    args = parser.parse_args()
    if args.plan and args.preflight:
        parser.error('choose plan or preflight')
    if args.plan:
        value, _, plan = validate_manifest(args.manifest, args.arm, args.pass_number, args.freeze, args.preparation)
        print(canonical({'arm': args.arm, 'pass': args.pass_number, 'trials': len(plan['trials']),
                         'checkpoints': len(plan['checkpoints']), 'plan_sha256': plan['plan_sha256'],
                         'namespace': value['output_namespaces'][f'{args.arm}:{args.pass_number}']}))
    elif args.preflight:
        if args.checkpoint is None or args.qualification_receipt is None:
            parser.error('preflight requires staged assets and historical receipts')
        print(canonical(preflight_only(args)))
    else:
        if args.gate is None or args.output is None or args.checkpoint is None or args.qualification_receipt is None:
            parser.error('execute requires gate, output, checkpoint, and historical qualification receipt')
        print(canonical(execute(args)))


if __name__ == '__main__':
    main()
