"""One-shot, gated hosted workflow recovery. No provider work in --plan/--check.

The original run.execute owns experiment semantics. This wrapper owns the fresh
request ledger and stops the run on the first ambiguous provider outcome.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[6]
CORE = ROOT / 'experiments/orchestration/averitec-controller-reliability-2026'
THREE = CORE / 'extensions/three_pass_verdict'
sys.path.insert(0, str(CORE))
import run as core_run  # noqa: E402
import providers  # noqa: E402
import gemini_provider  # noqa: E402
from data import code_binding, file_hash  # noqa: E402
from engine import ACTIONS, SCENARIOS, FaultTools, OracleController, ReplayTools, digest, run_episode  # noqa: E402
from evaluation_plan import CONDITIONS  # noqa: E402
from prompt_variants import PresentedController, instructions_for  # noqa: E402
from warmup import premeasure_warmup  # noqa: E402

SCHEMA = 'three-pass-hosted-workflow-recovery/v1'
SMOKE_SCHEMA = 'three-pass-hosted-workflow-recovery-smoke/v1'
GATE_SCHEMA = 'three-pass-hosted-workflow-recovery-astra-gate/v1'
OLD_LEDGER = THREE / 'hosted_budget.py'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('json_object_required')
    return value


def spec(value: dict) -> Path:
    if set(value) != {'path', 'sha256'} or not Path(value['path']).is_absolute():
        raise ValueError('absolute_file_spec_required')
    path = Path(value['path'])
    if not path.is_file() or sha(path) != value['sha256']:
        raise ValueError('file_hash_drift:' + str(path))
    return path


def new_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as target:
            target.write(data)
            target.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def append(path: Path, value: dict) -> None:
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())


def old_pending(path: Path) -> int:
    if not path.is_file():
        raise ValueError('original_hosted_ledger_missing')
    uri = 'file:' + str(path) + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        return int(db.execute("SELECT COUNT(*) FROM calls WHERE status='pending'").fetchone()[0])


def plan(freeze: dict, arm: str) -> dict:
    profile = CONDITIONS[arm]
    args = argparse.Namespace(role='evaluation', limit=100, controller=arm,
        scenarios=list(SCENARIOS), orders=['canonical'], repetitions=1,
        mode='checkpoints', prompt_only=False, prompt_variant=profile[0],
        generation_profile=profile[1], live_tools=False)
    tasks = core_run.planned_tasks(args, freeze)
    by_id = {row['case_id']: row for row in freeze['cases']}
    checkpoints = sum(len(run_episode(OracleController(), FaultTools(
        ReplayTools(by_id[task['case_id']]['records']), task['scenario']))['events']) for task in tasks)
    result = {'tasks': len(tasks), 'checkpoints': checkpoints,
              'case_ids': [row['case_id'] for row in freeze['cases']],
              'group_ids': [row['group_id'] for row in freeze['cases']],
              'task_ids': [row['trial_id'] for row in tasks]}
    if len(tasks) != 300 or checkpoints != 2299 or len(result['case_ids']) != 100:
        raise ValueError('recovery_plan_count_drift')
    result['plan_sha256'] = digest(result)
    return result


def validate(manifest_path: Path, *, smoke: bool) -> tuple[dict, dict | None]:
    m = read(manifest_path)
    if m.get('schema') != (SMOKE_SCHEMA if smoke else SCHEMA) or m.get('cohort') != 'recovery-1':
        raise ValueError('recovery_manifest_schema')
    arm = m.get('arm')
    if arm not in ('jev', 'gemini') or m.get('profile') != list(CONDITIONS[arm]):
        raise ValueError('original_profile_drift')
    if not isinstance(m.get('output'), str) or not Path(m['output']).is_absolute():
        raise ValueError('absolute_output_required')
    if m.get('source_commit') != subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip():
        raise ValueError('source_commit_drift')
    expected_code = code_binding()
    expected_code[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    if m.get('code_sha256') != expected_code:
        raise ValueError('code_binding_drift')
    if old_pending(spec(m['original_ledger'])):
        raise ValueError('original_hosted_ledger_has_pending_requests')
    for name in ('original_receipt', 'preparation'):
        spec(m[name])
    limits = m.get('limits', {})
    keys = ('maximum_total_generation_requests', 'maximum_pass_generation_requests',
            'maximum_measured_generation_requests', 'maximum_warmup_generation_requests',
            'maximum_smoke_generation_requests', 'maximum_input_tokens',
            'maximum_output_tokens', 'maximum_wall_seconds')
    if any(type(limits.get(k)) is not int or limits[k] < 1 for k in keys):
        raise ValueError('numeric_limits_required')
    if [limits[k] for k in keys[:5]] != [2301, 2300, 2299, 1, 1]:
        raise ValueError('generation_caps_drift')
    if smoke:
        return m, None
    freeze_path = spec(m['freeze'])
    freeze = read(freeze_path)
    if freeze.get('code_binding') != code_binding():
        raise ValueError('historical_freeze_code_binding_drift')
    spec(m['qualification_receipt'])
    expected = plan(freeze, arm)
    if any(m.get(k) != expected[k] for k in ('plan_sha256', 'case_ids', 'group_ids')):
        raise ValueError('frozen_plan_drift')
    smoke_receipt = read(spec(m['smoke_receipt']))
    if (any(type(smoke_receipt.get(k)) is not int or smoke_receipt[k] < 0
            for k in ('input_tokens', 'output_tokens'))
            or type(smoke_receipt.get('elapsed_seconds')) not in (int, float)
            or smoke_receipt['elapsed_seconds'] < 0):
        raise ValueError('smoke_usage_missing')
    if (smoke_receipt.get('schema') != 'three-pass-hosted-workflow-recovery-smoke-receipt/v1'
            or smoke_receipt.get('arm') != arm or smoke_receipt.get('status') != 'complete'
            or smoke_receipt.get('generation_requests') != 1):
        raise ValueError('smoke_receipt_not_complete')
    return m, freeze


def gate(manifest_path: Path, gate_path: Path, m: dict, *, smoke: bool) -> None:
    g = read(gate_path)
    if (g.get('schema') != GATE_SCHEMA or g.get('decision') != 'approved'
            or g.get('reviewer') != 'gpt-6-astra' or g.get('manifest_sha256') != sha(manifest_path)
            or g.get('arm') != m['arm'] or g.get('cohort') != 'recovery-1'
            or g.get('phase') != ('smoke' if smoke else 'pass')):
        raise ValueError('astra_gate_required')


class StopRun(BaseException):
    pass


@contextmanager
def metered_http(arm: str, ledger: Path, limits: dict, *, phase: str, smoke_used: int,
                 initial_input: int = 0, initial_output: int = 0, initial_wall: float = 0):
    original = providers._http_post
    counter = {'requests': 0, 'input': initial_input, 'output': initial_output, 'started': time.monotonic(), 'initial_wall': initial_wall}
    maximum = 1 if phase == 'smoke' else 2300

    def post(url: str, body: dict, headers: dict, timeout: float) -> dict:
        n = counter['requests'] + 1
        if n > maximum or n + smoke_used > limits['maximum_total_generation_requests']:
            raise StopRun('request_cap')
        if counter['initial_wall'] + time.monotonic() - counter['started'] >= limits['maximum_wall_seconds']:
            raise StopRun('wall_cap')
        if counter['input'] >= limits['maximum_input_tokens'] or counter['output'] >= limits['maximum_output_tokens']:
            raise StopRun('token_cap')
        request_id = f'{phase}-{n:04d}'
        append(ledger, {'kind': 'intent', 'request_id': request_id, 'phase': 'smoke' if phase == 'smoke' else ('warmup' if n == 1 else 'measured'),
                        'request_sha256': digest(body), 'reserved_input': limits['maximum_input_tokens'] - counter['input'],
                        'reserved_output': limits['maximum_output_tokens'] - counter['output'], 'at': time.time()})
        counter['requests'] = n
        started = time.monotonic()
        try:
            response = original(url, body, headers, timeout)
        except BaseException:
            append(ledger, {'kind': 'uncertain', 'request_id': request_id, 'reason': 'transport_or_decode',
                            'elapsed_seconds': time.monotonic() - started})
            raise StopRun('transport_or_decode') from None
        usage = response.get('usage' if arm == 'jev' else 'usageMetadata')
        if arm == 'jev':
            identity = response.get('model')
            input_tokens = usage.get('input_tokens') if isinstance(usage, dict) else None
            output_tokens = usage.get('output_tokens') if isinstance(usage, dict) else None
            expected = 'jev-1.13.0'
        else:
            identity = response.get('modelVersion')
            input_tokens, output_tokens = gemini_provider._gemini_usage(usage)
            expected = 'gemini-3.1-flash-lite'
        valid_usage = all(type(x) is int and x >= 0 for x in (input_tokens, output_tokens))
        append(ledger, {'kind': 'response', 'request_id': request_id, 'response_sha256': digest(response),
                        'returned_model': identity if isinstance(identity, str) else None,
                        'input_tokens': input_tokens, 'output_tokens': output_tokens,
                        'elapsed_seconds': time.monotonic() - started})
        if identity != expected or not valid_usage:
            raise StopRun('identity_or_usage_unknown')
        counter['input'] += input_tokens
        counter['output'] += output_tokens
        if (counter['input'] > limits['maximum_input_tokens'] or counter['output'] > limits['maximum_output_tokens']):
            raise StopRun('token_cap_exceeded')
        return response

    old_jev, old_gemini = core_run.JevController, core_run.GeminiController
    class StrictJev(old_jev):
        def choose(self, *args, **kwargs):
            result = super().choose(*args, **kwargs)
            if result.outcome != 'ok':
                raise StopRun('jev_' + result.outcome)
            return result
    class StrictGemini(old_gemini):
        def choose(self, *args, **kwargs):
            result = super().choose(*args, **kwargs)
            if result.outcome != 'ok':
                raise StopRun('gemini_' + result.outcome)
            return result
    providers._http_post = post
    gemini_provider._http_post = post
    core_run.JevController = StrictJev
    core_run.GeminiController = StrictGemini
    try:
        yield counter
    finally:
        providers._http_post = original
        gemini_provider._http_post = original
        core_run.JevController = old_jev
        core_run.GeminiController = old_gemini


def controller(arm: str, settings: dict):
    variant, profile = CONDITIONS[arm]
    if arm == 'jev':
        from prompt_variants import native_choice_criteria
        raw = providers.JevController(model=settings['models']['jev']['model'],
            timeout_seconds=30, choice_criteria=native_choice_criteria(variant))
    else:
        raw = gemini_provider.GeminiController(model=settings['models']['gemini']['model'],
            expected_version=settings['models']['gemini']['expected_version'], seed=settings['seed'],
            generation_profile=profile, timeout_seconds=30)
    return PresentedController(raw, variant)


def execute(manifest_path: Path, gate_path: Path, *, smoke: bool) -> dict:
    m, freeze = validate(manifest_path, smoke=smoke)
    gate(manifest_path, gate_path, m, smoke=smoke)
    output = Path(m['output'])
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    ledger = output / 'requests.jsonl'
    new_json(output / 'intent.json', {'manifest_sha256': sha(manifest_path), 'gate_sha256': sha(gate_path),
                                     'phase': 'smoke' if smoke else 'pass', 'arm': m['arm']})
    status, reason = 'incomplete', None
    count = None
    prior = {'input_tokens': 0, 'output_tokens': 0, 'elapsed_seconds': 0}
    started = time.monotonic()
    try:
        prior = prior if smoke else read(spec(m['smoke_receipt']))
        with metered_http(m['arm'], ledger, m['limits'], phase='smoke' if smoke else 'pass',
                          smoke_used=0 if smoke else 1, initial_input=prior['input_tokens'],
                          initial_output=prior['output_tokens'], initial_wall=prior['elapsed_seconds']) as count:
            if smoke:
                core_run.load_key('TYPESAFE_API_KEY' if m['arm'] == 'jev' else 'GEMINI_API_KEY')
                c = controller(m['arm'], core_run.config())
                premeasure_warmup(c, instructions=instructions_for(CONDITIONS[m['arm']][0]), actions=list(ACTIONS))
            else:
                args = argparse.Namespace(study_phase='evaluation', role='evaluation', limit=100, case_offset=0,
                    controller=m['arm'], scenarios=list(SCENARIOS), orders=['canonical'], repetitions=1,
                    mode='checkpoints', prompt_only=False, prompt_variant=CONDITIONS[m['arm']][0],
                    generation_profile=CONDITIONS[m['arm']][1], live_tools=False, plan=False,
                    output=output/'results.jsonl', freeze=spec(m['freeze']), endpoint=None,
                    worker_endpoint=None, qualification_receipt=spec(m['qualification_receipt']))
                rows = core_run.execute(args, core_run.config(), freeze)
                if len(rows) != 300 or count['requests'] != 2300:
                    raise StopRun('incomplete_checkpoint_count')
                if any(e.get('outcome') != 'ok' for row in rows for e in row.get('events', []) if isinstance(e, dict)):
                    raise StopRun('invalid_checkpoint_result')
            status = 'complete'
    except BaseException as exc:
        reason = str(exc) if isinstance(exc, (StopRun, ValueError)) else type(exc).__name__
    receipt = {'schema': 'three-pass-hosted-workflow-recovery-smoke-receipt/v1' if smoke else 'three-pass-hosted-workflow-recovery-receipt/v1',
               'arm': m['arm'], 'cohort': 'recovery-1', 'status': status, 'reason': reason,
               'manifest_sha256': sha(manifest_path), 'gate_sha256': sha(gate_path),
               'generation_requests': count['requests'] if count else 0,
               'input_tokens': (count['input'] - prior['input_tokens']) if count else 0,
               'output_tokens': (count['output'] - prior['output_tokens']) if count else 0,
               'elapsed_seconds': time.monotonic() - started,
               'ledger_sha256': sha(ledger) if ledger.exists() else None}
    new_json(output/'receipt.json', receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--plan', action='store_true')
    operation.add_argument('--check', action='store_true')
    operation.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    m, freeze = validate(args.manifest, smoke=args.smoke)
    if args.plan:
        print(json.dumps({'arm': m['arm'], 'cohort': 'recovery-1', 'plan': None if args.smoke else plan(freeze, m['arm']), 'model_calls': 0}, sort_keys=True))
    elif args.check:
        if args.gate:
            gate(args.manifest, args.gate, m, smoke=args.smoke)
        print(json.dumps({'manifest_sha256': sha(args.manifest), 'gate_checked': bool(args.gate), 'model_calls': 0}, sort_keys=True))
    else:
        if not args.gate:
            parser.error('--execute requires --gate')
        print(json.dumps(execute(args.manifest, args.gate, smoke=args.smoke), sort_keys=True))


if __name__ == '__main__':
    main()
