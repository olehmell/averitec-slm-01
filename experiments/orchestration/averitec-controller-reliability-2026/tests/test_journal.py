import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from data import prepare_selection


def _inputs(tmp_path):
    args = argparse.Namespace(role='development', limit=1, repetitions=1,
                              orders=['canonical'], scenarios=['nominal'], controller='oracle',
                              mode='checkpoints', prompt_only=False, live_tools=False,
                              plan=False, output=tmp_path/'rows.jsonl', worker_endpoint=None)
    settings = run.config()
    freeze = run.create_freeze(args, settings, prepare_selection(settings), synthetic=True)
    return args, settings, freeze


def test_completed_replay_is_not_reissued_and_digest_checked(tmp_path, monkeypatch):
    args, settings, freeze = _inputs(tmp_path)
    monkeypatch.setattr(run, 'export_traces', lambda *a, **k: pytest.fail('remote tracing entered measured run'))
    first = run.execute(args, settings, freeze, synthetic=True)
    trace_path = Path(str(args.output) + '.traces.jsonl')
    trace_before = trace_path.read_bytes()
    trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert {row['trace_id'] for row in trace_rows} == {first[0]['trace_id']}
    assert first[0]['preparation_trace_id'] == freeze['cases'][0]['preparation_trace_id']
    assert len([row for row in trace_rows if row['event'] == 'span_start']) == 9
    monkeypatch.setattr(run, 'replay_checkpoints', lambda *a, **k: pytest.fail('completed trial reissued'))
    second = run.execute(args, settings, freeze, synthetic=True)
    assert first == second
    assert trace_path.read_bytes() == trace_before
    assert len(run.read_complete_run(args.output)) == 1
    args.output.write_text(args.output.read_text().replace('checkpoints_evaluated', 'changed'))
    with pytest.raises(ValueError, match='result_row_digest'):
        run.execute(args, settings, freeze, synthetic=True)


def test_interrupted_intent_is_counted_without_repeat_requests(tmp_path, monkeypatch):
    args, settings, freeze = _inputs(tmp_path)
    def interrupt(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(run, 'replay_checkpoints', interrupt)
    with pytest.raises(KeyboardInterrupt):
        run.execute(args, settings, freeze, synthetic=True)
    intent = json.loads(Path(str(args.output) + '.intents.jsonl').read_text())
    with pytest.raises(ValueError, match='incomplete_run_resume_before_summary'):
        run.read_complete_run(args.output)
    # The still-failing callable must not be reached on resume.
    rows = run.execute(args, settings, freeze, synthetic=True)
    assert len(rows) == 1
    assert rows[0]['outcome'] == 'interrupted_unknown_outcome'
    assert rows[0]['trace_id'] == intent['trace_id']
    assert rows[0]['planned_decisions'] == 8
    summary = run.summarize(run.read_complete_run(args.output), bootstrap_samples=50)
    assert summary['arms'][0]['first_attempt_rule_compliance'] == 0
    assert summary['arms'][0]['interrupted_trials'] == 1


def test_resume_cannot_mix_prompt_conditions(tmp_path):
    args, settings, freeze = _inputs(tmp_path)
    run.execute(args, settings, freeze, synthetic=True)
    args.prompt_only = True
    with pytest.raises(ValueError, match='resume_identity_mismatch'):
        run.execute(args, settings, freeze, synthetic=True)


def test_preparation_trace_journal_is_a_required_bound_input(tmp_path):
    args, settings, freeze = _inputs(tmp_path)
    journal = tmp_path / freeze['preparation_trace_journal']
    journal.write_text(journal.read_text().replace('synthetic fact', 'changed fact'))
    with pytest.raises(ValueError, match='preparation_trace_journal_drift'):
        run.execute(args, settings, freeze, synthetic=True)
