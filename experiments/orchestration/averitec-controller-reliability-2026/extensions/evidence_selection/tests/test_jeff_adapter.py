from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jeff_adapter import JeffEvidenceSelector, RETURNED_MODEL

OBS = {'claim': 'A claim.', 'candidate': {'id': 'C01', 'text': 'A snippet.', 'url': 'https://example.test'}}

def payload():
    return {'model': RETURNED_MODEL, 'answers': {'next_action': {
        'type': 'choice', 'choice': 'include', 'probabilities': {'include': .8, 'exclude': .2},
        'confidence': .6}}, 'usage': {'input_tokens': 20, 'output_tokens': 6}}

def test_one_call_and_native_usage():
    calls = []
    def execute(body):
        calls.append(body)
        return payload()
    adapter = JeffEvidenceSelector(execute=execute, preflight=lambda body: None)
    result = adapter.choose(OBS, 'Fixed instructions', ['include', 'exclude'])
    assert result.outcome == 'ok' and result.output_tokens is None
    assert len(calls) == 1 and calls[0]['state'] == OBS
    assert calls[0]['questions']['next_action']['instructions'] == 'Fixed instructions'

def test_preflight_blocks_forward_and_gold():
    def blocked(body):
        raise ValueError('truncated')
    def never(body):
        pytest.fail('unexpected inference')
    adapter = JeffEvidenceSelector(execute=never, preflight=blocked)
    with pytest.raises(ValueError, match='truncated'):
        adapter.choose(OBS, 'Fixed', ['include', 'exclude'])
    with pytest.raises(ValueError, match='observation_keys'):
        adapter.choose({**OBS, 'grade': 2}, 'Fixed', ['include', 'exclude'])

@pytest.mark.parametrize('change', ['model', 'nan', 'choice', 'usage'])
def test_invalid_response(change):
    p = payload()
    if change == 'model': p['model'] = 'jev'
    if change == 'nan': p['answers']['next_action']['probabilities']['include'] = float('nan')
    if change == 'choice': p['answers']['next_action']['choice'] = 'exclude'
    if change == 'usage': p['usage']['input_tokens'] = True
    adapter = JeffEvidenceSelector(execute=lambda body: p, preflight=lambda body: None)
    assert adapter.choose(OBS, 'Fixed', ['include', 'exclude']).outcome != 'ok'

def test_failure_has_no_retry():
    calls = []
    def fail(body):
        calls.append(body)
        raise RuntimeError('private diagnostic')
    result = JeffEvidenceSelector(execute=fail, preflight=lambda body: None).choose(OBS, 'Fixed', ['include', 'exclude'])
    assert len(calls) == 1 and result.error_code == 'jeff_native_failure'
