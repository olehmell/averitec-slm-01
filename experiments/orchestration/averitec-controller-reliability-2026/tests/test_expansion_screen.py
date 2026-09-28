"""Five-arm integration tests use synthetic tools and mock providers only."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
import providers
from data import check_freeze, file_hash
from engine import ACTIONS
from providers import ControllerResult
from test_run_variants import inputs
from warmup import WARMUP_ACTIONS, WarmupFailure, premeasure_warmup


@pytest.mark.parametrize('controller,count', [('jev',2),('qwen',2),('lfm',3),('lfm26',3),('gemini',2)])
def test_expansion_exact_matrix(tmp_path, monkeypatch, controller, count):
    args, settings, freeze = inputs(tmp_path, limit=2)
    args.controller, args.scenarios, args.plan = controller, list(run.SCENARIOS), True
    args.expansion_screen = True
    calls = []
    monkeypatch.setattr(run, 'execute', lambda condition, *_: calls.append(condition) or [])
    run.development_sweep(args, settings, freeze)
    assert len(calls) == count
    assert [(c.prompt_variant,c.generation_profile) for c in calls] == run.expansion_conditions(settings)[controller]
    assert len({c.output for c in calls}) == count


def test_expansion_cannot_report_one_case_as_complete(tmp_path, monkeypatch):
    args, settings, freeze = inputs(tmp_path)
    args.expansion_screen, args.scenarios = True, list(run.SCENARIOS)
    monkeypatch.setattr(run, 'execute', lambda *_: pytest.fail('must reject before dispatch'))
    with pytest.raises(ValueError, match='expansion_requires_two_development_cases'):
        run.development_sweep(args, settings, freeze)


def test_whole_expansion_synthetic_journals_complete(tmp_path, monkeypatch):
    args, settings, freeze = inputs(tmp_path, limit=2)
    args.expansion_screen, args.scenarios, args.output = True, list(run.SCENARIOS), tmp_path
    real_execute = run.execute
    monkeypatch.setattr(run, 'execute', lambda condition, settings, freeze: real_execute(condition, settings, freeze, synthetic=True))
    rows = []
    for controller in run.expansion_conditions(settings):
        args.controller = controller
        rows.extend(run.development_sweep(args, settings, freeze))
    assert len(rows) == 72
    assert all(e['compliant'] for row in rows for e in row['events'])
    assert len({r['trial_id'] for r in rows}) == 72
    for controller, pairs in run.expansion_conditions(settings).items():
        for variant, profile in pairs:
            assert len(run.read_complete_run(tmp_path/f'{controller}-{variant}-{profile}.jsonl')) == 6


def test_expansion_matrix_and_budget_cannot_grow():
    settings = run.config()
    settings['expansion_screening']['conditions']['lfm'].append(['v1','lfm_native'])
    with pytest.raises(ValueError, match='expansion_matrix_not_predeclared'):
        run.expansion_conditions(settings)
    settings = run.config()
    settings['expansion_screening']['screening_case_limit'] = 20
    with pytest.raises(ValueError, match='expansion_budget_not_predeclared'):
        run.expansion_conditions(settings)


def test_expansion_all_models_have_pinned_identity_and_cache_hashes():
    settings = run.config()
    path = run.ROOT / settings['model_assets_manifest']
    assert file_hash(path) == settings['model_assets_manifest_sha256']
    assets = json.loads(path.read_text())['models']
    for arm in ('lfm','lfm26','qwen'):
        model = settings['models'][arm]
        assert any(m['repository'] == model['model'] and m['revision'] == model['revision'] for m in assets)
    assert settings['models']['gemini']['expected_version']
    assert sum(len(v) for v in run.expansion_conditions(settings).values()) == 12


def test_old_code_binding_is_not_rebound(tmp_path):
    _args, _settings, freeze = inputs(tmp_path)
    freeze['code_binding']['previous-controller-code.py'] = '0' * 64
    with pytest.raises(ValueError, match='freeze_code_drift'):
        check_freeze(freeze, freeze['selection'], allow_synthetic=True)


def test_gemini_actual_boundary_and_complete_resume(tmp_path, monkeypatch):
    args, settings, freeze = inputs(tmp_path)
    args.controller, args.generation_profile, args.endpoint = 'gemini','gemini_native',None
    calls=[]
    monkeypatch.setattr(run, 'require_committed_config', lambda: None)
    monkeypatch.setattr(run, 'check_freeze', lambda *a, **kw: None)
    monkeypatch.setattr(run, 'load_key', lambda name: calls.append(('key',name)))
    class Fake:
        def __init__(self, model, expected_version, timeout_seconds, **kwargs):
            self.model, self.expected_returned_model = model,expected_version
            self.timeout_seconds, self.tracer = timeout_seconds,None
            self.resolved_profile = {'name': 'gemini_native'}
        def choose(self, observation, instructions, actions):
            calls.append(('request', observation, list(actions), self.timeout_seconds))
            return ControllerResult('finish','ok',1,10,3,self.model,
                                    returned_model=self.expected_returned_model)
    monkeypatch.setattr(run,'GeminiController',Fake)
    rows=run.execute(args,settings,freeze)
    requests=[c for c in calls if c[0]=='request']
    assert len(requests)==9 and requests[0][-1]==120
    assert requests[0][1]['pending_tool']=='decompose'
    assert all(c[2]==list(ACTIONS) for c in requests)
    assert rows[0]['events'][0]['observation']['stage']=='decompose'
    assert not rows[0]['events'][0]['compliant']
    metadata=json.loads(Path(str(args.output)+'.run.json').read_text())
    assert metadata['generation_settings']['timeout_seconds']==30
    assert metadata['generation_settings']['expected_version']==settings['models']['gemini']['expected_version']
    run.execute(args,settings,freeze)
    assert len([c for c in calls if c[0]=='request'])==9


def test_key_loader_is_scoped_and_never_sources_dotenv(tmp_path,monkeypatch):
    (tmp_path/'.env').write_text('OTHER_SECRET=not-for-model\nGEMINI_API_KEY="test-key"\nSHELL_TRICK=$(false)\n')
    monkeypatch.setattr(run,'ROOT',tmp_path)
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    run.load_key('GEMINI_API_KEY')
    assert run.os.environ['GEMINI_API_KEY']=='test-key'
    assert 'SHELL_TRICK' not in run.os.environ
    monkeypatch.delenv('GEMINI_API_KEY')
    with pytest.raises(ValueError,match='unsupported_api_key_name'):
        run.load_key('OTHER_SECRET')


@pytest.mark.parametrize('reasoning',[None,'', '   '])
def test_lfm26_requires_real_separate_reasoning(monkeypatch,reasoning):
    monkeypatch.setattr(providers,'_http_post',lambda *_: {
        'model':'LiquidAI/LFM2.5-2.6B', 'choices':[{'finish_reason':'stop',
        'message':{'reasoning':reasoning,'content':'{"action":"decompose"}'}}]})
    controller=providers.OpenAIController('http://mock/v1','LiquidAI/LFM2.5-2.6B',generation_profile='lfm26_native')
    result=controller.choose({},'instructions',list(ACTIONS))
    assert result.outcome=='invalid_output' and result.error_code=='missing_required_reasoning'


@pytest.mark.parametrize('returned,status',[('pinned-revision','ok'),('other-revision','failed')])
def test_warmup_alias_and_version_are_separate(returned,status):
    class Fake:
        model='gemini-3.1-flash-lite'
        expected_returned_model='pinned-revision'
        timeout_seconds=30
        tracer=None
        def choose(self,*args):
            return ControllerResult('finish','ok',1,1,1,self.model,returned_model=returned)
    if status=='ok':
        receipt=premeasure_warmup(Fake(),instructions='instructions',actions=list(WARMUP_ACTIONS))
        assert receipt['expected_returned_model']=='pinned-revision'
    else:
        with pytest.raises(WarmupFailure):
            premeasure_warmup(Fake(),instructions='instructions',actions=list(WARMUP_ACTIONS))
