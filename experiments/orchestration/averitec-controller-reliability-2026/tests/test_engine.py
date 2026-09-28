import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import (ACTIONS, METRICS, STAGES, INSTRUCTIONS, FaultTools, OracleController,
                    RecordingTools, ReplayTools, State, expected_action, replay_checkpoints,
                    run_episode, validate_observation)
from providers import ControllerResult


class EmptyTools:
    def __init__(self):
        self.actions = []

    def run(self, action):
        self.actions.append(action)
        return {'status': 'empty', 'metrics': dict.fromkeys(METRICS, 0), 'error_code': None}


def test_empty_evidence_still_completes_every_stage_and_one_verdict():
    tools = EmptyTools()
    result = run_episode(OracleController(), tools)
    assert result['outcome'] == 'completed'
    assert tools.actions == list(STAGES)
    assert result['controller_calls'] == 8


@pytest.mark.parametrize('scenario,expected,count', [
    ('nominal','completed',8), ('retrieval_timeout_once','completed',9),
    ('retrieval_timeout_persistent','correct_abort',5),
])
def test_fault_routing(scenario, expected, count):
    tools = EmptyTools()
    result = run_episode(OracleController(), FaultTools(tools, scenario))
    assert result['outcome'] == expected
    assert result['controller_calls'] == count
    assert result['protocol_correct_termination']
    if scenario == 'retrieval_timeout_persistent':
        assert 'retrieve' not in tools.actions  # failure injected before dispatch


def test_full_enum_keeps_wrong_choices_possible_and_invalid_not_executed():
    class Wrong:
        def choose(self, observation, instructions, actions):
            assert set(actions) == set(ACTIONS)
            assert 'expected_action' not in json.dumps(observation)
            return ControllerResult('finish','ok',1,None,None,'bad')
    tools = EmptyTools()
    result = run_episode(Wrong(), tools)
    assert result['outcome'] == 'rule_violation'
    assert result['events'][0]['violation'] == 'early_finish'
    assert not tools.actions
    assert not result['full_pipeline_completed']


def test_checkpoint_evaluation_preserves_later_hard_states_after_failure():
    class Bad:
        def choose(self, *args):
            return ControllerResult(None,'invalid_output',1,None,None,'bad')
    reference = run_episode(OracleController(), EmptyTools())
    result = replay_checkpoints(Bad(), reference)
    assert len(result['events']) == 8
    assert not any(row['compliant'] for row in result['events'])
    assert result['full_pipeline_completed'] is None


def test_transport_error_never_becomes_compliance_failure_with_fake_action():
    class Offline:
        def choose(self,*args):
            return ControllerResult(None,'transport_error',1,None,None,'offline')
    result = run_episode(Offline(), EmptyTools())
    assert result['outcome'] == 'transport_error'
    assert result['events'][0]['action'] is None


def test_observation_rejects_text_unknown_fields_and_bad_counts():
    observation = State().observation()
    with pytest.raises(ValueError):
        validate_observation({**observation, 'claim': 'secret text'})
    observation['metrics']['facet_count'] = 'gold data'
    with pytest.raises(ValueError):
        validate_observation(observation)


def test_budget_instruction_is_machine_verifiable():
    assert expected_action(State(max_calls=7).observation()) == 'abort'
    assert expected_action(State(max_calls=8).observation()) == 'decompose'


def test_replay_rejects_unrecorded_dispatch_and_order():
    recorder = RecordingTools(EmptyTools())
    run_episode(OracleController(), recorder)
    replay = ReplayTools(recorder.records)
    with pytest.raises(ValueError, match='replay_tool_sequence_mismatch'):
        replay.run('verdict')
    assert run_episode(OracleController(), replay)['outcome'] == 'completed'
