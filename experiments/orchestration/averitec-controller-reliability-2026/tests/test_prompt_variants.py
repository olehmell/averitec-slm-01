from pathlib import Path
import sys

import pytest

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from engine import ACTIONS, INSTRUCTIONS, State
from prompt_variants import (PresentedController, instructions_for, present_observation,
                             presentation_identity)


def test_v1_is_exact_and_not_mutated():
    state = State().observation()
    shown = present_observation(state, 'v1')
    assert shown == state and shown is not state
    assert instructions_for('v1') == INSTRUCTIONS


@pytest.mark.parametrize('attempts,status,last_tool', [(0,'ok','queries'),
                                                     (1,'timeout','retrieve'),
                                                     (2,'invalid','retrieve')])
def test_last_result_is_not_misattributed_to_pending(attempts, status, last_tool):
    state = State(completed=['decompose','queries'], attempts=attempts, last_status=status)
    shown = present_observation(state.observation(), 'v2')
    assert shown['pending_tool'] == 'retrieve'
    assert shown['last_tool_result'] == {'tool': last_tool, 'status': status}
    assert shown['failed_attempts_on_pending_tool'] == attempts
    assert shown['metrics'] == state.metrics
    assert 'expected_action' not in shown


def test_initial_and_terminal_presentations():
    initial = present_observation(State().observation(), 'v2')
    assert initial['last_tool_result'] is None
    terminal = present_observation(State(completed=list(ACTIONS[:-2]), last_status='empty').observation(), 'v2')
    assert terminal['pending_tool'] is None
    assert terminal['last_tool_result'] == {'tool':'verdict','status':'empty'}


def test_examples_add_instructions_not_runtime_information():
    state = State().observation()
    assert present_observation(state, 'v3') == present_observation(state, 'v2')
    assert instructions_for('v3').startswith(instructions_for('v2'))
    assert presentation_identity('v3')['synthetic_teaching_examples'] == 6
    assert presentation_identity('v2')['instructions_sha256'] != presentation_identity('v3')['instructions_sha256']


@pytest.mark.parametrize('state', [
    State(completed=['decompose', 'queries'], attempts=1, calls=4, last_status='timeout'),
    State(completed=['decompose', 'queries'], attempts=2, calls=5, last_status='invalid'),
    State(completed=['decompose', 'queries', 'retrieve', 'qa', 'coverage', 'select'], calls=11,
          last_status='empty'),
])
def test_v4_has_exactly_v2_presentation_for_normal_retry_and_budget_states(state):
    observation = state.observation()
    assert present_observation(observation, 'v4') == present_observation(observation, 'v2')
    assert presentation_identity('v4')['observation_presentation'] == 'v2'
    assert presentation_identity('v4')['synthetic_teaching_examples'] == 0


def test_v4_is_compact_instructions_only_without_examples_or_oracle_state():
    instructions = instructions_for('v4')
    assert instructions != instructions_for('v2')
    assert 'Illustrative examples' not in instructions
    assert 'expected_action' not in instructions
    assert 'pending_tool' in instructions


def test_wrapper_preserves_complete_global_vocabulary_and_raw_result():
    class Capture:
        model = 'raw-model'
        tracer = None
        def choose(self, observation, instructions, actions):
            self.request = observation, instructions, actions
            return 'raw_wrong_answer'
    raw = Capture()
    wrapped = PresentedController(raw, 'v3')
    sentinel = object()
    wrapped.tracer = sentinel
    assert raw.tracer is sentinel
    assert wrapped.choose(State().observation(), INSTRUCTIONS, list(reversed(ACTIONS))) == 'raw_wrong_answer'
    assert raw.request[2] == list(reversed(ACTIONS))
    assert raw.request[0]['pending_tool'] == 'decompose'
    assert wrapped.expected_returned_model == raw.model


def test_wrapper_delegates_distinct_expected_returned_model():
    class Capture:
        model = 'requested-alias'
        expected_returned_model = 'provider-version'
        tracer = None
    assert PresentedController(Capture(), 'v4').expected_returned_model == 'provider-version'


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match='unknown_prompt_variant'):
        instructions_for('best_answer')
