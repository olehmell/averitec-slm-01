"""Small observable workflow; evaluation rules never constrain output enums."""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

STAGES = ('decompose', 'queries', 'retrieve', 'qa', 'coverage', 'select', 'verdict')
ACTIONS = STAGES + ('finish', 'abort')
SCENARIOS = ('nominal', 'retrieval_timeout_once', 'retrieval_timeout_persistent')
METRICS = ('facet_count', 'query_count', 'candidate_count', 'qa_count', 'selected_count')
STATUSES = ('ok', 'empty', 'timeout', 'invalid')
INSTRUCTIONS = '''You control a fact-checking workflow using numeric state only.
Return exactly one action. Rules, in priority order:
1. Complete tools in this order: decompose, queries, retrieve, qa, coverage,
   select, verdict. Tools own all texts. Never predict factual correctness.
2. A timeout or invalid result does not complete a stage. Retry that same stage
   exactly once. If that retry also fails, choose abort.
3. An ok or empty result completes a stage. Empty evidence is a valid outcome;
   continue the remaining stages and invoke verdict even with no evidence.
4. Choose finish only after all seven stages, including verdict, completed.
5. If calls_remaining is smaller than the minimum calls needed to complete
   all incomplete stages plus finish, choose abort. Otherwise abort is allowed
   only after the failed retry in rule 2.
6. No skipped stages, repeats of completed stages, extra actions, or commentary.
stage is the next incomplete tool; completed_stages is the execution history.
attempts_on_stage counts failed attempts at stage. calls_remaining includes this
decision. tool_status reports the preceding tool outcome. Unknown measurements
are null, never evidence of success. All named actions are available in the
output vocabulary, including actions that would violate these rules now.'''


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass
class State:
    completed: list[str] = field(default_factory=list)
    attempts: int = 0
    calls: int = 0
    max_calls: int = 12
    last_status: str | None = None
    metrics: dict = field(default_factory=lambda: dict.fromkeys(METRICS))

    def observation(self) -> dict:
        value = {
            'schema': 'averitec-reliability-observation/v1',
            'stage': STAGES[len(self.completed)] if len(self.completed) < len(STAGES) else 'terminal',
            'completed_stages': list(self.completed),
            'attempts_on_stage': self.attempts,
            'tool_status': self.last_status,
            'calls_remaining': self.max_calls - self.calls,
            'metrics': deepcopy(self.metrics),
        }
        validate_observation(value)
        return value


def validate_observation(value: dict) -> None:
    required = {'schema', 'stage', 'completed_stages', 'attempts_on_stage',
                'tool_status', 'calls_remaining', 'metrics'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('observation_fields')
    if value['schema'] != 'averitec-reliability-observation/v1':
        raise ValueError('observation_version')
    completed = value['completed_stages']
    if not isinstance(completed, list) or completed != list(STAGES[:len(completed)]) or len(completed) > len(STAGES):
        raise ValueError('observation_sequence')
    stage = STAGES[len(completed)] if len(completed) < len(STAGES) else 'terminal'
    if value['stage'] != stage or value['tool_status'] not in (None,) + STATUSES:
        raise ValueError('observation_enum')
    for key in ('attempts_on_stage', 'calls_remaining'):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('observation_counter')
    if value['attempts_on_stage'] > 2:
        raise ValueError('observation_attempts')
    if set(value['metrics']) != set(METRICS):
        raise ValueError('observation_metrics')
    for item in value['metrics'].values():
        if item is not None and (type(item) is not int or item < 0):
            raise ValueError('observation_metric_value')


def expected_action(observation: dict) -> str:
    """Oracle for explicit instructions, not a gold verdict or quality target."""
    validate_observation(observation)
    remaining = len(STAGES) - len(observation['completed_stages']) + 1
    if observation['calls_remaining'] < remaining or observation['attempts_on_stage'] >= 2:
        return 'abort'
    return 'finish' if observation['stage'] == 'terminal' else observation['stage']


def checked_tool_result(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {'status', 'metrics', 'error_code'}:
        raise ValueError('tool_result_fields')
    if value['status'] not in STATUSES or set(value['metrics']) != set(METRICS):
        raise ValueError('tool_result_schema')
    if any(type(item) is not int or item < 0 for item in value['metrics'].values()):
        raise ValueError('tool_result_numbers')
    # Error detail never reaches the model; only bounded codes persist publicly.
    code = value['error_code']
    if code is not None and (not isinstance(code, str) or len(code) > 80 or not all(c.isalnum() or c == '_' for c in code)):
        raise ValueError('tool_result_error_code')
    return deepcopy(value)


class OracleController:
    """Harness validation only. Never labeled as a model observation."""
    def choose(self, observation, instructions, actions):
        from providers import ControllerResult
        return ControllerResult(action=expected_action(observation), outcome='ok',
                                latency_ms=0.0, model='deterministic_oracle',
                                input_tokens=None, output_tokens=None)


class RecordingTools:
    def __init__(self, tools):
        self.tools, self.records = tools, []

    def run(self, action):
        result = checked_tool_result(self.tools.run(action))
        self.records.append({'action': action, 'result': deepcopy(result)})
        return result


class ReplayTools:
    def __init__(self, records):
        self.records = deepcopy(records)
        self.position = 0

    def run(self, action):
        if self.position >= len(self.records) or self.records[self.position]['action'] != action:
            raise ValueError('replay_tool_sequence_mismatch')
        result = checked_tool_result(self.records[self.position]['result'])
        self.position += 1
        return result


class FaultTools:
    """A timeout happens before dispatch; it does not consume a frozen result."""
    def __init__(self, tools, scenario):
        if scenario not in SCENARIOS:
            raise ValueError('unknown_scenario')
        self.tools, self.scenario, self.retrieval_attempts = tools, scenario, 0
        self.metrics = dict.fromkeys(METRICS, 0)

    def run(self, action):
        if action == 'retrieve':
            self.retrieval_attempts += 1
            fail = (self.scenario == 'retrieval_timeout_persistent' or
                    (self.scenario == 'retrieval_timeout_once' and self.retrieval_attempts == 1))
            if fail:
                return {'status': 'timeout', 'metrics': deepcopy(self.metrics),
                        'error_code': 'injected_pre_dispatch_timeout'}
        result = checked_tool_result(self.tools.run(action))
        self.metrics = deepcopy(result['metrics'])
        return result


def _decision(controller, observation, step, actions, tracer):
    context = (tracer.span('controller.decision', input={
        'observation': observation, 'instructions': INSTRUCTIONS, 'actions': actions},
        metadata={'step': step}) if tracer else nullcontext())
    with context as span:
        result = controller.choose(deepcopy(observation), INSTRUCTIONS, actions)
        expected = expected_action(observation)
        event = {
            'step': step, 'observation': observation, 'observation_sha256': digest(observation),
            'expected_action': expected, 'action': result.action,
            'provider_outcome': result.outcome, 'model': result.model,
            'returned_model': getattr(result, 'returned_model', None),
            'latency_ms': result.latency_ms, 'input_tokens': result.input_tokens,
            'output_tokens': result.output_tokens, 'error_code': result.error_code,
            'confidence': result.confidence, 'probabilities': result.probabilities,
            'compliant': result.outcome == 'ok' and result.action == expected,
        }
        if result.outcome == 'ok' and result.action != expected:
            event['violation'] = ('early_finish' if result.action == 'finish' else
                                  'retry_rule' if observation['attempts_on_stage'] else 'wrong_stage')
        if span:
            span.update(output=event, level='DEFAULT' if event['compliant'] else 'WARNING')
        return result, event


def _tool_call(tools, action, step, tracer):
    context = (tracer.span('tool.' + action, input={'action': action},
                           metadata={'step': step}) if tracer else nullcontext())
    with context as span:
        result = checked_tool_result(tools.run(action))
        if span:
            # Live worker snapshots are analysis-only, never controller input.
            target = tools.tools if isinstance(tools, (FaultTools, RecordingTools)) else tools
            if isinstance(target, RecordingTools):
                target = target.tools
            snapshot = target.trace_snapshot() if hasattr(target, 'trace_snapshot') else None
            span.update(output={'result': result, 'worker_state': snapshot},
                        level='WARNING' if result['status'] in ('timeout', 'invalid') else 'DEFAULT')
        return result


def run_episode(controller, tools, *, max_calls=12, actions=None, tracer=None) -> dict:
    """Stop on the first model violation; executor intervention is not success."""
    actions = list(ACTIONS if actions is None else actions)
    if len(actions) != len(ACTIONS) or set(actions) != set(ACTIONS):
        raise ValueError('global_action_vocabulary_required')
    state, events = State(max_calls=max_calls), []
    outcome = 'step_cap'
    for step in range(max_calls):
        observation = state.observation()
        expected = expected_action(observation)
        result, event = _decision(controller, observation, step, actions, tracer)
        state.calls += 1
        events.append(event)
        if result.outcome != 'ok':
            outcome = result.outcome
            break
        if result.action != expected:
            outcome = 'rule_violation'
            event['violation'] = ('early_finish' if result.action == 'finish' else
                                  'retry_rule' if state.attempts else 'wrong_stage')
            break
        if result.action in ('finish', 'abort'):
            outcome = 'completed' if result.action == 'finish' else 'correct_abort'
            break
        tool_result = _tool_call(tools, result.action, step, tracer)
        event['tool_result'] = tool_result
        state.metrics = deepcopy(tool_result['metrics'])
        state.last_status = tool_result['status']
        if tool_result['status'] in ('ok', 'empty'):
            state.completed.append(result.action)
            state.attempts = 0
        else:
            state.attempts += 1
    return {
        'outcome': outcome, 'events': events, 'completed_stages': state.completed,
        'full_pipeline_completed': outcome == 'completed' and state.completed == list(STAGES),
        'protocol_correct_termination': outcome in ('completed', 'correct_abort'),
        'controller_calls': state.calls,
    }


def replay_checkpoints(controller, reference: dict, *, actions=None, tracer=None) -> dict:
    """Evaluate all frozen states even if an earlier response violates a rule."""
    events = []
    for event in reference['events']:
        observation = deepcopy(event['observation'])
        _, measured = _decision(controller, observation, event['step'], list(actions or ACTIONS), tracer)
        events.append(measured)
    return {'outcome': 'checkpoints_evaluated', 'events': events,
            'full_pipeline_completed': None, 'protocol_correct_termination': None,
            'controller_calls': len(events)}
