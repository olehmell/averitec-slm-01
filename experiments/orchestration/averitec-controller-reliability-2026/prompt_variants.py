"""Predeclared, information-equivalent presentations of the same workflow state.

The executor/oracle retains v1 observations. Only the controller-facing view
changes; no expected action, action mask, factual text, or repair is introduced.
"""
from __future__ import annotations

from copy import deepcopy

from engine import INSTRUCTIONS, STAGES, digest, validate_observation

VARIANTS = ('v1', 'v2', 'v3', 'v4')
CLEAR_INSTRUCTIONS = '''Choose the single action to execute NOW in this workflow.
You control execution, not factual correctness. Tools own all text.
The seven tools, in order, are:
decompose, queries, retrieve, qa, coverage, select, verdict.

Apply this decision procedure from top to bottom; stop at the first match:
1. Count unfinished tools (7 minus the number of completed_tools). If
   remaining_decisions is less than that count plus 1 for finish, choose abort.
2. If failed_attempts_on_pending_tool is 2, choose abort.
3. If all seven tools are in completed_tools, choose finish.
4. Otherwise choose pending_tool itself. Execute it NOW, not the tool after it.

State definitions:
- completed_tools lists only tools that already returned ok or empty.
- pending_tool is the FIRST UNFINISHED tool, not the last executed tool.
  It is null only when all seven tools have completed.
- last_tool_result names the tool that produced the previous result and its
  status. It does NOT mean pending_tool completed. It is null before any tool.
- failed_attempts_on_pending_tool counts failed attempts, not total attempts.
  A timeout or invalid result leaves that tool unfinished: after one failure
  execute that SAME tool once more; after two failures abort.
- remaining_decisions includes this decision and the final finish decision.
- metrics are observations, not permission to skip tools. Zero evidence is
  valid: even with zero candidates or answers, complete every remaining tool,
  including verdict. null means unknown, not success.

All nine action names are in the output vocabulary; some violate the rules in
this state. Never skip a tool, repeat a completed tool, or finish before verdict
has completed. Return one action only.'''


COMPACT_INSTRUCTIONS = '''Choose one action NOW. Tools are ordered:
decompose, queries, retrieve, qa, coverage, select, verdict.

Apply these rules in order and stop at the first match:
1. Let unfinished = 7 - count(completed_tools). If remaining_decisions
   is less than unfinished + 1, choose abort.
2. If failed_attempts_on_pending_tool is 2, choose abort.
3. If all seven tools are completed, choose finish.
4. Otherwise choose pending_tool exactly.

pending_tool is the first unfinished tool, not the previous tool. An ok or
empty result completes a tool. A timeout or invalid result leaves it pending:
retry that same tool once, then abort. remaining_decisions includes this action
and finish. Metrics, including zero or null, never permit skipping a tool.
All nine actions remain available; return one action only.'''


def present_observation(observation: dict, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError('unknown_prompt_variant')
    validate_observation(observation)
    if variant == 'v1':
        return deepcopy(observation)
    pending = None if observation['stage'] == 'terminal' else observation['stage']
    completed = observation['completed_stages']
    last_name = (pending if observation['attempts_on_stage'] else
                 completed[-1] if completed else None)
    return {
        'schema': 'averitec-reliability-presentation/v2',
        'completed_tools': list(completed),
        'pending_tool': pending,
        'last_tool_result': ({'tool': last_name, 'status': observation['tool_status']}
                             if observation['tool_status'] is not None else None),
        'failed_attempts_on_pending_tool': observation['attempts_on_stage'],
        'remaining_decisions': observation['calls_remaining'],
        'metrics': deepcopy(observation['metrics']),
    }


# These task-independent teaching examples are written from the workflow rules,
# not read from the AVeriTeC selection, tool freeze, or evaluation observations.
# Partial states keep examples short; omitted metrics never change these rules.
EXAMPLES = '''

Illustrative examples (only decision-relevant fields shown):
1. completed_tools=[decompose,queries], pending_tool=retrieve,
   last_tool_result={tool:queries,status:ok},
   failed_attempts_on_pending_tool=0, remaining_decisions=10 -> retrieve
   (retrieve has NOT run yet; do not choose qa.)
2. completed_tools=[decompose,queries,retrieve], pending_tool=qa,
   last_tool_result={tool:retrieve,status:empty}, metrics.candidate_count=0,
   failed_attempts_on_pending_tool=0, remaining_decisions=9 -> qa
   (Empty evidence does not permit finish.)
3. completed_tools=[decompose,queries], pending_tool=retrieve,
   last_tool_result={tool:retrieve,status:timeout},
   failed_attempts_on_pending_tool=1, remaining_decisions=8 -> retrieve
4. completed_tools=[decompose,queries], pending_tool=retrieve,
   last_tool_result={tool:retrieve,status:timeout},
   failed_attempts_on_pending_tool=2, remaining_decisions=7 -> abort
5. completed_tools=[decompose,queries,retrieve,qa,coverage,select,verdict],
   pending_tool=null, last_tool_result={tool:verdict,status:empty},
   failed_attempts_on_pending_tool=0, remaining_decisions=3 -> finish
6. completed_tools=[decompose,queries,retrieve,qa,coverage,select],
   pending_tool=verdict, failed_attempts_on_pending_tool=0,
   remaining_decisions=1 -> abort
   (verdict plus finish requires two decisions.)'''


def instructions_for(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError('unknown_prompt_variant')
    if variant == 'v1':
        return INSTRUCTIONS
    if variant == 'v4':
        return COMPACT_INSTRUCTIONS
    return CLEAR_INSTRUCTIONS + (EXAMPLES if variant == 'v3' else '')


def presentation_identity(variant: str) -> dict:
    return {'prompt_variant': variant, 'instructions_sha256': digest(instructions_for(variant)),
            'observation_presentation': 'v1' if variant == 'v1' else 'v2',
            'synthetic_teaching_examples': 6 if variant == 'v3' else 0}


def native_choice_criteria(variant: str) -> dict[str, str]:
    """Static Choice descriptions; never a per-observation action filter."""
    if variant not in ('v2', 'v3'):
        raise ValueError('native_criteria_require_clear_presentation')
    result = {stage: f'Execute {stage} NOW when pending_tool is {stage}, '
                     'remaining_decisions is sufficient, and failed_attempts_on_pending_tool '
                     'is below 2. After one failed attempt, retry this same unfinished tool.'
              for stage in STAGES}
    result['finish'] = ('Finish only when all seven tools, including verdict, are in completed_tools '
                        'and at least one decision remains. Empty evidence alone never permits finish.')
    result['abort'] = ('Abort when failed_attempts_on_pending_tool is 2, or remaining_decisions '
                       'is less than the number of unfinished tools plus 1 for finish. Otherwise do not abort.')
    return result


class PresentedController:
    """Apply the presentation at the real request boundary, never at the oracle."""
    def __init__(self, controller, variant: str):
        self.controller = controller
        self.variant = variant
        self.instructions = instructions_for(variant)

    @property
    def tracer(self):
        return self.controller.tracer

    @tracer.setter
    def tracer(self, value):
        self.controller.tracer = value

    @property
    def model(self):
        return self.controller.model

    @property
    def expected_returned_model(self):
        return getattr(self.controller, 'expected_returned_model', self.controller.model)

    @property
    def resolved_profile(self):
        return getattr(self.controller, 'resolved_profile', None)

    @property
    def timeout_seconds(self):
        return self.controller.timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value):
        self.controller.timeout_seconds = value

    def choose(self, observation, _instructions, actions):
        return self.controller.choose(present_observation(observation, self.variant),
                                      self.instructions, actions)
