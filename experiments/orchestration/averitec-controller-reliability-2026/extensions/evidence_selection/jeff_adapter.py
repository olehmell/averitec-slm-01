"""Jeff binary selector boundary. Not registered in sealed historical runners.

The caller owns a pinned native Engine, a lossless preflight and durable call
intents. No environment credentials, HTTP fallback or retry path is provided.
"""
from time import perf_counter
import math

from providers import ControllerResult
from selector_runtime import _validate_request, _trace_request

JEFF_COMMIT = '34b32f99a727c47b679adde33f4702a001e02979'
MODEL = 'knowledgator/gliformer-large-v1'
MODEL_REVISION = 'd0a4e53d09cebe6bc963dd9be319d4279084bb2d'
RETURNED_MODEL = 'gliformer-large-v1'


class JeffEvidenceSelector:
    """One native request per call, with mandatory preflight before inference.

    execute accepts the wire request and returns its JSON-compatible response.
    preflight must raise if any state/instructions/options would be truncated.
    Identity and checkpoint verification belong to the future launch gate; this
    class alone is not evidence that real-runtime qualification has passed.
    """

    def __init__(self, *, execute, preflight, tracer=None):
        if not callable(execute) or not callable(preflight):
            raise ValueError('jeff_execute_and_lossless_preflight_required')
        self.execute, self.preflight, self.tracer = execute, preflight, tracer

    def choose(self, observation, instructions, actions):
        clean, instructions, actions = _validate_request(observation, instructions, actions)
        body = {'model': RETURNED_MODEL, 'state': clean, 'questions': {
            'next_action': {'type': 'choice', 'instructions': instructions,
                            'criteria': {action: None for action in actions}}}}
        self.preflight(body)
        started = perf_counter()
        def failure(outcome, code):
            return ControllerResult(None, outcome, (perf_counter()-started)*1000,
                                    None, None, MODEL, error_code=code)
        with _trace_request(self.tracer, body=body, model=MODEL) as trace:
            try:
                payload = self.execute(body)
            except Exception:
                trace.update(metadata={'outcome': 'transport_error', 'error_code': 'jeff_native_failure'})
                return failure('transport_error', 'jeff_native_failure')
            trace.update(output=payload, metadata={'output_token_semantics': 'nominal_not_generated'})
        if not isinstance(payload, dict):
            return failure('invalid_output', 'jeff_payload_invalid')
        if payload.get('model') != RETURNED_MODEL:
            return failure('version_mismatch', 'jeff_model_mismatch')
        answers = payload.get('answers')
        answer = answers.get('next_action') if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get('type') != 'choice':
            return failure('invalid_output', 'jeff_answer_invalid')
        probs, choice, confidence = answer.get('probabilities'), answer.get('choice'), answer.get('confidence')
        number = lambda x: type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1
        if (not isinstance(probs, dict) or set(probs) != set(actions)
                or not all(number(v) for v in probs.values())
                or abs(sum(probs.values())-1) > 1e-6
                or not isinstance(choice, str) or choice not in actions
                or probs[choice] != max(probs.values()) or not number(confidence)):
            return failure('invalid_output', 'jeff_choice_invalid')
        usage = payload.get('usage')
        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('input_tokens', 'output_tokens')):
            return failure('invalid_output', 'jeff_usage_invalid')
        return ControllerResult(choice, 'ok', (perf_counter()-started)*1000,
                                usage['input_tokens'], None, MODEL,
                                probabilities=dict(probs), confidence=confidence,
                                returned_model=RETURNED_MODEL)
