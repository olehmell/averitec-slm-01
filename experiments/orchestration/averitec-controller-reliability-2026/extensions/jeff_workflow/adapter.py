"""Pinned native Jeff controller for the nine-action workflow extension."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

EXT = Path(__file__).resolve().parent
EXP = EXT.parents[1]
EVIDENCE_EXT = EXP / "extensions" / "evidence_selection"
for path in (EXP, EVIDENCE_EXT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from engine import ACTIONS, INSTRUCTIONS
from prompt_variants import instructions_for, present_observation
from providers import ControllerResult
import jeff_native as native

MODEL = native.MODEL
MODEL_REVISION = native.MODEL_REVISION
RETURNED_MODEL = native.RETURNED_MODEL
QUESTION_ID = "next_action"
PROFILE = "jeff_native_workflow_v1"
NATIVE_INSTRUCTIONS = instructions_for("v2")
# Jeff serializes each class probability to four decimal places. For nine
# labels, nearest rounding can move the displayed sum by at most 0.00045.
PROBABILITY_SUM_TOLERANCE = 0.0005


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _validate_actions(actions: Any) -> list[str]:
    if (not isinstance(actions, list) or len(actions) != len(ACTIONS)
            or len(set(actions)) != len(actions) or set(actions) != set(ACTIONS)):
        raise ValueError("jeff_workflow_invalid_actions")
    return list(actions)


def _validate_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise ValueError("jeff_workflow_invalid_observation")
    try:
        encoded = json.dumps(observation, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
        clean = json.loads(encoded)
    except (TypeError, ValueError):
        raise ValueError("jeff_workflow_invalid_observation") from None
    if not isinstance(clean, dict) or not clean:
        raise ValueError("jeff_workflow_invalid_observation")
    return clean


class JeffWorkflowRuntime(native.JeffNativeRuntime):
    """Jeff runtime with a closed nine-action request instead of the binary arm."""

    def _request(self, body: dict[str, Any]) -> Any:
        if not isinstance(body, dict) or body.get("model") != RETURNED_MODEL:
            raise ValueError("jeff_workflow_model_required")
        try:
            request = self._modules["SystemOneRequest"].model_validate(body)
        except Exception:
            raise ValueError("jeff_workflow_request_invalid") from None
        if set(request.questions) != {QUESTION_ID}:
            raise ValueError("jeff_workflow_single_question_required")
        question = request.questions[QUESTION_ID]
        criteria = getattr(question, "criteria", None)
        if getattr(question, "type", None) != "choice" or not isinstance(criteria, Mapping):
            raise ValueError("jeff_workflow_choice_required")
        actions = list(criteria)
        _validate_actions(actions)
        if any(criteria[action] is not None for action in actions):
            raise ValueError("jeff_workflow_null_criteria_required")
        return request

    def _item(self, request: Any) -> tuple[dict[str, Any], int]:
        text = self._modules["serialize_state"](request.state, self._engine.opts.state_format)
        groups = self._modules["build_groups"](request.questions, self._engine.opts)
        expected = tuple(request.questions[QUESTION_ID].criteria)
        if len(groups) != 1 or tuple(groups[0].labels) != expected:
            raise RuntimeError("jeff_workflow_preflight_group_mismatch")
        prepared = self._backend.model.prepare_inputs([text])
        if not isinstance(prepared, tuple) or len(prepared) != 3 or len(prepared[0]) != 1:
            raise RuntimeError("jeff_workflow_preflight_prepare_inputs_invalid")
        tokenized_text = prepared[0][0]
        try:
            expected_tokens = [token for token, _start, _end
                               in self._backend.model.data_processor.words_splitter(text)]
        except Exception:
            raise RuntimeError("jeff_workflow_preflight_wordsplitter_unavailable") from None
        if list(tokenized_text) != expected_tokens:
            raise RuntimeError("jeff_workflow_preflight_prepare_inputs_trimmed")
        item = {"tokenized_text": tokenized_text, "classification": [{
            "name": groups[0].name,
            "description": groups[0].description,
            "all_labels": list(groups[0].labels),
            "true_labels": [],
        }]}
        return item, len(text)


def create_runtime(checkpoint: str | Path, device: str = "cuda") -> JeffWorkflowRuntime:
    """Load the verified base runtime, then replace only its closed request gate."""
    base = native.create_runtime(checkpoint, device=device)
    identity = deepcopy(base.identity)
    evidence = dict(identity.get("evidence", {}))
    evidence.update({
        "adapter": "jeff_native_workflow_runtime/v1",
        "task": "nine_action_workflow_control",
        "action_vocabulary": list(ACTIONS),
        "instructions_sha256": hashlib.sha256(NATIVE_INSTRUCTIONS.encode()).hexdigest(),
        "qualification": "workflow_runtime_loaded_but_forward_not_yet_qualified",
    })
    identity["evidence"] = evidence
    return JeffWorkflowRuntime(
        checkpoint=checkpoint,
        device=device,
        engine=base._engine,
        backend=base._backend,
        modules=base._modules,
        identity=identity,
    )


class _NullSpan:
    def update(self, **_kwargs: Any) -> None:
        return None


@contextmanager
def _trace(tracer: Any, body: dict[str, Any]):
    if tracer is None:
        yield _NullSpan()
    else:
        with tracer.span("controller.request", kind="generation", input=body,
                         model=MODEL) as span:
            yield span


class NativeJeffController:
    """One preflighted native forward per workflow decision; no retry or fallback."""

    model = MODEL
    expected_returned_model = RETURNED_MODEL

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda",
                 tracer: Any = None, _runtime: Any = None) -> None:
        self.runtime = _runtime or create_runtime(checkpoint, device=device)
        self.tracer = tracer
        self.timeout_seconds = 120

    @staticmethod
    def _body(observation: Any, instructions: Any, actions: Any) -> dict[str, Any]:
        clean = present_observation(_validate_observation(observation), "v2")
        if not isinstance(instructions, str) or instructions != INSTRUCTIONS:
            raise ValueError("jeff_workflow_instructions_mismatch")
        ordered = _validate_actions(actions)
        return {"model": RETURNED_MODEL, "state": clean, "questions": {
            QUESTION_ID: {"type": "choice", "instructions": NATIVE_INSTRUCTIONS,
                          "criteria": {action: None for action in ordered}}
        }}

    def preflight(self, observation: Any, actions: Any) -> dict[str, Any]:
        report = self.runtime.preflight(self._body(observation, INSTRUCTIONS, actions))
        return {**report, "lossless": True}

    def choose(self, observation: Any, instructions: Any, actions: Any) -> ControllerResult:
        body = self._body(observation, instructions, actions)
        self.runtime.preflight(body)
        started = perf_counter()

        def failure(outcome: str, code: str) -> ControllerResult:
            return ControllerResult(None, outcome, (perf_counter() - started) * 1000,
                                    None, None, MODEL, error_code=code)

        with _trace(self.tracer, body) as span:
            try:
                payload = self.runtime.execute(body)
            except Exception:
                span.update(metadata={"outcome": "transport_error",
                                      "error_code": "jeff_workflow_native_failure"})
                return failure("transport_error", "jeff_workflow_native_failure")
            span.update(output=payload,
                        metadata={"output_token_semantics": "nominal_not_generated"})
        if not isinstance(payload, dict) or payload.get("model") != RETURNED_MODEL:
            return failure("version_mismatch", "jeff_workflow_model_mismatch")
        answers = payload.get("answers")
        answer = answers.get(QUESTION_ID) if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            return failure("invalid_output", "jeff_workflow_answer_invalid")
        probabilities = answer.get("probabilities")
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        ordered = list(body["questions"][QUESTION_ID]["criteria"])
        number = lambda value: (type(value) in (int, float) and math.isfinite(value)
                                and 0 <= value <= 1)
        if (not isinstance(probabilities, dict) or set(probabilities) != set(ordered)
                or not all(number(value) for value in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > PROBABILITY_SUM_TOLERANCE
                or not isinstance(choice, str) or choice not in ordered
                or probabilities[choice] != max(probabilities.values())
                or not number(confidence)):
            return failure("invalid_output", "jeff_workflow_choice_invalid")
        usage = payload.get("usage")
        if (not isinstance(usage, dict) or type(usage.get("input_tokens")) is not int
                or usage["input_tokens"] < 0):
            return failure("invalid_output", "jeff_workflow_usage_invalid")
        return ControllerResult(choice, "ok", (perf_counter() - started) * 1000,
                                usage["input_tokens"], None, MODEL,
                                probabilities=dict(probabilities), confidence=confidence,
                                returned_model=RETURNED_MODEL)

    def identity(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in self.runtime.identity.items() if key != "evidence"},
            "evidence": dict(self.runtime.identity.get("evidence", {})),
            "instruction_profile": PROFILE,
            "instructions_sha256": hashlib.sha256(NATIVE_INSTRUCTIONS.encode()).hexdigest(),
            "action_vocabulary_sha256": _digest(list(ACTIONS)),
            "latency_boundary": "synchronized_native_forward_no_network",
            "output_token_semantics": "nominal_not_generated",
            "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
        }

    def close(self) -> None:
        return None
