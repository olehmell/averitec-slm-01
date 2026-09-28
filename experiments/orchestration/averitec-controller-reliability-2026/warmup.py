"""One-call premeasurement warmup for controller deployments.

This module has no journal or output-file side effects.  The caller can attach
an isolated tracer if it wants a warmup trace; measured trial tracing remains
separate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from providers import ControllerResult


WARMUP_TIMEOUT_SECONDS = 120
WARMUP_ACTIONS = (
    "decompose", "queries", "retrieve", "qa", "coverage", "select", "verdict", "finish", "abort",
)
WARMUP_OBSERVATION = {
    "schema": "averitec-reliability-observation/v1",
    "stage": "decompose",
    "completed_stages": [],
    "attempts_on_stage": 0,
    "tool_status": None,
    "calls_remaining": 12,
    "metrics": {
        "facet_count": None,
        "query_count": None,
        "candidate_count": None,
        "qa_count": None,
        "selected_count": None,
    },
}


class WarmableController(Protocol):
    model: str
    timeout_seconds: float
    tracer: Any

    def choose(self, observation: dict[str, Any], instructions: str, actions: list[str]) -> ControllerResult: ...


@dataclass(frozen=True)
class WarmupReceipt:
    """Safe warmup result suitable for a deployment receipt."""

    status: str
    duration_ms: float
    usage: dict[str, int | None]
    profile: dict[str, Any] | None
    model: str
    expected_returned_model: str
    returned_model: str | None
    outcome: str
    error_code: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "duration_ms": self.duration_ms,
            "usage": dict(self.usage),
            "profile": self.profile,
            "model": self.model,
            "expected_returned_model": self.expected_returned_model,
            "returned_model": self.returned_model,
            "outcome": self.outcome,
            "error_code": self.error_code,
        }


class WarmupFailure(RuntimeError):
    """A failed warmup, carrying its safe receipt and requiring run abort."""

    def __init__(self, receipt: WarmupReceipt) -> None:
        super().__init__("controller_warmup_failed")
        self.receipt = receipt


def _profile_identity(controller: Any) -> dict[str, Any] | None:
    value = getattr(controller, "resolved_profile", None)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid_controller_profile_identity")
    return value


def premeasure_warmup(
    controller: WarmableController,
    *,
    instructions: str,
    actions: list[str],
    expected_model: str | None = None,
    tracer: Any | None = None,
) -> dict[str, Any]:
    """Make exactly one bounded, non-measured controller request.

    A syntactically valid action is enough: warmup verifies deployment and
    output acceptance, not the workflow oracle's expected action.  Any failed
    request raises :class:`WarmupFailure`; callers must let it abort before
    writing a measured intent or requesting a measured decision.
    """
    if list(actions) != list(WARMUP_ACTIONS):
        raise ValueError("warmup_requires_global_action_vocabulary")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("warmup_requires_instructions")
    model = expected_model if expected_model is not None else controller.model
    if not isinstance(model, str) or not model or controller.model != model:
        raise ValueError("warmup_model_identity_mismatch")
    returned_identity = getattr(controller, 'expected_returned_model', model)
    if not isinstance(returned_identity, str) or not returned_identity:
        raise ValueError('warmup_model_identity_mismatch')
    prior_timeout = controller.timeout_seconds
    prior_tracer = controller.tracer
    if not isinstance(prior_timeout, (int, float)) or isinstance(prior_timeout, bool) or prior_timeout <= 0:
        raise ValueError("invalid_controller_timeout")
    started = perf_counter()
    try:
        controller.timeout_seconds = WARMUP_TIMEOUT_SECONDS
        # Do not let a previously attached measured tracer receive the warmup.
        # A caller that wants observability supplies an isolated warmup tracer.
        controller.tracer = tracer
        result = controller.choose(deepcopy(WARMUP_OBSERVATION), instructions, list(actions))
    finally:
        controller.timeout_seconds = prior_timeout
        controller.tracer = prior_tracer
    receipt = WarmupReceipt(
        status="ok" if (result.outcome == "ok" and result.action in actions and result.model == model
                         and result.returned_model == returned_identity) else "failed",
        duration_ms=(perf_counter() - started) * 1000,
        usage={"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        profile=_profile_identity(controller),
        model=model,
        expected_returned_model=returned_identity,
        returned_model=result.returned_model,
        outcome=result.outcome,
        error_code=result.error_code,
    )
    if receipt.status != "ok":
        raise WarmupFailure(receipt)
    return receipt.as_dict()
