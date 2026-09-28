"""Strict metrics-only observation contract for the LFM controller.

The controller never receives ``CaseState`` directly.  This module is the
only projection from executor state to the controller wire request.  Its
output contains closed enums, booleans, finite numbers, and ``null`` only;
case text, identifiers, URLs, QA text, source text, and embeddings have no
representable field.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from averitec_fixed import url_family
from inference.contracts import (
    CaseState,
    INVALID_ARTIFACT,
    READY_INSUFFICIENT_EVIDENCE,
    READY_WITH_EVIDENCE,
    validate_evidence_item,
)
from inference.ledger import CaseLedger


CONTROLLER_OBSERVATION_SCHEMA = "averitec-controller-observation/v1"
CONTROLLER_STAGES = frozenset({
    "loop_control",
    "delegation_control",
    "researcher_control",
    "qa_specialist_control",
    "coverage_specialist_control",
    "graph_search_control",
    "graph_qa_control",
    "graph_package_control",
})
CONTROLLER_ACTIONS = frozenset({
    "decompose", "queries", "search", "qa", "coverage", "finish",
    "researcher", "qa_specialist", "coverage_specialist",
    "refine_search", "repeat_qa", "advance",
})
ACTION_RESULTS = frozenset({"not_started", "progress", "no_progress", "local_finish"})
PACKAGE_STATES = frozenset({
    "not_assessed", READY_WITH_EVIDENCE, READY_INSUFFICIENT_EVIDENCE, INVALID_ARTIFACT,
})

_TOP_LEVEL_FIELDS = frozenset({
    "schema", "route", "volume", "progress", "resources", "readiness",
})
_GROUP_FIELDS = {
    "route": frozenset({"stage", "allowed_actions", "last_action_result"}),
    "volume": frozenset({
        "total_facet_count", "query_count", "candidate_count", "valid_qa_count",
        "selected_evidence_count", "unique_source_count",
    }),
    "progress": frozenset({
        "new_evidence_count", "covered_facet_count", "coverage_ratio", "gap_count",
        "conflict_pair_count", "repeat_count", "no_progress_step_count",
    }),
    "resources": frozenset({
        "model_calls_spent", "model_calls_remaining", "controller_calls_spent",
        "input_tokens_spent", "input_tokens_remaining", "output_tokens_spent",
        "output_tokens_remaining", "search_calls_spent", "search_calls_remaining",
        "wall_seconds_remaining", "terminal_verdict_reserved",
    }),
    "readiness": frozenset({
        "evidence_package_state", "coverage_measured", "conflicts_measured",
    }),
}


def _cap(ledger: CaseLedger, key: str, fallback: int) -> int:
    value = ledger.caps.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("controller_resource_cap:" + key)
    return int(value)


def _valid_evidence_keys(state: CaseState) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for item in state.qa_candidates:
        try:
            validate_evidence_item(item)
        except (TypeError, ValueError):
            continue
        keys.add((item["source_hash"], item["span_start"], item["span_end"]))
    return keys


def _unique_source_count(state: CaseState) -> int:
    families: set[str] = set()
    for item in state.qa_candidates:
        try:
            validate_evidence_item(item)
            families.add(url_family(item["url"]))
        except (TypeError, ValueError):
            continue
    return len(families)


def controller_observation(
    state: CaseState,
    ledger: CaseLedger,
    *,
    stage: str,
    allowed_actions: set[str],
) -> dict[str, Any]:
    """Project executor state to the complete LFM-visible observation.

    ``new_evidence_count`` compares provenance-valid spans with the previous
    controller checkpoint.  Coverage/gap values remain null until the Qwen
    coverage checker has run.  Conflict metrics remain null until a dedicated
    checker exists and executes.
    """
    if stage not in CONTROLLER_STAGES:
        raise ValueError("controller_stage")
    if not allowed_actions or not allowed_actions <= CONTROLLER_ACTIONS:
        raise ValueError("controller_allowed_actions")

    evidence_keys = _valid_evidence_keys(state)
    new_evidence_count = len(evidence_keys - state.controller_evidence_checkpoint)
    state.controller_evidence_checkpoint = evidence_keys

    coverage_measured = bool(state.coverage)
    total_facets = len(state.facets)
    covered_count: int | None = None
    coverage_ratio: float | None = None
    gap_count: int | None = None
    if coverage_measured:
        covered = state.coverage.get("covered_facet_ids")
        score = state.coverage.get("score")
        if (not isinstance(covered, list)
                or any(not isinstance(item, str) for item in covered)
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0):
            raise ValueError("controller_coverage_state")
        covered_count = len(set(covered))
        if covered_count > total_facets:
            raise ValueError("controller_coverage_state")
        if total_facets > 0:
            coverage_ratio = covered_count / total_facets
            gap_count = total_facets - covered_count

    model_cap = _cap(ledger, "max_model_calls_per_case", 12)
    input_cap = _cap(ledger, "max_input_tokens_per_case", 24_576)
    output_cap = _cap(ledger, "max_output_tokens_per_case", 9_728)
    search_cap = _cap(ledger, "max_search_calls_per_case", 4)
    readiness = state.readiness if state.readiness is not None else "not_assessed"

    value = {
        "schema": CONTROLLER_OBSERVATION_SCHEMA,
        "route": {
            "stage": stage,
            "allowed_actions": sorted(allowed_actions),
            "last_action_result": state.last_action_result,
        },
        "volume": {
            "total_facet_count": total_facets,
            "query_count": len(state.query_plan),
            "candidate_count": len(state.candidate_passages),
            "valid_qa_count": len(evidence_keys),
            "selected_evidence_count": len(state.selected_evidence),
            "unique_source_count": _unique_source_count(state),
        },
        "progress": {
            "new_evidence_count": new_evidence_count,
            "covered_facet_count": covered_count,
            "coverage_ratio": coverage_ratio,
            "gap_count": gap_count,
            # No conflict checker is implemented in A2.  Null is required;
            # zero would falsely claim that the measurement ran.
            "conflict_pair_count": None,
            "repeat_count": state.controller_repeat_count,
            "no_progress_step_count": state.stagnation,
        },
        "resources": {
            "model_calls_spent": ledger.model_calls,
            "model_calls_remaining": max(0, model_cap - ledger.model_calls),
            "controller_calls_spent": ledger.controller_calls,
            "input_tokens_spent": ledger.input_tokens,
            "input_tokens_remaining": max(0, input_cap - ledger.input_tokens),
            "output_tokens_spent": ledger.output_tokens,
            "output_tokens_remaining": max(0, output_cap - ledger.output_tokens),
            "search_calls_spent": ledger.search_calls,
            "search_calls_remaining": max(0, search_cap - ledger.search_calls),
            "wall_seconds_remaining": max(0, int(ledger.remaining_seconds())),
            "terminal_verdict_reserved": ledger.finalization_available(),
        },
        "readiness": {
            "evidence_package_state": readiness,
            "coverage_measured": coverage_measured,
            "conflicts_measured": False,
        },
    }
    validate_controller_observation(value)
    return value


def validate_controller_observation(value: Any) -> None:
    """Reject unknown fields, open strings, invalid numbers, and enum drift."""
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("controller_observation_schema")
    if value.get("schema") != CONTROLLER_OBSERVATION_SCHEMA:
        raise ValueError("controller_observation_schema")
    for group, fields in _GROUP_FIELDS.items():
        if not isinstance(value.get(group), dict) or set(value[group]) != fields:
            raise ValueError("controller_observation_schema")

    route = value["route"]
    if (route["stage"] not in CONTROLLER_STAGES
            or route["last_action_result"] not in ACTION_RESULTS
            or not isinstance(route["allowed_actions"], list)
            or not route["allowed_actions"]
            or route["allowed_actions"] != sorted(set(route["allowed_actions"]))
            or not set(route["allowed_actions"]) <= CONTROLLER_ACTIONS):
        raise ValueError("controller_observation_enum")

    if value["readiness"]["evidence_package_state"] not in PACKAGE_STATES:
        raise ValueError("controller_observation_enum")
    if not isinstance(value["readiness"]["coverage_measured"], bool) or value["readiness"]["conflicts_measured"] is not False:
        raise ValueError("controller_observation_schema")

    nullable = {"covered_facet_count", "coverage_ratio", "gap_count", "conflict_pair_count"}
    for group in ("volume", "progress", "resources"):
        for key, item in value[group].items():
            if group == "resources" and key == "terminal_verdict_reserved":
                if not isinstance(item, bool):
                    raise ValueError("controller_observation_number")
                continue
            if key in nullable and item is None:
                continue
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or item < 0:
                raise ValueError("controller_observation_number")
    ratio = value["progress"]["coverage_ratio"]
    if ratio is not None and not 0.0 <= ratio <= 1.0:
        raise ValueError("controller_observation_number")
    if value["progress"]["conflict_pair_count"] is not None or value["readiness"]["conflicts_measured"] is not False:
        raise ValueError("controller_conflict_not_implemented")
    if value["readiness"]["coverage_measured"]:
        if value["volume"]["total_facet_count"] > 0 and any(value["progress"][key] is None for key in ("covered_facet_count", "coverage_ratio", "gap_count")):
            raise ValueError("controller_coverage_measurement")
    elif any(value["progress"][key] is not None for key in ("covered_facet_count", "coverage_ratio", "gap_count")):
        raise ValueError("controller_coverage_measurement")


def canonical_observation(value: dict[str, Any]) -> str:
    validate_controller_observation(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def observation_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_observation(value).encode("utf-8")).hexdigest()
