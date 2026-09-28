"""Offline, gold-free assembly of frozen selector evidence packages.

This module makes no model calls. Only the seven explicitly allowed candidate
fields enter a package; reference grades and verdicts are never copied.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


PASSAGE_FIELDS = (
    "id", "passage_id", "text", "url", "source_start",
    "source_text_length", "source_text_sha256",
)
MODES = frozenset({"selector", "include_all", "include_none"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_candidates(document: Mapping[str, Any], expected_cases: int, candidates_per_case: int) -> list[dict]:
    if not isinstance(document, Mapping) or document.get("gold_included") is not False:
        raise ValueError("candidates_must_be_gold_free")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError("candidate_case_count")
    seen_cases: set[str] = set()
    validated: list[dict] = []
    for case in cases:
        if not isinstance(case, Mapping) or any(not _nonempty_string(case.get(key)) for key in ("case_id", "group_id", "claim")):
            raise ValueError("invalid_candidate_case")
        case_id = case["case_id"]
        if case_id in seen_cases:
            raise ValueError("duplicate_candidate_case_id")
        seen_cases.add(case_id)
        candidates = case.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != candidates_per_case:
            raise ValueError(f"candidate_count:{case_id}")
        seen_ids: set[str] = set()
        passages = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or any(not _nonempty_string(candidate.get(key)) for key in ("id", "passage_id", "text", "url")):
                raise ValueError(f"invalid_candidate:{case_id}")
            if (type(candidate.get("source_start")) is not int or candidate["source_start"] < 0
                    or type(candidate.get("source_text_length")) is not int or candidate["source_text_length"] < 0
                    or not isinstance(candidate.get("source_text_sha256"), str)
                    or not _HEX64.fullmatch(candidate["source_text_sha256"])):
                raise ValueError(f"invalid_source_metadata:{case_id}")
            candidate_id = candidate["id"]
            if candidate_id in seen_ids:
                raise ValueError(f"duplicate_candidate_id:{case_id}")
            seen_ids.add(candidate_id)
            passages.append({key: candidate[key] for key in PASSAGE_FIELDS})
        validated.append({"case_id": case_id, "group_id": case["group_id"], "claim": case["claim"], "candidates": passages})
    return validated


def build_packages(
    candidates_document: Mapping[str, Any],
    measured_rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    mode: str = "selector",
    expected_cases: int = 100,
    candidates_per_case: int = 10,
) -> list[dict]:
    """Build one status record per planned case in frozen candidate order.

    Selector failures carry ``selected_passages=None`` and may not be sent to
    conversion. A valid all-exclude case carries an empty list and is ready.
    The count arguments permit small synthetic fixtures; production defaults
    enforce the frozen 100-by-10 study shape.
    """
    if mode not in MODES:
        raise ValueError("unknown_package_mode")
    if type(expected_cases) is not int or expected_cases < 1 or type(candidates_per_case) is not int or candidates_per_case < 1:
        raise ValueError("invalid_expected_shape")
    cases = _validate_candidates(candidates_document, expected_cases, candidates_per_case)
    by_case = {case["case_id"]: {candidate["id"] for candidate in case["candidates"]} for case in cases}
    if mode != "selector" and measured_rows is not None:
        raise ValueError("control_modes_do_not_accept_measured_rows")
    if mode == "selector" and measured_rows is None:
        raise ValueError("selector_rows_required")

    rows_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if mode == "selector":
        for row in measured_rows:  # type: ignore[union-attr]
            if not isinstance(row, Mapping):
                raise ValueError("invalid_measured_row")
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or case_id not in by_case:
                raise ValueError(f"unknown_measured_case_id:{case_id}")
            rows_by_case[case_id].append(row)

    packages = []
    for case in cases:
        case_id = case["case_id"]
        candidates = case["candidates"]
        errors: list[str] = []
        selected: list[dict] = []
        if mode == "selector":
            rows = rows_by_case[case_id]
            counts = Counter(row.get("id") for row in rows if isinstance(row.get("id"), str))
            for row in rows:
                candidate_id = row.get("id")
                if not isinstance(candidate_id, str) or candidate_id not in by_case[case_id]:
                    errors.append(f"unknown_candidate_id:{candidate_id}")
            for candidate in candidates:
                candidate_id = candidate["id"]
                count = counts[candidate_id]
                if count == 0:
                    errors.append(f"missing_candidate_id:{candidate_id}")
                    continue
                if count != 1:
                    errors.append(f"duplicate_candidate_id:{candidate_id}")
                    continue
                row = next(row for row in rows if row.get("id") == candidate_id)
                if row.get("outcome") != "ok":
                    errors.append(f"invalid_outcome:{candidate_id}")
                if row.get("action") not in ("include", "exclude"):
                    errors.append(f"invalid_action:{candidate_id}")
                if row.get("outcome") == "ok" and row.get("action") == "include":
                    selected.append(candidate.copy())
        elif mode == "include_all":
            selected = [candidate.copy() for candidate in candidates]

        packages.append({
            "schema": "three-pass-selected-passages/v1",
            "case_id": case_id,
            "group_id": case["group_id"],
            "claim": case["claim"],
            "mode": mode,
            "status": "failed" if errors else "ready",
            "errors": errors,
            "selected_passages": None if errors else selected,
        })
    return packages
