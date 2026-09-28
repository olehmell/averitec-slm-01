"""Protocol-v4 state, evidence, and terminal-output contracts.

These checks deliberately do not import a model.  They are used by the
preflight, local smoke, and remote runtime so that a malformed artifact cannot
be turned into a plausible-looking benchmark prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Iterable
import unicodedata


READY_WITH_EVIDENCE = "ready_with_evidence"
READY_INSUFFICIENT_EVIDENCE = "ready_insufficient_evidence"
INVALID_ARTIFACT = "invalid_artifact"
READINESS_STATES = frozenset({READY_WITH_EVIDENCE, READY_INSUFFICIENT_EVIDENCE, INVALID_ARTIFACT})
VERDICT_LABELS = frozenset({"Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"})
CONDITIONS = (
    "fixed_flow", "loop_agent", "orchestrator_agent", "graph_loop_agent",
    "graph_no_loops", "fixed_repeats",
)
ADAPTIVE_CONDITIONS = frozenset({"loop_agent", "orchestrator_agent", "graph_loop_agent"})
GRAPH_NODES = ("decompose", "search", "qa", "package", "verdict")
PROCESS_CHECKPOINT_SCHEMA = "averitec-process-checkpoint/v1"


class ContractError(ValueError):
    """A protocol input, transition, or output violates a frozen contract."""


# These rules intentionally describe a small, frozen subset of Unicode text
# segmentation rather than attempting semantic validation.  Model offsets are
# Python Unicode code-point offsets.  A boundary may not split the listed
# extended-grapheme constructions, and may not cut through Latin/Cyrillic
# words or conventional numeric tokens.  CJK text is deliberately not treated
# as one unsplittable lexical token merely because its code points are letters.
_APOSTROPHES = frozenset({"'", "\u2019", "\u02bc"})
_WORD_HYPHENS = frozenset({"-", "\u2010", "\u2011"})
_NUMERIC_DASHES = _WORD_HYPHENS | frozenset({"\u2012", "\u2013"})
_DECIMAL_SEPARATORS = frozenset({".", ",", "_", "\u066b", "\u066c"})
_NUMERIC_SIGNS = frozenset({"+", "-", "\u2212"})


def _is_variation_selector(value: str) -> bool:
    codepoint = ord(value)
    return 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF


def _is_emoji_modifier(value: str) -> bool:
    return 0x1F3FB <= ord(value) <= 0x1F3FF


def _is_regional_indicator(value: str) -> bool:
    return 0x1F1E6 <= ord(value) <= 0x1F1FF


def _hangul_jamo_class(value: str) -> str | None:
    codepoint = ord(value)
    if 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97C:
        return "L"
    if 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6:
        return "V"
    if 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB:
        return "T"
    if 0xAC00 <= codepoint <= 0xD7A3:
        return "LV" if (codepoint - 0xAC00) % 28 == 0 else "LVT"
    return None


def _is_latin_or_cyrillic_letter(value: str) -> bool:
    if not unicodedata.category(value).startswith("L"):
        return False
    name = unicodedata.name(value, "")
    return "LATIN" in name or "CYRILLIC" in name


def _is_numeric(value: str) -> bool:
    return value.isdecimal()


def _lexical_boundary_invalid(text: str, boundary: int) -> bool:
    """Reject only documented Latin/Cyrillic and numeric token interiors."""
    left_index = boundary - 1
    # A combining sequence is one grapheme, but its base remains part of the
    # surrounding lexical token after the sequence ends (Cafe\u0301teria).
    while left_index >= 0 and (unicodedata.category(text[left_index]).startswith("M")
                               or _is_variation_selector(text[left_index])):
        left_index -= 1
    left, right = text[left_index], text[boundary]
    left_letter, right_letter = _is_latin_or_cyrillic_letter(left), _is_latin_or_cyrillic_letter(right)
    left_number, right_number = _is_numeric(left), _is_numeric(right)
    if (left_number and right_number) or ((left_letter or right_letter) and (left_letter or left_number) and (right_letter or right_number)):
        return True

    def internal_connector(position: int) -> bool:
        value = text[position]
        if position == 0 or position + 1 == len(text):
            return False
        before, after = text[position - 1], text[position + 1]
        before_letter, after_letter = _is_latin_or_cyrillic_letter(before), _is_latin_or_cyrillic_letter(after)
        before_number, after_number = _is_numeric(before), _is_numeric(after)
        if value in _NUMERIC_SIGNS and after_number and not before_number:
            return True
        if value in _APOSTROPHES:
            return (before_letter and after_letter) or (before_number and after_number)
        if value in _WORD_HYPHENS:
            return (before_letter and after_letter) or (before_number and after_number)
        if value in _NUMERIC_DASHES | _DECIMAL_SEPARATORS:
            return before_number and after_number
        return False

    if boundary == 1 and text[0] in _NUMERIC_SIGNS | (_DECIMAL_SEPARATORS - {"_"}) and _is_numeric(text[1]):
        return True
    return (internal_connector(boundary - 1) if boundary > 0 else False) or internal_connector(boundary)


def span_boundary_reason(text: str, boundary: int) -> str | None:
    """Return a deterministic rejection code if ``boundary`` splits text.

    The caller supplies a Python code-point boundary.  Edges are valid; the
    function makes no attempt to expand, normalize, snap, or otherwise repair
    an offset.
    """
    if boundary <= 0 or boundary >= len(text):
        return None
    left, right = text[boundary - 1], text[boundary]
    if (left == "\r" and right == "\n") or left == "\u200d" or right == "\u200d":
        return "qa_span_grapheme_boundary"
    if _is_regional_indicator(left) and _is_regional_indicator(right):
        preceding = 0
        cursor = boundary - 1
        while cursor >= 0 and _is_regional_indicator(text[cursor]):
            preceding += 1
            cursor -= 1
        if preceding % 2:
            return "qa_span_grapheme_boundary"
    left_jamo, right_jamo = _hangul_jamo_class(left), _hangul_jamo_class(right)
    if ((left_jamo == "L" and right_jamo in {"L", "V", "LV", "LVT"})
            or (left_jamo in {"LV", "V"} and right_jamo in {"V", "T"})
            or (left_jamo in {"LVT", "T"} and right_jamo == "T")):
        return "qa_span_grapheme_boundary"
    if (unicodedata.category(right).startswith("M") or _is_variation_selector(right)
            or _is_emoji_modifier(right)):
        return "qa_span_grapheme_boundary"
    if _lexical_boundary_invalid(text, boundary):
        return "qa_span_lexical_boundary"
    return None


def span_fidelity_reason(text: str, start: int, end: int) -> str | None:
    """Check a nonempty exact span without modifying its supplied offsets."""
    answer = text[start:end]
    if answer and (answer[0].isspace() or answer[-1].isspace()):
        return "qa_span_edge_whitespace"
    return span_boundary_reason(text, start) or span_boundary_reason(text, end)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_hash(value: str) -> str:
    """Hash exact source text; no whitespace normalization is permitted."""
    return sha256_text(value)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_evidence_item(item: dict[str, Any]) -> None:
    """Validate a provenance-bound QA item and its exact inclusive/exclusive span."""
    required = {"qa_id", "facet_ids", "question", "answer", "passage_id", "url", "scraped_text", "source_hash", "span_start", "span_end"}
    if set(item) - (required | {"relevance_score", "support_score", "supporting_passage_ids", "support_validated"}):
        raise ContractError("evidence_unknown_field")
    if not required.issubset(item) or not all(_nonempty(item[key]) for key in ("qa_id", "question", "answer", "passage_id", "url", "scraped_text", "source_hash")):
        raise ContractError("evidence_required_field")
    if (not isinstance(item["facet_ids"], list) or not item["facet_ids"]
            or any(not _nonempty(value) for value in item["facet_ids"])):
        raise ContractError("evidence_facet_ids")
    start, end, text = item["span_start"], item["span_end"], item["scraped_text"]
    if (not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool)
            or isinstance(end, bool) or start < 0 or end <= start or end > len(text)):
        raise ContractError("evidence_span_bounds")
    if text[start:end] != item["answer"]:
        raise ContractError("evidence_answer_not_exact_span")
    if span_fidelity_reason(text, start, end) is not None:
        raise ContractError("evidence_span_fidelity")
    if item["source_hash"] != source_hash(text):
        raise ContractError("evidence_source_hash")
    if item.get("support_validated") is not True:
        raise ContractError("evidence_support_not_validated")


def readiness_for(evidence: Iterable[dict[str, Any]], *, invalid_reason: str | None = None) -> str:
    if invalid_reason is not None:
        return INVALID_ARTIFACT
    values = list(evidence)
    try:
        for item in values:
            validate_evidence_item(item)
    except (ContractError, TypeError):
        return INVALID_ARTIFACT
    return READY_WITH_EVIDENCE if values else READY_INSUFFICIENT_EVIDENCE


def validate_package(package: dict[str, Any]) -> str:
    if not isinstance(package, dict) or not _nonempty(package.get("case_id")) or not isinstance(package.get("evidence"), list):
        return INVALID_ARTIFACT
    declared = package.get("readiness")
    derived = readiness_for(package["evidence"], invalid_reason=package.get("invalid_reason"))
    if declared != derived or declared not in READINESS_STATES:
        return INVALID_ARTIFACT
    return derived


_VERDICT_LINE = re.compile(r"^\s*verdict\s*:\s*[\"']?(.+?)[\"']?\s*$", re.IGNORECASE)


def parse_hero_terminal_output(value: str) -> tuple[str, str]:
    """Accept exactly one terminal ``Verdict:`` line and preserve explanation.

    The parser never searches labels in prose and never asks a model to repair
    malformed output.  It accepts harmless field-case/quote variation only.
    """
    if not isinstance(value, str) or not value.strip():
        raise ContractError("verdict_output_missing")
    lines = value.strip().splitlines()
    matches = [(index, _VERDICT_LINE.match(line)) for index, line in enumerate(lines) if _VERDICT_LINE.match(line)]
    if len(matches) != 1:
        raise ContractError("verdict_terminal_line_ambiguous")
    index, match = matches[0]
    if index != len(lines) - 1:
        raise ContractError("verdict_not_terminal")
    label = match.group(1).strip().strip("\"'")
    if label not in VERDICT_LABELS:
        raise ContractError("verdict_label_invalid")
    explanation = "\n".join(lines[:index]).strip()
    if not explanation:
        raise ContractError("verdict_justification_missing")
    return label, explanation


def graph_transition(node: str, action: str) -> str:
    """Return the next graph node for the bounded evidence-feedback graph."""
    allowed = {
        "decompose": {"advance": "search"},
        "search": {"advance": "qa"},
        "qa": {"repeat_qa": "qa", "advance": "package"},
        # The one intentional back-edge occurs only after QA and coverage have
        # produced an evidence board. Query refinement can therefore depend on
        # what the system has actually read.
        "package": {"refine_search": "search", "advance": "verdict"},
        "verdict": {"finish": "verdict"},
    }
    if node not in allowed or action not in allowed[node]:
        raise ContractError("graph_transition_invalid")
    return allowed[node][action]


@dataclass
class CaseState:
    case_id: str
    claim: str
    condition: str
    facets: list[dict[str, str]] = field(default_factory=list)
    query_plan: list[str] = field(default_factory=list)
    query_history: list[str] = field(default_factory=list)
    candidate_passages: list[Any] = field(default_factory=list)
    qa_candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_evidence: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    readiness: str | None = None
    graph_node: str | None = None
    graph_repeats: dict[str, int] = field(default_factory=dict)
    stagnation: int = 0
    search_attempts: int = 0
    qa_attempts: int = 0
    no_progress_actions: set[str] = field(default_factory=set)
    controller_evidence_checkpoint: set[tuple[str, int, int]] = field(default_factory=set)
    controller_action_counts: dict[str, int] = field(default_factory=dict)
    controller_repeat_count: int = 0
    last_action_result: str = "not_started"
    terminal_reason: str | None = None

    def package(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "readiness": self.readiness, "evidence": self.selected_evidence}
