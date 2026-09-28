"""Protocol-v4 shared state/tools and four bounded orchestration policies."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable
import unicodedata

from averitec_fixed import Passage
from inference.controller_contract import (
    CONTROLLER_OBSERVATION_SCHEMA,
    canonical_observation,
    controller_observation,
    observation_sha256,
)
from inference.contracts import (CONDITIONS, ContractError, CaseState, INVALID_ARTIFACT,
    PROCESS_CHECKPOINT_SCHEMA, READY_INSUFFICIENT_EVIDENCE, READY_WITH_EVIDENCE, graph_transition,
    parse_hero_terminal_output, readiness_for, source_hash, span_fidelity_reason,
    validate_evidence_item)
from inference.ledger import BudgetExceeded, CaseLedger
from inference.tools import ClosedWorldTools, LazyDenseEmbedder


@dataclass(frozen=True)
class CompletionResult:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


Completion = Callable[..., str | CompletionResult]


class PolicyError(ValueError): pass


def _limit(ledger: CaseLedger, key: str, fallback: int) -> int:
    value = ledger.caps.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 1:
        raise PolicyError("policy_cap:" + key)
    return int(value)


def _call(complete: Completion, ledger: CaseLedger, prompt: str, maximum: int, *, role: str,
          repair: bool = False, json_schema: dict[str, Any] | None = None) -> str:
    reserved = ledger.reserve_model(prompt, maximum, role=role, repair=repair)
    try:
        timeout = max(1, int(ledger.remaining_seconds()))
        # Conventional synthetic callbacks deliberately remain a two-argument
        # interface.  Only the local transport advertises schema support.
        if json_schema is not None and getattr(complete, "accepts_json_schema", False):
            result = complete(prompt, maximum, timeout, json_schema=json_schema)
        elif getattr(complete, "accepts_timeout", False):
            result = complete(prompt, maximum, timeout)
        else:
            result = complete(prompt, maximum)
    except Exception:
        ledger.event("completion_failed", role=role, repair=repair)
        raise
    receipt = result if isinstance(result, CompletionResult) else CompletionResult(result)
    if not isinstance(receipt.content, str): raise PolicyError("model_completion_schema")
    # The ledger's public event carries only a bounded semantic summary.  An
    # opted-in private trace receives the exact completion body, including a
    # repair response, but never this prompt or transport configuration.
    ledger.private_completion(receipt.content, role=role, repair=repair)
    ledger.settle_model(maximum, receipt.content, reserved_input=reserved, reported_input=receipt.input_tokens, reported_output=receipt.output_tokens)
    ledger.event("completion", role=role, repair=repair, content_length=len(receipt.content), finish_reason=receipt.finish_reason)
    return receipt.content


JsonContract = Callable[[dict[str, Any]], str | None]


QA_CONTRACT_VERSION = "semantic-span-fidelity/v4-segments"

# A segment is a source-bound, deterministic unit the model can select but
# never edit.  Sentences are kept whole whenever possible; exceptionally long
# sentences are split only at an already-valid whitespace/punctuation
# boundary.  This keeps an arbitrary character offset (and therefore a
# one-character partial word) out of the model-to-provenance boundary.
_QA_SEGMENT_MAX_CHARS = 400
_QA_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "u.s", "u.k", "inc", "ltd", "co", "corp", "fig", "no",
})
_QA_TERMINAL_TRAILERS = frozenset("\"'”’)]}")


def _object(properties: dict[str, Any], required: list[str], *, title: str) -> dict[str, Any]:
    return {"title": title, "type": "object", "properties": properties,
            "required": required, "additionalProperties": False}


def _facets_schema(maximum: int) -> dict[str, Any]:
    facet = _object({"id": {"type": "string", "minLength": 1},
                     "text": {"type": "string", "minLength": 1}}, ["id", "text"], title="facet")
    return _object({"facets": {"type": "array", "minItems": 1, "maxItems": maximum, "items": facet}}, ["facets"], title="facets_response")


def _hyde_schema(maximum: int, required_hyde: int | None) -> dict[str, Any]:
    values: dict[str, Any] = {"type": "array", "maxItems": maximum,
                              "items": {"type": "string", "minLength": 1}}
    if required_hyde is not None:
        values["minItems"] = required_hyde
    return _object({"hyde": values}, ["hyde"], title="hyde_response")


def _qa_schema(maximum: int, facet_ids: set[str], segment_ids: set[str]) -> dict[str, Any]:
    # A real QA needs at least one facet and one source-bound segment.  Empty
    # enum domains are not compilable by pinned xgrammar, so represent this
    # fail-closed boundary as an array that can contain no QA at all instead.
    # The harmless item schema is unreachable under maxItems=0.
    if not facet_ids or not segment_ids:
        return _object({
            "contract_version": {"type": "string", "enum": [QA_CONTRACT_VERSION]},
            "qas": {"type": "array", "minItems": 0, "maxItems": 0,
                    "items": {"type": "string"}},
        }, ["contract_version", "qas"], title="qa_response")
    identifier = {"type": "string", "enum": sorted(facet_ids)}
    segment = {"type": "string", "enum": sorted(segment_ids)}
    qa = _object({"facet_ids": {"type": "array", "minItems": 1, "items": identifier},
                  "question": {"type": "string", "minLength": 1},
                  # vLLM's xgrammar backend rejects uniqueItems with HTTP 400.
                  # Duplicate IDs remain fail-closed in generate_qa below.
                  "segment_ids": {"type": "array", "minItems": 1, "maxItems": 3,
                                  "items": segment},
                  "support_validated": {"type": "boolean"},
                  "support_score": {"type": "number", "minimum": 0, "maximum": 1},
                  "relevance_score": {"type": "number", "minimum": 0, "maximum": 1}},
                 ["facet_ids", "question", "segment_ids", "support_validated", "support_score", "relevance_score"], title="qa")
    return _object({"contract_version": {"type": "string", "enum": [QA_CONTRACT_VERSION]},
                    "qas": {"type": "array", "maxItems": maximum, "items": qa}},
                   ["contract_version", "qas"], title="qa_response")


def _qa_contract(data: dict[str, Any], maximum: int) -> str | None:
    """Reject every legacy or expanded model QA root before materialization."""
    if set(data) != {"contract_version", "qas"}:
        return "qa_root_schema"
    if data["contract_version"] != QA_CONTRACT_VERSION:
        return "qa_contract_version"
    qas = data["qas"]
    if not isinstance(qas, list) or len(qas) > maximum:
        return "qa_schema"
    required = {"facet_ids", "question", "segment_ids",
                "support_validated", "support_score", "relevance_score"}
    if any(not isinstance(qa, dict) or set(qa) != required for qa in qas):
        return "qa_item_schema"
    return None


def _coverage_schema(facet_ids: set[str]) -> dict[str, Any]:
    # Coverage is normalized through set() after parsing; do not emit the
    # xgrammar-unsupported uniqueItems keyword in the wire schema.
    # An empty facet domain has only one valid covered-ID list: [].  Do not
    # encode it as an empty enum, which pinned xgrammar cannot compile.
    if not facet_ids:
        return _object({"covered_facet_ids": {"type": "array", "minItems": 0,
                                                "maxItems": 0,
                                                "items": {"type": "string"}},
                        "gaps": {"type": "array", "maxItems": 0,
                                 "items": {"type": "string", "minLength": 1}}},
                       ["covered_facet_ids", "gaps"], title="coverage_response")
    return _object({"covered_facet_ids": {"type": "array",
                                             "items": {"type": "string", "enum": sorted(facet_ids)}},
                    "gaps": {"type": "array", "maxItems": len(facet_ids),
                             "items": {"type": "string", "minLength": 1}}},
                   ["covered_facet_ids", "gaps"], title="coverage_response")


def _controller_schema(allowed: set[str]) -> dict[str, Any]:
    return _object({"action": {"type": "string", "enum": sorted(allowed)}}, ["action"], title="controller_response")


def production_structured_output_schemas() -> tuple[tuple[str, dict[str, Any]], ...]:
    """Enumerate every schema family sent to the Qwen structured endpoint.

    The values which vary by case (facet and segment identifiers) are replaced
    by deterministic, valid representatives.  They exercise the same JSON
    Schema keywords and nesting as a real request without requiring a claim,
    corpus, model, or network connection.  Keep every controller action set
    here: a new call site must become a new preflight entry as well.
    """
    facet_ids = {"f1", "f2", "f3", "f4"}
    segment_ids = {"passage-a::segment-001", "passage-a::segment-002", "passage-a::segment-003"}
    return (
        ("facets_response", _facets_schema(4)),
        ("hyde_response", _hyde_schema(8, None)),
        ("fixed_hyde_response", _hyde_schema(8, 4)),
        ("qa_response", _qa_schema(10, facet_ids, segment_ids)),
        ("qa_response_empty_enums", _qa_schema(10, set(), set())),
        ("coverage_response", _coverage_schema(facet_ids)),
        ("coverage_response_empty_facets", _coverage_schema(set())),
        ("controller_loop", _controller_schema({"decompose", "queries", "search", "qa", "coverage", "finish"})),
        ("controller_orchestrator", _controller_schema({"researcher", "qa_specialist", "coverage_specialist", "finish"})),
        ("controller_researcher", _controller_schema({"queries", "search", "refine_search", "finish"})),
        ("controller_qa_specialist", _controller_schema({"qa", "finish"})),
        ("controller_coverage_specialist", _controller_schema({"coverage", "finish"})),
        ("controller_graph_qa", _controller_schema({"repeat_qa", "advance"})),
        ("controller_graph_package", _controller_schema({"refine_search", "advance"})),
    )


def _json(complete: Completion, ledger: CaseLedger, prompt: str, maximum: int, *, role: str,
          required: set[str], contract: JsonContract | None = None,
          json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return JSON only after its stage-local contract has passed.

    The repair prompt carries an allowlisted reason code, never the rejected
    completion.  This makes recoverable schema failures visible in the ledger
    while keeping raw model output out of diagnostic artifacts.
    """
    raw = _call(complete, ledger, prompt, maximum, role=role, json_schema=json_schema)
    for attempt in range(2):
        try:
            value = json.loads(raw)
            if not isinstance(value, dict) or not required.issubset(value):
                raise PolicyError("generation_schema")
            reason = contract(value) if contract is not None else None
            if reason is not None:
                raise PolicyError(reason)
            return value
        except json.JSONDecodeError:
            reason = "generation_schema"
        except PolicyError as error:
            reason = str(error)
        except (TypeError, ValueError):
            reason = "generation_schema"
        if attempt:
            raise PolicyError(reason)
        ledger.event("json_contract_repair", role=role, reason=reason)
        raw = _call(complete, ledger, "Repair only JSON; correct " + reason + "; return required keys " + ", ".join(sorted(required)) + ".\n" + prompt, maximum, role=role, repair=True, json_schema=json_schema)
    raise AssertionError


def _facets_contract(data: dict[str, Any], maximum: int) -> str | None:
    values = data["facets"]
    if not isinstance(values, list) or not 1 <= len(values) <= maximum:
        return "facets_schema"
    identifiers: list[str] = []
    for index, item in enumerate(values, 1):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            return "facets_schema"
        identifier = item.get("id", f"f{index}")
        if not isinstance(identifier, str) or not identifier.strip():
            return "facets_schema"
        identifiers.append(identifier.strip())
    return "facets_duplicate_id" if len(set(identifiers)) != len(identifiers) else None


def _hyde_contract(data: dict[str, Any], maximum: int, required_hyde: int | None) -> str | None:
    values = data["hyde"]
    if (not isinstance(values, list) or len(values) > maximum
            or (required_hyde is not None and len(values) != required_hyde)
            or any(not isinstance(item, str) or not item.strip() for item in values)):
        return "query_plan_schema"
    return None


def _tools(corpus: list[Passage], blocked: set[str], ledger: CaseLedger, embed: Callable[[list[str]], Any] | None) -> ClosedWorldTools:
    return ClosedWorldTools(
        corpus,
        blocked,
        ledger,
        LazyDenseEmbedder(embed),
        sparse_limit=_limit(ledger, "sparse_limit", 10_000),
        dense_limit=_limit(ledger, "dense_limit", 20),
    )


def decompose_claim(state: CaseState, complete: Completion, ledger: CaseLedger) -> None:
    maximum = _limit(ledger, "max_facets", 4)
    value = _json(complete, ledger, "Return JSON {\"facets\":[{\"id\":\"f1\",\"text\":\"...\"}]}. Decompose factual facets without judging truth. Claim: " + state.claim, 500, role="generation", required={"facets"}, contract=lambda data: _facets_contract(data, maximum), json_schema=_facets_schema(maximum))["facets"]
    if not isinstance(value, list) or not 1 <= len(value) <= maximum: raise PolicyError("facets_schema")
    facets = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip(): raise PolicyError("facets_schema")
        identifier = item.get("id", f"f{index}")
        if not isinstance(identifier, str) or not identifier.strip(): raise PolicyError("facets_schema")
        facets.append({"id": identifier.strip(), "text": item["text"].strip()})
    if len({item["id"] for item in facets}) != len(facets): raise PolicyError("facets_duplicate_id")
    state.facets = facets; ledger.event("tool_state", tool="decompose_claim", facet_ids=[item["id"] for item in facets])


def formulate_queries(state: CaseState, complete: Completion, ledger: CaseLedger, *, required_hyde: int | None = None) -> None:
    evidence_board = [{key: item[key] for key in ("facet_ids", "question", "answer", "passage_id")}
                      for item in state.qa_candidates[:_limit(ledger, "max_qas", 10)]]
    prior_queries = list(dict.fromkeys([*state.query_history, *state.query_plan]))
    context = {"claim": state.claim, "facets": state.facets, "query_history": prior_queries,
        "accepted_evidence": evidence_board, "gaps": state.gaps,
        "observed_candidate_ids": [str(item.passage_id) for item in state.candidate_passages]}
    maximum = _limit(ledger, "max_hyde_expansions", 8)
    if required_hyde is not None and required_hyde > maximum:
        raise PolicyError("fixed_hyde_cap")
    count_instruction = f"Write exactly {required_hyde}" if required_hyde is not None else f"Write at most {maximum}"
    value = _json(complete, ledger, "Return JSON {\"hyde\":[\"...\"]}. " + count_instruction + " fact-check retrieval hypotheses. Use accepted evidence as bounded memory. Refine searches toward unresolved or contradictory facts, and do not merely restate facts already established; hypotheses are not evidence. Context: " + json.dumps(context), 700, role="generation", required={"hyde"}, contract=lambda data: _hyde_contract(data, maximum, required_hyde), json_schema=_hyde_schema(maximum, required_hyde))["hyde"]
    if (not isinstance(value, list) or len(value) > maximum or (required_hyde is not None and len(value) != required_hyde)
            or any(not isinstance(item, str) or not item.strip() for item in value)):
        raise PolicyError("query_plan_schema")
    proposed = [item.strip() for item in value if item.strip() and item.strip() != state.claim]
    novel = [item for item in proposed if item not in prior_queries]
    previous = tuple(state.query_plan)
    # Retrieval uses only the current hypotheses. Historical queries remain in
    # bounded prompt memory but are not averaged into every later dense query,
    # where they would dilute a targeted evidence-conditioned refinement.
    state.query_plan = list(dict.fromkeys([state.claim, *proposed]))[:1 + maximum]
    state.query_history = list(dict.fromkeys([*prior_queries, *state.query_plan]))[:1 + 2 * maximum]
    ledger.event(
        "tool_state", tool="formulate_queries", query_count=len(state.query_plan),
        fixed_hyde_count=required_hyde, novel_hypotheses=len(novel),
        state_changed=tuple(state.query_plan) != previous,
    )


def search_sparse_dense(state: CaseState, tools: ClosedWorldTools) -> None:
    if not state.query_plan: raise PolicyError("query_plan_missing")
    state.search_attempts += 1
    state.candidate_passages = tools.hybrid(state.query_plan); tools.ledger.event("tool_state", tool="search_dense", candidate_count=len(state.candidate_passages), search_attempts=state.search_attempts)


def _qa_passage_bindings(state: CaseState, maximum: int) -> dict[str, Passage]:
    """Return unambiguous, source-verified model windows or stop before a call."""
    passages: dict[str, Passage] = {}
    for source in state.candidate_passages[:maximum]:
        passage_id, source_text = source.passage_id, source.source_text
        if (not isinstance(passage_id, str) or not passage_id
                or not isinstance(source.text, str) or not isinstance(source_text, str)
                or not isinstance(source.source_start, int) or isinstance(source.source_start, bool)
                or source.source_start < 0
                or source.source_start + len(source.text) > len(source_text)
                or source_text[source.source_start:source.source_start + len(source.text)] != source.text):
            raise PolicyError("qa_source_window_mismatch")
        if passage_id in passages:
            raise PolicyError("qa_ambiguous_passage_binding")
        passages[passage_id] = source
    return passages


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Return exact non-whitespace bounds without altering source text."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        return None
    # Quotation marks and terminal punctuation alone cannot carry a fact.
    # Keep letters, numbers, symbols (including emoji), and exact offsets.
    if not any(not value.isspace() and not unicodedata.category(value).startswith("P")
               for value in text[start:end]):
        return None
    return start, end


def _sentence_terminal_end(text: str, index: int) -> int:
    """Include terminal punctuation and directly attached closing delimiters."""
    end = index + 1
    while end < len(text) and text[end] in ".!?":
        end += 1
    while end < len(text) and text[end] in _QA_TERMINAL_TRAILERS:
        end += 1
    return end


def _period_is_abbreviation_or_decimal(text: str, index: int) -> bool:
    if index > 0 and index + 1 < len(text) and text[index - 1].isdecimal() and text[index + 1].isdecimal():
        return True
    start = index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    token = text[start:index]
    lowered = token.casefold()
    if lowered in _QA_ABBREVIATIONS:
        return True
    initials = token.split(".")
    return ((len(token) == 1 and token.isalpha() and token.isupper())
            or (len(initials) > 1 and all(len(part) == 1 and part.isalpha() for part in initials)))


def _reliable_sentence_terminal(text: str, index: int) -> bool:
    """Conservatively decide whether punctuation ends a sentence.

    Periods inside decimals, common abbreviations, and initialisms are never
    sentence breaks.  Elsewhere a break needs end-of-text or whitespace plus a
    plausible uppercase next sentence, avoiding eager splits at prose periods.
    """
    if text[index] == "." and _period_is_abbreviation_or_decimal(text, index):
        return False
    end = _sentence_terminal_end(text, index)
    if end == len(text):
        return True
    if not text[end].isspace():
        return False
    while end < len(text) and text[end].isspace():
        end += 1
    return end == len(text) or text[end].isupper()


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return conservative sentence spans without altering the source string."""
    spans: list[tuple[int, int]] = []
    start = 0
    for index, value in enumerate(text):
        if value in ".!?" and _reliable_sentence_terminal(text, index):
            end = _sentence_terminal_end(text, index)
            spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _short_segment_boundary(text: str, start: int, end: int) -> int | None:
    """Find the latest safe natural break in a long sentence, if one exists."""
    upper = min(start + _QA_SEGMENT_MAX_CHARS, end)
    for boundary in range(upper, start, -1):
        # Break before whitespace so the previous segment never gains a
        # trailing whitespace edge.  A punctuation break is also exact.
        natural = boundary < end and text[boundary].isspace()
        natural = natural or (boundary > start and text[boundary - 1] in ".;,:!?")
        if natural and span_fidelity_reason(text, start, boundary) is None:
            return boundary
    return None


def _window_segment_spans(text: str) -> list[tuple[int, int]]:
    """Deterministically partition a QA window into selectable source spans.

    Returned offsets are Python Unicode character offsets into ``text``.  The
    only omitted characters are inter-segment whitespace; materializing a
    multi-segment choice still spans that untouched source interval exactly.
    An unbreakable overlong token remains one segment rather than being cut at
    an arbitrary character boundary.
    """
    segments: list[tuple[int, int]] = []
    for sentence_start, sentence_end in _sentence_spans(text):
        span = _trimmed_span(text, sentence_start, sentence_end)
        if span is None:
            continue
        start, end = span
        while end - start > _QA_SEGMENT_MAX_CHARS:
            boundary = _short_segment_boundary(text, start, end)
            if boundary is None:
                break
            chunk = _trimmed_span(text, start, boundary)
            if chunk is None:
                break
            segments.append(chunk)
            start = boundary
            while start < end and text[start].isspace():
                start += 1
        final = _trimmed_span(text, start, end)
        if final is not None:
            segments.append(final)
    return segments


def _qa_segment_bindings(passages: dict[str, Passage]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Bind stable segment IDs to one verified passage and exact window offsets."""
    bindings: dict[str, dict[str, Any]] = {}
    by_passage: dict[str, list[dict[str, Any]]] = {}
    for passage_id, source in passages.items():
        passage_segments: list[dict[str, Any]] = []
        for ordinal, (window_start, window_end) in enumerate(_window_segment_spans(source.text), 1):
            segment_id = f"{passage_id}::segment-{ordinal:03d}"
            if segment_id in bindings:
                raise PolicyError("qa_duplicate_segment_binding")
            segment = {"segment_id": segment_id, "passage_id": passage_id, "source": source,
                       "window_start": window_start, "window_end": window_end}
            bindings[segment_id] = segment
            passage_segments.append(segment)
        by_passage[passage_id] = passage_segments
    return bindings, by_passage


def generate_qa(state: CaseState, complete: Completion, ledger: CaseLedger) -> None:
    state.qa_attempts += 1
    max_candidates = _limit(ledger, "max_qa_context_passages", 10)
    # The model sees only bounded, provenance-verified retrieval segments.
    # It chooses stable IDs; materialization never searches, normalizes,
    # clips, or otherwise recovers model text.
    passages = _qa_passage_bindings(state, max_candidates)
    segment_bindings, segments_by_passage = _qa_segment_bindings(passages)
    sources = [{"passage_id": identifier,
                "segments": [{"segment_id": segment["segment_id"],
                              "text": item.text[segment["window_start"]:segment["window_end"]]}
                             for segment in segments_by_passage[identifier]]}
               for identifier, item in passages.items()]
    # This identifies the exact ranked model input window without retaining
    # either prompt text or source text in the private diagnostic.
    ledger.private_qa_input([{"rank": rank, "passage_id": identifier,
                              "source_hash": source_hash(item.source_text),
                              "window_sha256": source_hash(item.text),
                              "source_window_start": item.source_start,
                              "window_length": len(item.text)}
                             for rank, (identifier, item) in enumerate(passages.items(), 1)],
                            contract_version=QA_CONTRACT_VERSION)
    prompt = (
        "Return JSON {\"contract_version\":\"semantic-span-fidelity/v4-segments\",\"qas\":[{\"facet_ids\":[\"f1\"],\"question\":\"...\",\"segment_ids\":[\"...\"],\"support_validated\":true,\"support_score\":0.75,\"relevance_score\":0.75}]}. "
        "Examine all supplied passages for distinct facts needed to verify the claim. "
        "Use Prior accepted evidence as memory: return only facts that add missing coverage, materially corroborate a weak fact, or preserve a genuine contradiction. "
        "Do not repeat a fact merely because another source or wording states it. "
        "Return a separate QA for each distinct answerable fact, up to the configured maximum. "
        "Do not stop after the first valid QA. Include relevant refuting or conflicting facts. "
        "Multiple QAs may address one facet or use the same passage when their answer spans establish different facts. "
        "Select the shortest sufficient contiguous span first, then write its question. Include any needed negation, condition, unit, time, and subject-action relation. "
        "Omit a QA when no single sufficient span exists in its window; there is no quota. "
        "For every QA, select one to three adjacent segment_ids in their presented order from one passage. "
        "Do not return answer text, character offsets, passage IDs, or any keys outside this v4 schema. "
        "Mark support_validated true only when the exact source span from the first through last selected segment answers the stated question; otherwise omit it. "
        "Return an empty array when no supported, marginally useful QA exists. Claim: " + state.claim
        + "\nFacets: " + json.dumps(state.facets)
        + "\nUnresolved gaps: " + json.dumps(state.gaps)
        + "\nPrior accepted evidence: " + json.dumps([
            {key: item[key] for key in ("facet_ids", "question", "answer", "passage_id")}
            for item in state.qa_candidates[:_limit(ledger, "max_qas", 10)]
        ])
        + "\nSources: " + json.dumps(sources)
    )
    max_qas = _limit(ledger, "max_qas", 10)
    values = _json(complete, ledger, prompt, 1400, role="generation",
                   required={"contract_version", "qas"},
                   contract=lambda data: _qa_contract(data, max_qas),
                   json_schema=_qa_schema(max_qas, {item["id"] for item in state.facets}, set(segment_bindings)))["qas"]
    if not isinstance(values, list) or len(values) > max_qas: raise PolicyError("qa_schema")
    valid: list[dict[str, Any]] = []
    for number, qa in enumerate(values, 1):
        selected_ids = qa.get("segment_ids") if isinstance(qa, dict) else None

        def reject(reason: str, *, passage_id: str | None = None,
                   absolute_start: int | None = None, absolute_end: int | None = None) -> None:
            ledger.private_qa_decision(
                returned_index=number, accepted=False, reason=reason,
                contract_version=QA_CONTRACT_VERSION,
                passage_id=passage_id,
                absolute_start=absolute_start, absolute_end=absolute_end,
            )

        if not isinstance(qa, dict):
            reject("qa_not_object")
            continue
        facets, question = qa.get("facet_ids"), qa.get("question")
        if not isinstance(selected_ids, list) or not 1 <= len(selected_ids) <= 3:
            reject("qa_segment_count")
            continue
        if any(not isinstance(segment_id, str) for segment_id in selected_ids):
            reject("qa_segment_id_type")
            continue
        if len(set(selected_ids)) != len(selected_ids):
            reject("qa_duplicate_segments")
            continue
        if any(segment_id not in segment_bindings for segment_id in selected_ids):
            reject("qa_unknown_segment")
            continue
        selected = [segment_bindings[segment_id] for segment_id in selected_ids]
        passage_ids = {segment["passage_id"] for segment in selected}
        if len(passage_ids) != 1:
            reject("qa_segments_cross_passage")
            continue
        passage_id = selected[0]["passage_id"]
        ordered = segments_by_passage[passage_id]
        positions = [ordered.index(segment) for segment in selected]
        if positions != sorted(positions):
            reject("qa_segments_out_of_order", passage_id=passage_id)
            continue
        if positions != list(range(positions[0], positions[0] + len(positions))):
            reject("qa_segments_noncontiguous", passage_id=passage_id)
            continue
        if not isinstance(facets, list) or not facets:
            reject("qa_facets_missing", passage_id=passage_id)
            continue
        elif not isinstance(question, str) or not question.strip():
            reject("qa_question_missing", passage_id=passage_id)
            continue
        if any(not isinstance(facet, str) or facet not in {item["id"] for item in state.facets} for facet in facets):
            reject("qa_unknown_facet", passage_id=passage_id)
            continue
        if qa.get("support_validated") is not True:
            reject("qa_support_unvalidated", passage_id=passage_id)
            continue
        relevance, support = qa.get("relevance_score"), qa.get("support_score")
        if (not isinstance(relevance, (int, float)) or isinstance(relevance, bool)
                or not isinstance(support, (int, float)) or isinstance(support, bool)
                or not 0 <= relevance <= 1 or not 0 <= support <= 1):
            reject("qa_score_invalid", passage_id=passage_id)
            continue
        source = selected[0]["source"]
        window_start, window_end = selected[0]["window_start"], selected[-1]["window_end"]
        absolute_start, absolute_end = source.source_start + window_start, source.source_start + window_end
        answer = source.text[window_start:window_end]
        if not answer:
            reject("qa_segment_empty", passage_id=passage_id,
                   absolute_start=absolute_start, absolute_end=absolute_end)
            continue
        # Check both coordinate systems.  The raw-source check prevents a
        # retrieved window edge from concealing a split word or grapheme.
        reason = span_fidelity_reason(source.text, window_start, window_end)
        if reason is None:
            reason = span_fidelity_reason(source.source_text, absolute_start, absolute_end)
        if reason is not None:
            reject(reason, passage_id=passage_id,
                   absolute_start=absolute_start, absolute_end=absolute_end)
            continue
        identifier = source_hash("\x1f".join((passage_id, str(absolute_start), answer, question, ",".join(sorted(facets)))))[:24]
        item = {"qa_id": f"{state.case_id}-qa-{identifier}", "facet_ids": facets, "question": question, "answer": answer, "passage_id": passage_id, "url": source.url, "scraped_text": source.source_text, "source_hash": source_hash(source.source_text), "span_start": absolute_start, "span_end": absolute_end, "relevance_score": relevance, "support_score": support, "support_validated": True}
        try:
            validate_evidence_item(item)
        except (ContractError, TypeError, ValueError):
            reject("qa_contract_invalid", passage_id=passage_id,
                   absolute_start=absolute_start, absolute_end=absolute_end)
            continue
        valid.append(item)
        ledger.private_qa_decision(
            returned_index=number, accepted=True, reason="qa_accepted",
            contract_version=QA_CONTRACT_VERSION, passage_id=passage_id,
            absolute_start=absolute_start, absolute_end=absolute_end,
            qa_id=item["qa_id"],
        )
    existing = {(_semantic_identity(item)): item for item in state.qa_candidates}
    for item in valid:
        existing.setdefault(_semantic_identity(item), item)
    state.qa_candidates = sorted(existing.values(), key=lambda item: item["qa_id"])
    ledger.event("tool_state", tool="generate_qa", valid_count=len(valid), retained_count=len(state.qa_candidates), rejected_count=len(values)-len(valid))


def assess_coverage(state: CaseState, complete: Completion, ledger: CaseLedger) -> None:
    known = {item["id"] for item in state.facets}
    backed = {facet for item in state.qa_candidates for facet in item["facet_ids"]}
    # With non-empty facets, empty accepted QA cannot support any coverage
    # assertion.  For an empty facet domain retain the closed wire-schema
    # round-trip below: it separately protects its no-gap invariant.
    if known and not backed:
        state.coverage = {"covered_facet_ids": [], "score": 0.0}
        state.gaps = ["no accepted QA"] if known else []
        ledger.event("tool_state", tool="assess_coverage", coverage=0.0, gaps=len(state.gaps), reason="no_accepted_qa")
        _record_process_checkpoint(state, ledger, stage="post_coverage")
        return
    data = _json(complete, ledger, "Return JSON {\"covered_facet_ids\":[\"f1\"],\"gaps\":[\"...\"]}. Assess only the accepted QA pool; a facet is covered only when one accepted QA binds it. " + json.dumps({"facets": state.facets, "qas": [{key: item[key] for key in ("facet_ids", "question", "answer", "passage_id")} for item in state.qa_candidates]}), 500, role="generation", required={"covered_facet_ids", "gaps"}, json_schema=_coverage_schema(backed))
    covered, gaps = data["covered_facet_ids"], data["gaps"]
    if not isinstance(covered, list) or not isinstance(gaps, list) or any(item not in backed for item in covered) or any(not isinstance(item, str) for item in gaps): raise PolicyError("coverage_schema")
    if not known:
        # The empty-domain wire schema permits only []/[].  Repeat that
        # invariant after parsing so synthetic or faulty transports cannot
        # manufacture coverage or an ungrounded gap, and avoid 0 / 0.
        if covered or gaps:
            raise PolicyError("coverage_schema")
        state.coverage = {"covered_facet_ids": [], "score": 0.0}
        state.gaps = []
        ledger.event("tool_state", tool="assess_coverage", coverage=0.0, gaps=0)
        _record_process_checkpoint(state, ledger, stage="post_coverage")
        return
    state.coverage = {"covered_facet_ids": sorted(set(covered)), "score": len(set(covered))/len(known)}; state.gaps = [item.strip() for item in gaps if item.strip()]; ledger.event("tool_state", tool="assess_coverage", coverage=state.coverage["score"], gaps=len(state.gaps))
    _record_process_checkpoint(state, ledger, stage="post_coverage")


def _choose_evidence(state: CaseState, *, limit: int) -> list[dict[str, Any]]:
    """Return the deterministic evidence board without mutating controller state."""
    chosen: list[dict[str, Any]] = []; covered: set[str] = set(); remaining: list[dict[str, Any]] = []
    identities: set[tuple[str, int, int, str]] = set()
    for item in state.qa_candidates:
        identity = _semantic_identity(item)
        if identity not in identities:
            identities.add(identity); remaining.append(item)
    while remaining and len(chosen) < limit:
        remaining.sort(key=lambda value: (-len(set(value["facet_ids"]) - covered), -float(value["relevance_score"]), value["qa_id"]))
        item = remaining.pop(0)
        if _semantic_identity(item) in {_semantic_identity(entry) for entry in chosen}: continue
        chosen.append(item); covered.update(item["facet_ids"])
    return chosen


def select_evidence(state: CaseState, ledger: CaseLedger, *, limit: int | None = None) -> None:
    """Greedily select marginal facet coverage; same source may answer two facts."""
    limit = _limit(ledger, "max_qas", 10) if limit is None else limit
    chosen = _choose_evidence(state, limit=limit)
    state.selected_evidence = chosen; state.readiness = readiness_for(chosen); ledger.event("tool_state", tool="select_evidence", readiness=state.readiness, evidence_count=len(chosen))


def _record_process_checkpoint(state: CaseState, ledger: CaseLedger, *, stage: str,
                               evidence: list[dict[str, Any]] | None = None) -> None:
    """Capture a gold-free, provenance-complete trajectory state.

    The snapshot is deliberately invisible to the controller and contains no
    claim, gap text, prompts, model completions, gold annotations, or verdict.
    """
    board = _choose_evidence(state, limit=_limit(ledger, "max_qas", 10)) if evidence is None else evidence
    usage = ledger.usage()
    checkpoint = {
        "schema": PROCESS_CHECKPOINT_SCHEMA,
        "checkpoint_index": len(ledger.process_checkpoints) + 1,
        "stage": stage,
        "package_sha256": source_hash(json.dumps(board, sort_keys=True, separators=(",", ":"), allow_nan=False)),
        "readiness": readiness_for(board),
        "search_attempts": state.search_attempts,
        "qa_attempts": state.qa_attempts,
        "model_assessed_coverage": state.coverage.get("score") if state.coverage else None,
        "gap_count": len(state.gaps) if state.coverage else None,
        "evidence": board,
        "usage": {key: usage[key] for key in (
            "model_calls", "controller_calls", "worker_calls", "search_calls",
            "input_tokens", "output_tokens", "reserved_input_tokens", "reserved_output_tokens",
            "measured_input_tokens", "measured_output_tokens", "measured_calls",
            "token_accounting", "wall_clock_seconds",
        )},
    }
    ledger.process_checkpoint(checkpoint)


def _semantic_identity(item: dict[str, Any]) -> tuple[str, int, int, str]:
    """Same source fact is one candidate even if an upstream call changes qa_id."""
    return (item["passage_id"], item["span_start"], item["span_end"], item["answer"])


def predict_verdict(state: CaseState, verdict_complete: Completion | None, ledger: CaseLedger) -> dict[str, Any]:
    if state.readiness not in {READY_WITH_EVIDENCE, READY_INSUFFICIENT_EVIDENCE}: raise PolicyError("invalid_artifact")
    if verdict_complete is None: raise PolicyError("hero_verdict_completion_required")
    qas = "\n".join(f"Q{number}: {item['question']}\nA{number}: {item['answer']}" for number, item in enumerate(state.selected_evidence, 1))
    raw = _call(verdict_complete, ledger, "You are the AVeriTeC HerO verdict component. Use only the claim and numbered Q/A below. Explain briefly, then terminate with exactly one line in this form: Verdict: Supported | Refuted | Not Enough Evidence | Conflicting Evidence/Cherrypicking.\nClaim: " + state.claim + "\n" + qas, 500, role="verdict")
    label, justification = parse_hero_terminal_output(raw)
    # The parsed terminal label is public semantics; free-form explanation and
    # the full completion are private raw output.  The digest supports local
    # trace integrity checks without exposing either one in public artifacts.
    del justification
    return {"pred_label": label, "raw_output_sha256": source_hash(raw), "evidence": state.selected_evidence, "readiness": state.readiness}


def _finish(state: CaseState, verdict_complete: Completion | None, ledger: CaseLedger, *, reason: str = "forward_pipeline_complete") -> dict[str, Any]:
    select_evidence(state, ledger)
    _record_process_checkpoint(state, ledger, stage="final_evidence", evidence=state.selected_evidence)
    state.terminal_reason = _terminal_reason(state, reason)
    ledger.event(
        "finalization", terminal_reason=state.terminal_reason,
        acquisition_status=acquisition_status(state), readiness=state.readiness,
        evidence_count=len(state.selected_evidence),
    )
    prediction = predict_verdict(state, verdict_complete, ledger)
    prediction["acquisition_status"] = acquisition_status(state)
    prediction["terminal_reason"] = state.terminal_reason
    return prediction


def _fixed(state: CaseState, complete: Completion, _controller_complete: Completion | None, verdict: Completion | None, tools: ClosedWorldTools, ledger: CaseLedger) -> dict[str, Any]:
    decompose_claim(state, complete, ledger); formulate_queries(state, complete, ledger, required_hyde=_limit(ledger, "fixed_hyde_expansions", 4)); search_sparse_dense(state, tools); generate_qa(state, complete, ledger); return _finish(state, verdict, ledger)


def acquisition_status_for(state: CaseState) -> str:
    """Describe observed acquisition progress without claiming evidence quality."""
    if state.search_attempts == 0:
        return "not_attempted"
    if not state.candidate_passages:
        return "searched_no_candidates"
    if state.qa_attempts == 0:
        return "candidates_unassessed"
    if not state.qa_candidates:
        return "qa_attempted_no_valid_items"
    return "valid_items_available"


# Short internal spelling retained for policy call sites; the public helper
# name makes the derivation explicit to regression tests and audit utilities.
acquisition_status = acquisition_status_for


def _terminal_reason(state: CaseState, requested: str) -> str:
    if acquisition_status(state) == "not_attempted":
        return "unresearched_abstention"
    return requested


def _controller_prompt(observation: dict[str, Any]) -> str:
    """Serialize the validated metrics-only observation and nothing case-textual."""
    return (
        "You are a metrics-only workflow controller. Choose exactly one closed-enum "
        "value from route.allowed_actions. Return JSON {\"action\":\"...\"} only. "
        "Do not infer a verdict. Observation JSON: " + canonical_observation(observation)
    )


def _controller(complete: Completion, ledger: CaseLedger, state: CaseState, allowed: set[str], *, stage: str, optional: bool = False) -> str | None:
    observation = controller_observation(state, ledger, stage=stage, allowed_actions=allowed)
    prompt = _controller_prompt(observation)
    digest = observation_sha256(observation)
    ledger.event(
        "controller_observation", schema=CONTROLLER_OBSERVATION_SCHEMA,
        observation_sha256=digest, stage=stage, allowed=sorted(allowed),
    )
    if optional and not ledger.can_reserve_model(prompt, 160, role="controller"):
        ledger.event("optional_action_skipped", action="controller", stage=stage, reason="terminal_verdict_reserved")
        return None
    action = _json(complete, ledger, prompt, 160, role="controller", required={"action"}, json_schema=_controller_schema(allowed))["action"]
    if action not in allowed: raise PolicyError("controller_action")
    prior = state.controller_action_counts.get(action, 0)
    state.controller_action_counts[action] = prior + 1
    if prior > 0 or action.startswith("repeat_"):
        state.controller_repeat_count += 1
    ledger.event(
        "controller_decision", stage=stage, action=action, allowed=sorted(allowed),
        observation_schema=CONTROLLER_OBSERVATION_SCHEMA,
        observation_sha256=digest,
    )
    return action


def _optional_action(callback: Callable[[], None], ledger: CaseLedger, *, action: str) -> bool:
    """Do not let an adaptive repeat consume the terminal verdict allocation."""
    try:
        callback()
    except BudgetExceeded:
        ledger.event("optional_action_skipped", action=action, reason="terminal_verdict_reserved")
        return False
    return True


def _state_signature(state: CaseState) -> tuple[Any, ...]:
    """Stable, typed operational state used to detect actual controller progress."""
    return (
        tuple((item["id"], item["text"]) for item in state.facets),
        tuple(state.query_plan),
        tuple(state.query_history),
        tuple(str(item.passage_id) for item in state.candidate_passages),
        tuple((item["qa_id"], tuple(item["facet_ids"]), item["passage_id"],
               item["span_start"], item["span_end"]) for item in state.qa_candidates),
        json.dumps(state.coverage, sort_keys=True, separators=(",", ":")),
        tuple(state.gaps),
    )


def _record_action_outcome(state: CaseState, *, action: str, progress: bool) -> None:
    """Update executor-owned progress signals for the next LFM checkpoint."""
    if progress:
        state.stagnation = 0
        state.no_progress_actions.clear()
        state.last_action_result = "progress"
    else:
        state.stagnation += 1
        state.no_progress_actions.add(action)
        state.last_action_result = "no_progress"


def _loop_actions(state: CaseState) -> set[str]:
    """Offer only actions whose typed input state exists, without auto-acting."""
    if not state.facets:
        return {"decompose"}
    if not state.query_plan:
        return {"queries"}
    # Once facets and a query plan exist, finalization is structurally valid
    # even when the controller elects not to search.  Keeping finish available
    # here preserves the free loop treatment rather than turning it into the
    # graph's mandatory forward pipeline.
    actions = {"queries", "search", "finish"}
    if state.candidate_passages:
        actions.add("qa")
    if state.qa_candidates:
        actions.add("coverage")
    if state.coverage:
        actions.add("finish")
    # A no-op remains visible, but the same action is not offered again until
    # another typed state change occurs.  This retains alternate free-loop
    # choices (notably search and finish) without manufacturing progress.
    return actions - state.no_progress_actions


def _specialist_actions(role: str, state: CaseState) -> set[str]:
    """Return the state-valid action vocabulary for one registered specialist."""
    if role == "researcher":
        actions = {"queries", "finish"}
        if state.query_plan:
            actions.add("search")
        if state.query_plan and state.qa_candidates and state.coverage:
            actions.add("refine_search")
        return actions
    if role == "qa_specialist" and state.candidate_passages:
        return {"qa", "finish"}
    if role == "coverage_specialist" and state.qa_candidates:
        return {"coverage", "finish"}
    raise PolicyError("specialist_state_invalid")


def _orchestrator_roles(state: CaseState, suppressed: set[str]) -> set[str]:
    """Expose only specialists with valid inputs and unsuppressed same-state work."""
    roles: set[str] = set()
    if state.query_plan:
        roles.add("researcher")
    if state.candidate_passages:
        roles.add("qa_specialist")
    if state.qa_candidates:
        roles.add("coverage_specialist")
    return roles - suppressed


def _loop(state: CaseState, complete: Completion, controller: Completion | None, verdict: Completion | None, tools: ClosedWorldTools, ledger: CaseLedger) -> dict[str, Any]:
    if controller is None:
        raise PolicyError("lfm_controller_completion_required")
    for _ in range(_limit(ledger, "max_loop_steps", 8)):
        action = _controller(controller, ledger, state, _loop_actions(state), stage="loop_control", optional=True)
        if action is None:
            return _finish(state, verdict, ledger, reason="terminal_reservation")
        before = _state_signature(state)
        if action == "decompose": completed = _optional_action(lambda: decompose_claim(state, complete, ledger), ledger, action=action)
        elif action == "queries": completed = _optional_action(lambda: formulate_queries(state, complete, ledger), ledger, action=action)
        elif action == "search": completed = _optional_action(lambda: search_sparse_dense(state, tools), ledger, action=action)
        elif action == "qa": completed = _optional_action(lambda: generate_qa(state, complete, ledger), ledger, action=action)
        elif action == "coverage": completed = _optional_action(lambda: assess_coverage(state, complete, ledger), ledger, action=action)
        else:
            return _finish(state, verdict, ledger, reason="loop_global_finish")
        if not completed:
            return _finish(state, verdict, ledger, reason="terminal_reservation")
        progress = _state_signature(state) != before
        _record_action_outcome(state, action=action, progress=progress)
        if not progress:
            ledger.event("controller_no_progress", action=action, acquisition_status=acquisition_status(state))
        if state.stagnation >= 2:
            raise PolicyError("no_progress")
    return _finish(state, verdict, ledger, reason="loop_step_cap")


def _orchestrator(state: CaseState, complete: Completion, controller: Completion | None, verdict: Completion | None, tools: ClosedWorldTools, ledger: CaseLedger) -> dict[str, Any]:
    if controller is None:
        raise PolicyError("lfm_controller_completion_required")
    decompose_claim(state, complete, ledger); formulate_queries(state, complete, ledger)
    suppressed_by_state: dict[tuple[Any, ...], set[str]] = {}
    role_calls: dict[str, int] = {}
    maximum_role_calls = _limit(ledger, "max_specialist_repeats", 2)
    for _ in range(_limit(ledger, "max_delegation_steps", 6)):
        before = _state_signature(state)
        roles = _orchestrator_roles(state, suppressed_by_state.get(before, set()))
        roles = {role for role in roles if role_calls.get(role, 0) < maximum_role_calls}
        if not roles:
            break
        # Bootstrap already establishes the minimum structurally valid state.
        # Preserve the orchestrator's treatment-level freedom to finalize
        # without forcing a delegation; only impossible specialist roles are
        # masked.
        roles.add("finish")
        role = _controller(controller, ledger, state, roles, stage="delegation_control", optional=True)
        if role is None:
            return _finish(state, verdict, ledger, reason="terminal_reservation")
        if role == "finish":
            return _finish(state, verdict, ledger, reason="orchestrator_global_finish")
        role_calls[role] = role_calls.get(role, 0) + 1
        ledger.delegation(role=role, repeat=role_calls[role] > 1)
        action = _controller(controller, ledger, state, _specialist_actions(role, state), stage=role + "_control", optional=True)
        if action is None:
            return _finish(state, verdict, ledger, reason="terminal_reservation")
        if action == "finish":
            suppressed_by_state.setdefault(before, set()).add(role)
            state.stagnation += 1
            state.last_action_result = "local_finish"
            ledger.event("orchestrator_role_suppressed", role=role, reason="no_typed_state_delta")
            continue
        if action == "queries": completed = _optional_action(lambda: formulate_queries(state, complete, ledger), ledger, action=action)
        elif action == "search": completed = _optional_action(lambda: search_sparse_dense(state, tools), ledger, action=action)
        elif action == "refine_search": completed = _optional_action(
            lambda: (formulate_queries(state, complete, ledger), search_sparse_dense(state, tools)),
            ledger, action=action,
        )
        elif action == "qa": completed = _optional_action(lambda: generate_qa(state, complete, ledger), ledger, action=action)
        else: completed = _optional_action(lambda: assess_coverage(state, complete, ledger), ledger, action=action)
        if not completed:
            return _finish(state, verdict, ledger, reason="terminal_reservation")
        progress = _state_signature(state) != before
        _record_action_outcome(state, action=action, progress=progress)
        if not progress:
            suppressed_by_state.setdefault(before, set()).add(role)
            ledger.event("orchestrator_role_suppressed", role=role, reason="no_typed_state_delta")
    if not state.coverage and not _optional_action(lambda: assess_coverage(state, complete, ledger), ledger, action="coverage"):
        return _finish(state, verdict, ledger, reason="terminal_reservation")
    return _finish(state, verdict, ledger, reason="orchestrator_roles_exhausted")


def _graph(state: CaseState, complete: Completion, controller: Completion | None, verdict: Completion | None, tools: ClosedWorldTools, ledger: CaseLedger) -> dict[str, Any]:
    if controller is None:
        raise PolicyError("lfm_controller_completion_required")
    node = "decompose"
    for _ in range(_limit(ledger, "max_graph_steps", 12)):
        before = _state_signature(state)
        if node == "decompose":
            decompose_claim(state, complete, ledger); action = "advance"
        elif node == "search":
            if not ledger.reserve_finalization(): node = "package"; continue
            if not _optional_action(lambda: formulate_queries(state, complete, ledger), ledger, action="queries"):
                return _finish(state, verdict, ledger, reason="terminal_reservation")
            if not _optional_action(lambda: search_sparse_dense(state, tools), ledger, action="search"):
                return _finish(state, verdict, ledger, reason="terminal_reservation")
            action = "advance"
        elif node == "qa":
            if not ledger.reserve_finalization(): node = "package"; continue
            if not _optional_action(lambda: generate_qa(state, complete, ledger), ledger, action="qa"):
                return _finish(state, verdict, ledger, reason="terminal_reservation")
            qa_repeat_remaining = state.graph_repeats.get("qa", 0) < _limit(
                ledger, "max_graph_local_repeats", 1,
            )
            allowed = {"repeat_qa", "advance"} if qa_repeat_remaining else {"advance"}
            action = _controller(controller, ledger, state, allowed, stage="graph_qa_control", optional=True)
            if action is None: return _finish(state, verdict, ledger, reason="terminal_reservation")
        elif node == "package":
            if not _optional_action(lambda: assess_coverage(state, complete, ledger), ledger, action="coverage"):
                return _finish(state, verdict, ledger, reason="terminal_reservation")
            select_evidence(state, ledger)
            feedback_remaining = state.graph_repeats.get("evidence_feedback", 0) < _limit(
                ledger, "max_graph_local_repeats", 1,
            )
            allowed = {"refine_search", "advance"} if feedback_remaining else {"advance"}
            action = _controller(controller, ledger, state, allowed, stage="graph_package_control", optional=True)
            if action is None: return _finish(state, verdict, ledger, reason="terminal_reservation")
        else: return _finish(state, verdict, ledger, reason="graph_evidence_feedback_complete")
        after = _state_signature(state)
        progress = before != after
        next_node = graph_transition(node, action)
        if next_node == node:
            state.graph_repeats[node] = state.graph_repeats.get(node, 0)+1
            if state.graph_repeats[node] > _limit(ledger, "max_graph_local_repeats", 1) or not progress: next_node = graph_transition(node, "advance")
        elif action == "refine_search":
            state.graph_repeats["evidence_feedback"] = state.graph_repeats.get("evidence_feedback", 0) + 1
            if state.graph_repeats["evidence_feedback"] > _limit(ledger, "max_graph_local_repeats", 1):
                next_node = graph_transition(node, "advance")
        _record_action_outcome(state, action=action, progress=progress)
        ledger.event("graph_transition", node=node, action=action, next_node=next_node, progress=progress); node = next_node
    raise PolicyError("graph_step_cap")


def _graph_no_loops(state: CaseState, complete: Completion, _controller: Completion | None,
                    verdict: Completion | None, tools: ClosedWorldTools,
                    ledger: CaseLedger) -> dict[str, Any]:
    """Execute the graph's forward path once, with every back edge disabled."""
    decompose_claim(state, complete, ledger)
    formulate_queries(state, complete, ledger)
    search_sparse_dense(state, tools)
    generate_qa(state, complete, ledger)
    assess_coverage(state, complete, ledger)
    ledger.event("graph_no_loops", path=["decompose", "search", "qa", "package", "verdict"])
    return _finish(state, verdict, ledger, reason="graph_forward_path_complete")


def _fixed_repeats(state: CaseState, complete: Completion, _controller: Completion | None,
                   verdict: Completion | None, tools: ClosedWorldTools,
                   ledger: CaseLedger) -> dict[str, Any]:
    """Run a predeclared number of identical query/search/QA/coverage cycles."""
    cycles = _limit(ledger, "fixed_repeat_cycles", 2)
    decompose_claim(state, complete, ledger)
    for cycle in range(1, cycles + 1):
        if cycle > 1:
            if not ledger.reserve_finalization():
                return _finish(state, verdict, ledger, reason="terminal_reservation")
            ledger.repeat(action="fixed_query_search_qa_coverage_cycle", number=cycle)
        actions = (
            ("queries", lambda: formulate_queries(
                state, complete, ledger,
                required_hyde=_limit(ledger, "fixed_hyde_expansions", 4),
            )),
            ("search", lambda: search_sparse_dense(state, tools)),
            ("qa", lambda: generate_qa(state, complete, ledger)),
            ("coverage", lambda: assess_coverage(state, complete, ledger)),
        )
        for action, callback in actions:
            if cycle == 1:
                callback()
            elif not _optional_action(callback, ledger, action=action):
                return _finish(state, verdict, ledger, reason="terminal_reservation")
        ledger.event("fixed_repeat_cycle", cycle=cycle, total_cycles=cycles)
    return _finish(state, verdict, ledger, reason="fixed_repeat_schedule_complete")


def fixed_evidence_policy(**kwargs: Any) -> list[dict[str, Any]]:
    state = CaseState(case_id=kwargs.get("case_id", "stage-case"), claim=kwargs["claim"], condition="fixed_flow"); ledger = kwargs["ledger"]; tools = _tools(kwargs["corpus"], kwargs["blocked_url_families"], ledger, kwargs.get("embed"))
    decompose_claim(state, kwargs["complete"], ledger); formulate_queries(state, kwargs["complete"], ledger, required_hyde=_limit(ledger, "fixed_hyde_expansions", 4)); search_sparse_dense(state, tools); generate_qa(state, kwargs["complete"], ledger); select_evidence(state, ledger)
    _record_process_checkpoint(state, ledger, stage="final_evidence", evidence=state.selected_evidence)
    if state.readiness == INVALID_ARTIFACT: raise PolicyError("invalid_artifact")
    return state.selected_evidence


def fixed_verdict_policy(*, claim: str, evidence: list[dict[str, Any]], verdict_complete: Completion | None, ledger: CaseLedger, **_ignore: Any) -> dict[str, Any]:
    state = CaseState(case_id="stage-case", claim=claim, condition="fixed_flow", selected_evidence=evidence); state.readiness = readiness_for(evidence); return predict_verdict(state, verdict_complete, ledger)


def infer_condition(condition: str, **kwargs: Any) -> dict[str, Any]:
    if condition not in CONDITIONS: raise ValueError("condition")
    ledger: CaseLedger = kwargs["ledger"]; state = CaseState(case_id=kwargs.get("case_id", "in-memory-case"), claim=kwargs["claim"], condition=condition); tools = _tools(kwargs["corpus"], kwargs["blocked_url_families"], ledger, kwargs.get("embed"))
    policies = {
        "fixed_flow": _fixed,
        "loop_agent": _loop,
        "orchestrator_agent": _orchestrator,
        "graph_loop_agent": _graph,
        "graph_no_loops": _graph_no_loops,
        "fixed_repeats": _fixed_repeats,
    }
    try:
        return {"status": "succeeded", "prediction": policies[condition](state, kwargs["complete"], kwargs.get("controller_complete"), kwargs.get("verdict_complete"), tools, ledger)}
    except BudgetExceeded as error:
        return {"status": "budget_exhausted", "prediction": {"pred_label": "INVALID", "evidence": [], "readiness": INVALID_ARTIFACT}, "error": str(error)}
    except (PolicyError, ContractError, ValueError) as error:
        return {"status": "failed", "prediction": {"pred_label": "INVALID", "evidence": [], "readiness": INVALID_ARTIFACT}, "error": str(error)}
