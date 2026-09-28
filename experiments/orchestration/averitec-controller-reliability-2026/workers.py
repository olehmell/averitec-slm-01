"""Private shared-text worker adapter for controller-reliability traces.

The controller sees only the return value of :meth:`run`.  Case text, source
text, identifiers, evidence, model completions, and ledger events stay in this
object and must be persisted, if needed, only by a private trace owner.
"""

from __future__ import annotations

import re
import sys
import hashlib
import json
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


_SIBLING = Path(__file__).resolve().parent / "support"
if str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))

from averitec_fixed import FORBIDDEN_RUNTIME_FIELDS, validate_runtime_claim  # noqa: E402
from fixed_inference import archive_passages  # noqa: E402
from fixed_runner import exclusions as load_exclusions, invoke_openai  # noqa: E402
from inference.contracts import CaseState, parse_hero_terminal_output  # noqa: E402
from inference.ledger import CaseLedger  # noqa: E402
from inference.policy import (  # noqa: E402
    PolicyError,
    _call,
    assess_coverage,
    decompose_claim,
    formulate_queries,
    generate_qa,
    select_evidence,
)
from inference.tools import ClosedWorldTools, LazyDenseEmbedder  # noqa: E402
from source_corpora import SourceCorpora  # noqa: E402


STAGES = frozenset(("decompose", "queries", "retrieve", "qa", "coverage", "select", "verdict"))
_CASE_FIELDS = frozenset(("case_id", "claim", "split"))
_CASE_ID = re.compile(r"averitec-(train|dev|test)-(\d{4})$")
_MODEL_ID = re.compile(r"[A-Za-z0-9._/-]{1,160}$")
_TRACE_CANDIDATE_LIMIT = 10
_ERROR_CODES = frozenset((
    "invalid_action", "invalid_case", "source_unavailable", "transport_failure",
    "worker_failure", "terminal_already_called", "terminal_not_ready",
    "terminal_retry_exhausted",
))
_CAPS = {
    "max_model_calls_per_case": 16,
    "max_input_tokens_per_case": 32_768,
    "max_output_tokens_per_case": 12_288,
    "max_search_calls_per_case": 12,
    "max_repairs_per_case": 0,
    "wall_clock_seconds_per_case": 900,
    "max_facets": 4,
    "max_hyde_expansions": 8,
    "max_qas": 10,
    "sparse_limit": 10_000,
    "finalization_input_token_reserve": 4096,
    "finalization_output_token_reserve": 500,
}


def _validate_case(case: dict[str, Any]) -> dict[str, str]:
    """Accept only runtime fields; gold cannot enter this adapter."""
    if not isinstance(case, dict) or set(case) != _CASE_FIELDS:
        raise ValueError("invalid_case")
    if FORBIDDEN_RUNTIME_FIELDS.intersection(case):
        raise ValueError("invalid_case")
    validate_runtime_claim(case)
    if not isinstance(case["split"], str) or _CASE_ID.fullmatch(case["case_id"]) is None:
        raise ValueError("invalid_case")
    split, _ = _CASE_ID.fullmatch(case["case_id"]).groups()  # type: ignore[union-attr]
    if split != case["split"]:
        raise ValueError("invalid_case")
    return {key: case[key] for key in _CASE_FIELDS}


def _models_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    suffix = "/v1/chat/completions"
    if value.endswith(suffix):
        return value[: -len("chat/completions")] + "models"
    return value + "/v1/models"


def _fetch_model_ids(endpoint: str, timeout_seconds: int) -> tuple[str, ...]:
    """Read only the standard OpenAI-compatible model inventory."""
    with urlopen(Request(_models_url(endpoint), method="GET"), timeout=timeout_seconds) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("model_endpoint_identity")
    values = [item.get("id") for item in rows if isinstance(item, dict)]
    if any(not isinstance(value, str) or _MODEL_ID.fullmatch(value) is None for value in values):
        raise ValueError("model_endpoint_identity")
    return tuple(sorted(set(values)))


class _BaseTools:
    """Private state and the sole metrics-only controller projection."""

    def _metrics(self) -> dict[str, int]:
        return {
            "facet_count": len(self._state.facets),
            "query_count": len(self._state.query_plan),
            "candidate_count": len(self._state.candidate_passages),
            "qa_count": len(self._state.qa_candidates),
            "selected_count": len(self._state.selected_evidence),
        }

    def _result(self, status: str, error_code: str | None = None) -> dict[str, Any]:
        if status not in {"ok", "empty", "timeout", "invalid"}:
            raise ValueError("worker_result_status")
        if error_code is not None and error_code not in _ERROR_CODES:
            error_code = "worker_failure"
        return {"status": status, "metrics": self._metrics(), "error_code": error_code}

    def trace_snapshot(self) -> dict[str, Any]:
        """Inference-only semantic state for authorized private tracing."""
        def passage(value: Any) -> dict[str, Any]:
            return {
                "text": value.text, "url": value.url, "passage_id": value.passage_id,
                "source_text_sha256": hashlib.sha256(value.source_text.encode("utf-8")).hexdigest(),
                "source_text_length": len(value.source_text), "source_start": value.source_start,
            }
        candidate_hash = hashlib.sha256()
        for item in self._state.candidate_passages:
            candidate_hash.update(str(item.passage_id).encode("utf-8"))
            candidate_hash.update(b"\n")
        return {
            "facets": deepcopy(self._state.facets),
            "query_plan": list(self._state.query_plan),
            "candidate_passages": {
                "total_count": len(self._state.candidate_passages),
                "ordered_ids_sha256": candidate_hash.hexdigest(),
                "qa_window": [passage(item) for item in self._state.candidate_passages[:_TRACE_CANDIDATE_LIMIT]],
            },
            "qa_candidates": deepcopy(self._state.qa_candidates),
            "coverage": deepcopy(self._state.coverage),
            "selected_evidence": deepcopy(self._state.selected_evidence),
            "readiness": self._state.readiness,
        }

    def _span(self, name: str, *, kind: str, input: dict[str, Any], model: str | None = None,
              metadata: dict[str, Any] | None = None) -> Any:
        if self._tracer is None:
            return nullcontext(None)
        kwargs: dict[str, Any] = {"kind": kind, "input": input, "metadata": metadata or {}}
        if model is not None:
            kwargs["model"] = model
        return self._tracer.span(name, **kwargs)

    @staticmethod
    def _usage_from_completion(value: Any) -> dict[str, int | None]:
        def count(name: str) -> int | None:
            item = getattr(value, name, None)
            return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None
        return {"input_tokens": count("input_tokens"), "output_tokens": count("output_tokens")}

    def _capture_completion(self, complete: Any) -> Any:
        def captured(*args: Any, **kwargs: Any) -> Any:
            prompt = args[0] if len(args) > 0 and isinstance(args[0], str) else ""
            maximum = args[1] if len(args) > 1 and isinstance(args[1], int) else None
            schema = kwargs.get("json_schema") if isinstance(kwargs.get("json_schema"), dict) else None
            # Explicitly trace only the contract arguments; never forward a
            # transport object, endpoint, headers, or arbitrary kwargs.
            with self._span(
                "averitec.worker.generation", kind="generation",
                input={"prompt": prompt, "max_tokens": maximum, "json_schema": deepcopy(schema)},
                model=self._model_binding["requested"],
                metadata={"requested_model": self._model_binding["requested"]},
            ) as span:
                try:
                    result = complete(*args, **kwargs)
                except Exception as error:
                    if span is not None:
                        span.update(output={"status": "failed"}, metadata={
                            "failure": "timeout" if isinstance(error, TimeoutError) else "generation_failure",
                        })
                    raise
                content = getattr(result, "content", result)
                if isinstance(content, str):
                    self._output_hashes.append(hashlib.sha256(content.encode("utf-8")).hexdigest())
                if span is not None:
                    span.update(output={"completion": content if isinstance(content, str) else None},
                                usage=self._usage_from_completion(result), metadata={"status": "ok"})
                return result
        for name in ("accepts_timeout", "accepts_json_schema"):
            if getattr(complete, name, False):
                setattr(captured, name, True)
        return captured

    def preparation_receipt(self) -> dict[str, Any]:
        """Return a public-safe binding and accounting receipt, never raw inputs."""
        return {
            "schema": "averitec-controller-worker-preparation/v1",
            "model_binding": dict(self._model_binding),
            "endpoint_health": dict(self._endpoint_health),
            "corpus": dict(self._corpus_receipt),
            "stages": deepcopy(self._stage_log),
            "ledger": self._ledger.usage(),
            "output_content_sha256": list(self._output_hashes),
        }

    def _empty_or_ok(self, action: str) -> dict[str, Any]:
        counts = self._metrics()
        key = {
            "decompose": "facet_count", "queries": "query_count", "retrieve": "candidate_count",
            "qa": "qa_count", "select": "selected_count",
        }.get(action)
        return self._result("empty" if key is not None and counts[key] == 0 else "ok")

    def _run_verdict(self) -> dict[str, Any]:
        if self._terminal_succeeded:
            return self._result("invalid", "terminal_already_called")
        if self._terminal_attempts >= 2:
            return self._result("invalid", "terminal_retry_exhausted")
        if self._state.readiness not in {"ready_with_evidence", "ready_insufficient_evidence"}:
            return self._result("invalid", "terminal_not_ready")
        self._terminal_attempts += 1
        qas = "\n".join(
            f"Q{number}: {item['question']}\nA{number}: {item['answer']}"
            for number, item in enumerate(self._state.selected_evidence, 1)
        )
        # Qwen is the shared text worker here, but the frozen terminal format is
        # useful to detect completion—not to score verdict correctness.  The raw
        # completion, parsed label, retry count, and ledger remain private.
        prompt = (
            "Use only the claim and numbered Q/A evidence. Give a non-empty brief justification, "
            "then terminate with exactly one line: Verdict: Supported | Refuted | Not Enough Evidence | "
            "Conflicting Evidence/Cherrypicking.\nClaim: " + self._case["claim"] + "\n" + qas
        )
        raw = _call(self._complete, self._ledger, prompt, 500, role="verdict")
        parse_hero_terminal_output(raw)
        self._terminal_succeeded = True
        return self._result("ok")

    def run(self, action: str) -> dict[str, Any]:
        before_calls = self._ledger.model_calls
        before_input = self._ledger.input_tokens
        before_output = self._ledger.output_tokens
        before_outputs = len(self._output_hashes)
        with self._span(
            "averitec.worker." + (action if action in STAGES else "invalid_action"), kind="tool",
            input={"action": action, "state": self.trace_snapshot()},
            metadata={"stage": action if action in STAGES else "invalid_action"},
        ) as span:
            result = self._run(action)
            if span is not None:
                span.update(output={"result": deepcopy(result), "state": self.trace_snapshot()}, usage={
                    "input_tokens": self._ledger.input_tokens - before_input,
                    "output_tokens": self._ledger.output_tokens - before_output,
                }, metadata={"status": result["status"], "error_code": result["error_code"]})
        self._stage_log.append({
            "stage": action if action in STAGES else "invalid_action",
            "status": result["status"], "error_code": result["error_code"],
            "model_calls": self._ledger.model_calls - before_calls,
            "output_content_sha256": self._output_hashes[before_outputs:],
        })
        return result

    def _run(self, action: str) -> dict[str, Any]:
        if action not in STAGES:
            return self._result("invalid", "invalid_action")
        try:
            if action == "decompose":
                decompose_claim(self._state, self._complete, self._ledger)
            elif action == "queries":
                formulate_queries(self._state, self._complete, self._ledger)
            elif action == "retrieve":
                if not self._state.query_plan:
                    return self._result("invalid", "worker_failure")
                self._state.search_attempts += 1
                # CPU-only scope: never call hybrid()/dense() or instantiate a model.
                self._state.candidate_passages = self._tools.sparse(" ".join(self._state.query_plan))
            elif action == "qa":
                if not self._state.candidate_passages:
                    return self._result("empty")
                generate_qa(self._state, self._complete, self._ledger)
            elif action == "coverage":
                if not self._state.facets:
                    return self._result("empty")
                assess_coverage(self._state, self._complete, self._ledger)
            elif action == "select":
                select_evidence(self._state, self._ledger)
            else:
                return self._run_verdict()
            return self._empty_or_ok(action)
        except TimeoutError:
            return self._result("timeout", "transport_failure")
        except (PolicyError, ValueError, OSError):
            return self._result("invalid", "worker_failure")
        except Exception:
            # Provider details must never be copied into controller observations.
            return self._result("invalid", "transport_failure")


class PipelineTools(_BaseTools):
    """Run shared Qwen text stages over a frozen CPU-BM25 per-case corpus."""

    def __init__(self, case: dict, *, endpoint: str, model: str, source_corpora: Path,
                 exclusions: Path, root: Path, timeout_seconds: int = 90, tracer: Any | None = None) -> None:
        self._case = _validate_case(case)
        if not isinstance(endpoint, str) or not endpoint or not isinstance(model, str) or not model:
            raise ValueError("invalid_case")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds < 1:
            raise ValueError("invalid_case")
        self._state = CaseState(case_id=self._case["case_id"], claim=self._case["claim"], condition="controller_reliability")
        self._ledger = CaseLedger(dict(_CAPS))
        self._tracer = tracer
        self._stage_log: list[dict[str, Any]] = []
        self._output_hashes: list[str] = []
        self._terminal_attempts = 0
        self._terminal_succeeded = False
        archive = SourceCorpora(Path(source_corpora), repository_root=Path(root))
        match = _CASE_ID.fullmatch(self._case["case_id"])
        assert match is not None
        try:
            # Hash and member checks happen before any model transport exists.
            verified_archives = archive.verify_cases([self._case])
            corpus = archive_passages(archive, split=match.group(1), index=int(match.group(2)), blocked=load_exclusions(Path(exclusions)))
        except (OSError, ValueError):
            # Construction is a local input validation boundary; callers must not
            # accidentally convert a missing corpus into a model request.
            raise ValueError("source_unavailable") from None
        self._tools = ClosedWorldTools(corpus, set(), self._ledger, LazyDenseEmbedder())
        try:
            reported = _fetch_model_ids(endpoint, timeout_seconds)
        except Exception:
            raise ValueError("model_endpoint_identity") from None
        if model not in reported:
            raise ValueError("model_endpoint_identity")
        self._model_binding = {"requested": model, "reported": model}
        self._endpoint_health = {"checked": True, "identity_verified": True}
        archive_hashes = sorted(verified_archives.values())
        self._corpus_receipt = {
            "manifest_sha256": archive.manifest_sha256,
            "selected_archive_sha256": archive_hashes[0] if len(archive_hashes) == 1 else hashlib.sha256(
                json.dumps(archive_hashes).encode("utf-8")
            ).hexdigest(),
            "member_verified": True,
        }
        transport = invoke_openai(endpoint, model, qwen_text_only=True)

        def request(prompt: str, maximum: int, _remaining: int, *, json_schema: dict[str, Any] | None = None) -> Any:
            return transport(prompt, maximum, timeout_seconds, json_schema=json_schema)

        request.accepts_timeout = True  # type: ignore[attr-defined]
        request.accepts_json_schema = True  # type: ignore[attr-defined]
        self._complete = self._capture_completion(request)


class SyntheticTools(_BaseTools):
    """Deterministic local smoke fixture; it makes no dataset or quality claim."""

    def __init__(self, case: dict, *, tracer: Any | None = None) -> None:
        self._case = _validate_case(case)
        self._ledger = CaseLedger(dict(_CAPS))
        self._tracer = tracer
        self._stage_log: list[dict[str, Any]] = []
        self._output_hashes: list[str] = []
        self._state = CaseState(case_id=self._case["case_id"], claim=self._case["claim"], condition="controller_reliability_synthetic")
        self._terminal_attempts = 0
        self._terminal_succeeded = False
        self._model_binding = {"requested": "synthetic", "reported": "synthetic"}
        self._endpoint_health = {"checked": False, "identity_verified": False}
        self._corpus_receipt = {"manifest_sha256": None, "selected_archive_sha256": None, "member_verified": False}
        self._complete = self._capture_completion(self._synthetic_complete)
        self._tools = None

    @staticmethod
    def _synthetic_complete(prompt: str, _maximum: int, *_args: Any, **_kwargs: Any) -> str:
        if "Decompose factual facets" in prompt:
            return '{"facets":[{"id":"f1","text":"synthetic fact"}]}'
        if "retrieval hypotheses" in prompt:
            return '{"hyde":["synthetic retrieval hypothesis"]}'
        if prompt.startswith('Return JSON {"covered_facet_ids"'):
            return '{"covered_facet_ids":[],"gaps":["synthetic no evidence"]}'
        return "Synthetic evidence is insufficient.\nVerdict: Not Enough Evidence"

    def _run(self, action: str) -> dict[str, Any]:
        if action == "retrieve":
            if not self._state.query_plan:
                return self._result("invalid", "worker_failure")
            self._state.search_attempts += 1
            self._state.candidate_passages = []
            return self._result("empty")
        return super()._run(action)
