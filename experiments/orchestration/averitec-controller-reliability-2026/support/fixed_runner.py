"""Shared v4 execution with exact bindings, staged gates and resumable accounting."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Callable, Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from averitec_fixed import url_family, validate_runtime_claim
from fixed_inference import archive_passages
from inference.checkpoint import (append, begin_pair, bind_run, completed, finish_pair,
                                  inflight_pairs, output_records, run_lock)
from inference.contracts import (ADAPTIVE_CONDITIONS, CONDITIONS, INVALID_ARTIFACT, READY_INSUFFICIENT_EVIDENCE,
                                 READY_WITH_EVIDENCE, validate_package)
from inference.ledger import BudgetExceeded, CaseLedger, PrivateTrace
from inference.policy import Completion, CompletionResult, fixed_evidence_policy, fixed_verdict_policy, infer_condition
from inference.tools import LazyDenseEmbedder
from source_corpora import SourceCorpora

EXPERIMENT_DIR = Path(__file__).resolve().parent
STAGE_SCHEMA = "averitec-evidence-stage/v3-process-checkpoints"
DEFAULT_CAPS = {"max_model_calls_per_case": 16, "max_input_tokens_per_case": 32768,
    "max_output_tokens_per_case": 12288, "max_search_calls_per_case": 12,
    "max_repairs_per_case": 2, "wall_clock_seconds_per_case": 900,
    "max_graph_steps": 12, "max_graph_local_repeats": 1}
HTTP_ERROR_BODY_LIMIT = 16 * 1024
PUBLIC_HTTP_DIAGNOSTIC_LIMIT = 256
_SCHEMA_KEYWORDS = ("uniqueItems", "additionalProperties", "required", "enum", "oneOf", "anyOf", "allOf", "$ref")
_PRODUCTION_SCHEMA_TITLES = frozenset(("facets_response", "hyde_response", "qa_response", "coverage_response", "controller_response"))
_SAFE_HTTP_DIAGNOSTIC = re.compile(
    r"http_error:(?:[1-5][0-9]{2}|unknown):reason_(?:schema_rejected|backend_rejected|request_rejected|unavailable)"
    r"(?::keyword_(?:uniqueItems|additionalProperties|required|enum|oneOf|anyOf|allOf|ref))?"
    r":schema_(?:none|[A-Za-z0-9_-]{1,64}):schema_sha256_(?:none|[0-9a-f]{64})"
    r":body_(?:json|malformed|truncated|read_failed)"
)


class HTTPFailureDiagnostic(Exception):
    """A public-safe HTTP failure description with no provider response data."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _schema_identity(json_schema: dict[str, Any] | None) -> tuple[str, str]:
    """Return a safe, deterministic identity for a request schema.

    Schema titles are local policy metadata but remain allowlisted here because
    this value becomes part of an append-only public result.  The schema body
    itself is represented only by its digest.
    """
    if json_schema is None:
        return "none", "none"
    title = json_schema.get("title")
    name = title if isinstance(title, str) and title in _PRODUCTION_SCHEMA_TITLES else "policy_response"
    try:
        return name, _digest(json_schema)
    except (TypeError, ValueError):
        return name, "none"


def _http_status(error: HTTPError) -> str:
    status = error.code
    return str(status) if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599 else "unknown"


def _http_error_body_detail(error: HTTPError) -> tuple[str, str | None, str]:
    """Classify at most 16 KiB of a JSON error body without retaining it.

    Any incomplete, unreadable, malformed, or non-JSON response is deliberately
    uninformative.  Valid JSON can yield only a fixed reason class and one
    allowlisted schema keyword; arbitrary server strings never cross this
    function's return boundary.
    """
    try:
        body = error.read(HTTP_ERROR_BODY_LIMIT + 1)
    except Exception:
        return "reason_unavailable", None, "body_read_failed"
    if not isinstance(body, bytes):
        return "reason_unavailable", None, "body_read_failed"
    if len(body) > HTTP_ERROR_BODY_LIMIT:
        return "reason_unavailable", None, "body_truncated"
    try:
        decoded = body.decode("utf-8")
        payload = json.loads(decoded, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("invalid_json_constant")))
    except (UnicodeDecodeError, ValueError, TypeError):
        return "reason_unavailable", None, "body_malformed"
    if not isinstance(payload, (dict, list)):
        return "reason_unavailable", None, "body_malformed"

    folded = decoded.casefold()
    schema_context = any(signal in folded for signal in
                         ("json schema", "response_format", "structured output", "guided decoding"))
    if schema_context:
        for keyword in _SCHEMA_KEYWORDS:
            if re.search(r"(?<![a-z0-9_$-])" + re.escape(keyword.casefold()) + r"(?![a-z0-9_$-])", folded):
                return "reason_schema_rejected", keyword.replace("$", ""), "body_json"
        return "reason_schema_rejected", None, "body_json"
    if any(signal in folded for signal in ("xgrammar", "backend", "vllm")):
        return "reason_backend_rejected", None, "body_json"
    return "reason_request_rejected", None, "body_json"


def _http_failure(error: HTTPError, json_schema: dict[str, Any] | None) -> HTTPFailureDiagnostic:
    """Create the sole public representation of a provider HTTP failure."""
    reason, keyword, body_state = _http_error_body_detail(error)
    name, schema_digest = _schema_identity(json_schema)
    fields = ["http_error", _http_status(error), reason]
    if keyword is not None:
        fields.append("keyword_" + keyword)
    fields.extend(("schema_" + name, "schema_sha256_" + schema_digest, body_state))
    message = ":".join(fields)
    if len(message.encode("ascii")) > PUBLIC_HTTP_DIAGNOSTIC_LIMIT or _SAFE_HTTP_DIAGNOSTIC.fullmatch(message) is None:
        message = "http_error:unknown:reason_unavailable:schema_none:schema_sha256_none:body_read_failed"
    return HTTPFailureDiagnostic(message)


def implementation_digest() -> str:
    names = ("averitec_fixed.py", "data_preparation.py", "source_corpora.py", "prepare_study_phase.py", "fixed_inference.py", "fixed_runner.py", "run.py", "config.yaml")
    paths = [EXPERIMENT_DIR / name for name in names]
    paths.extend(sorted((EXPERIMENT_DIR / "inference").glob("*.py")))
    return _digest({str(path.relative_to(EXPERIMENT_DIR)): sha256(path) for path in paths})


def model_bindings() -> dict[str, Any]:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))["models"]


def exclusions(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("blocked_url_families") if isinstance(value, dict) else None
    if not isinstance(rows, list) or any(not isinstance(item, str) for item in rows):
        raise ValueError("leakage_manifest_schema")
    return {url_family(item) for item in rows}


def invoke_openai(endpoint: str, model: str, *, qwen_text_only: bool = False,
                  structured_output: bool = False) -> Completion:
    """No hidden transport retry; every dispatched call has a ledger reservation.

    vLLM 0.17 accepts the OpenAI-compatible ``response_format`` envelope for
    JSON Schema constrained decoding.  It is emitted only for policy calls
    that supplied a schema; HerO's terminal textual verdict is intentionally
    left unconstrained.
    """
    def complete(prompt: str, maximum: int, timeout_seconds: int, *, json_schema: dict[str, Any] | None = None) -> CompletionResult:
        body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": maximum}
        if qwen_text_only:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if json_schema is not None and (qwen_text_only or structured_output):
            title = json_schema.get("title")
            name = title if isinstance(title, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", title) else "policy_response"
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": json_schema}}
        request = Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode())
        except HTTPError as error:
            raise _http_failure(error, json_schema) from None
        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list) or not payload["choices"]:
            raise ValueError("model_completion_schema")
        choice = payload["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("model_completion_schema")
        content, usage = choice["message"].get("content"), payload.get("usage", {})
        if not isinstance(content, str) or not isinstance(usage, dict):
            raise ValueError("model_completion_schema")
        def count(key: str) -> int | None:
            value = usage.get(key)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("model_usage_schema")
            return value
        return CompletionResult(content, count("prompt_tokens"), count("completion_tokens"), choice.get("finish_reason"))
    complete.accepts_timeout = True  # type: ignore[attr-defined]
    complete.accepts_json_schema = True  # type: ignore[attr-defined]
    return complete


def _validate_claims(claims: list[dict[str, str]]) -> None:
    if not isinstance(claims, list) or not claims:
        raise ValueError("runtime_cases_empty")
    ids = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"case_id", "claim", "split"}:
            raise ValueError("runtime_fields")
        validate_runtime_claim(claim)
        if claim["split"] not in {"dev", "train"} or claim["case_id"].split("-")[1] != claim["split"]:
            raise ValueError("runtime_case_split")
        ids.append(claim["case_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("runtime_case_id_duplicate")


def corpus_fingerprints(archive: SourceCorpora, claims: list[dict[str, str]]) -> dict[str, str]:
    splits = {case["split"] for case in claims}
    return {
        "source_corpora_manifest_sha256": archive.manifest_sha256,
        "source_corpora_archives_sha256": _digest(archive.fingerprints(splits=splits)),
    }


def _inputs(claims: list[dict[str, str]], archive: SourceCorpora, blocked: set[str], caps: dict[str, Any], supplied: dict[str, str] | None) -> dict[str, str]:
    _validate_claims(claims)
    config_path = EXPERIMENT_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    actual = {"implementation_sha256": implementation_digest(), "config_sha256": sha256(config_path),
        "model_bindings_sha256": _digest(config["models"]), "protocol_revision": config["protocol_revision"],
        "runtime_claim_content_sha256": _digest(claims), **corpus_fingerprints(archive, claims),
        "blocked_url_families_sha256": _digest(sorted(blocked)), "resource_caps_sha256": _digest(caps),
        "source_window_chars": str(caps.get("source_window_chars", 1200)),
        "condition_contract_sha256": _digest(CONDITIONS), "evidence_schema": STAGE_SCHEMA}
    for key, value in (supplied or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("input_fingerprint_schema")
        if key in actual and actual[key] != value:
            raise ValueError("input_binding_mismatch:" + key)
        actual[key] = value
    return actual


def _error(error: Exception) -> str:
    value = str(error)
    if (len(value.encode("ascii", "ignore")) == len(value)
            and len(value.encode("ascii")) <= PUBLIC_HTTP_DIAGNOSTIC_LIMIT
            and _SAFE_HTTP_DIAGNOSTIC.fullmatch(value) is not None):
        return value
    return value if re.fullmatch(r"[a-zA-Z0-9_:.-]{1,160}", value) else type(error).__name__


def _failure(error: str, *, status: str = "failed") -> dict[str, Any]:
    return {"status": status, "prediction": {"pred_label": "INVALID", "evidence": [], "readiness": INVALID_ARTIFACT}, "error": error}


def _ledger(caps: dict[str, Any], *, private_trace_root: Path | None, condition: str, case_id: str, phase: str) -> CaseLedger:
    """Make an optional private trace explicit per condition/case/stage.

    Public records never reference or embed the trace.  Each new phase uses a
    fresh file, so evidence and verdict execution cannot merge accidentally.
    """
    trace = PrivateTrace(private_trace_root, condition=condition, case_id=case_id, phase=phase) if private_trace_root is not None else None
    return CaseLedger(caps, private_trace=trace)


def _stage_problem(package: dict[str, Any]) -> str:
    """Keep a producer's safe invalid reason when the package is invalid.

    Stage artifacts never contain completion text.  Only the same compact,
    allowlisted error form produced by ``_error`` may cross the stage boundary.
    """
    reason = package.get("invalid_reason")
    if isinstance(reason, str) and _error(ValueError(reason)) == reason:
        return reason
    return "invalid_artifact"


def _corpus(archive: SourceCorpora, claim: dict[str, str], blocked: set[str], caps: dict[str, Any]) -> list[Any]:
    return archive_passages(archive, split=claim["split"], index=int(claim["case_id"].rsplit("-", 1)[1]), blocked=blocked, window_chars=int(caps.get("source_window_chars", 1200)))


def _verify_sources(evidence: list[dict[str, Any]], corpus: list[Any]) -> None:
    sources = {str(item.passage_id): item for item in corpus}
    for item in evidence:
        source = sources.get(item.get("passage_id"))
        if source is None or item.get("url") != source.url or item.get("scraped_text") != source.source_text:
            raise ValueError("evidence_corpus_provenance_mismatch")
        if item.get("span_start", -1) < source.source_start or item.get("span_end", -1) > source.source_start + len(source.text):
            raise ValueError("evidence_outside_retrieval_window")


def _stream_path(output: Path, condition: str) -> Path:
    return output.with_name(output.stem + "." + condition + output.suffix)


def _sync_streams(output: Path, conditions: list[str]) -> None:
    """Reconcile append-only condition views from the authoritative case journal.

    A crash after journal fsync but before a view append is repaired without a
    model call. Existing foreign, duplicated or corrupted view rows fail closed.
    """
    if not output.exists():
        return
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    for condition in conditions:
        path = _stream_path(output, condition)
        expected = [row for row in rows if row["condition"] == condition]
        raw = path.read_bytes() if path.exists() else b""
        if raw and not raw.endswith(b"\n"):
            raise ValueError("condition_stream_partial_line")
        existing = [json.loads(line) for line in raw.decode().splitlines() if line]
        if existing != expected[:len(existing)] or len(existing) > len(expected):
            raise ValueError("condition_stream_mismatch")
        for row in expected[len(existing):]:
            append(path, row)


def run_conditions(*, claims: list[dict[str, str]], archive: SourceCorpora, blocked_url_families: set[str], condition_ids: Iterable[str], complete: Completion, verdict_complete: Completion | None, controller_complete: Completion | None = None, output: Path | None = None, embed: Callable[[list[str]], Any] | None = None, caps: dict[str, Any] | None = None, input_fingerprints: dict[str, str] | None = None, mode: str = "main", order_seed: int = 42, private_trace_root: Path | None = None) -> list[dict[str, Any]]:
    if output is None:
        return _run_conditions_locked(
            claims=claims, archive=archive, blocked_url_families=blocked_url_families,
            condition_ids=condition_ids, complete=complete, verdict_complete=verdict_complete,
            controller_complete=controller_complete, output=output, embed=embed, caps=caps,
            input_fingerprints=input_fingerprints, mode=mode, order_seed=order_seed,
            private_trace_root=private_trace_root,
        )
    with run_lock(output):
        return _run_conditions_locked(
            claims=claims, archive=archive, blocked_url_families=blocked_url_families,
            condition_ids=condition_ids, complete=complete, verdict_complete=verdict_complete,
            controller_complete=controller_complete, output=output, embed=embed, caps=caps,
            input_fingerprints=input_fingerprints, mode=mode, order_seed=order_seed,
            private_trace_root=private_trace_root,
        )


def _run_conditions_locked(*, claims: list[dict[str, str]], archive: SourceCorpora, blocked_url_families: set[str], condition_ids: Iterable[str], complete: Completion, verdict_complete: Completion | None, controller_complete: Completion | None = None, output: Path | None = None, embed: Callable[[list[str]], Any] | None = None, caps: dict[str, Any] | None = None, input_fingerprints: dict[str, str] | None = None, mode: str = "main", order_seed: int = 42, private_trace_root: Path | None = None) -> list[dict[str, Any]]:
    selected, limits = list(condition_ids), dict(DEFAULT_CAPS if caps is None else caps)
    if not selected or len(selected) != len(set(selected)) or any(item not in CONDITIONS for item in selected):
        raise ValueError("conditions")
    if mode not in {"main", "staged_qualification"}:
        raise ValueError("run_mode")
    if mode == "staged_qualification":
        if selected != ["fixed_flow"] or output is None:
            raise ValueError("staged_fixed_output_required")
        artifact = output.with_suffix(output.suffix + ".evidence-stage.json")
        if not artifact.exists():
            write_fixed_evidence_stage(claims=claims, archive=archive, blocked_url_families=blocked_url_families, complete=complete, artifact=artifact, embed=embed, caps=limits, input_fingerprints=input_fingerprints, private_trace_root=private_trace_root)
        return replay_fixed_evidence_stage(claims=claims, archive=archive, blocked_url_families=blocked_url_families, artifact=artifact, verdict_complete=verdict_complete, output=output, caps=limits, input_fingerprints=input_fingerprints, private_trace_root=private_trace_root)
    if any(condition in ADAPTIVE_CONDITIONS for condition in selected) and controller_complete is None:
        raise ValueError("lfm_controller_completion_required")
    if verdict_complete is None:
        raise ValueError("hero_verdict_completion_required")
    inputs = _inputs(claims, archive, blocked_url_families, limits, input_fingerprints)
    inputs["execution_mode"] = mode
    inputs["condition_order"] = "balanced_rotation:" + str(order_seed)
    inputs["document_cache_policy"] = "cold_per_condition_model_resident"
    manifest = bind_run(output, cases=claims, conditions=selected, inputs=inputs) if output else None
    journal = output_records(output, manifest=manifest) if output and manifest else {}
    done = set(journal)
    recovered: list[dict[str, Any]] = []
    if output and manifest:
        for key, intent in inflight_pairs(output, manifest=manifest).items():
            if intent["state"] == "finalized":
                if key not in done:
                    raise ValueError("finalized_pair_missing_output")
                if intent["record_sha256"] != _digest(journal[key]):
                    raise ValueError("finalized_pair_output_mismatch")
                continue
            if key in done:
                finish_pair(output, manifest=manifest, record=journal[key])
                continue
            record = {
                "case_id": key[0], "condition": key[1], "run_sha256": manifest["sha256"],
                **_failure("interrupted_unknown_outcome"),
                "usage": CaseLedger(limits).usage(),
                "usage_status": "unavailable_interrupted_unknown_outcome",
                "process_checkpoints": [],
                "process_trace_status": "unavailable_interrupted_unknown_outcome",
                "trace": [{"kind": "recovery", "terminal_reason": "interrupted_unknown_outcome"}],
            }
            append(output, record)
            finish_pair(output, manifest=manifest, record=record)
            done.add(key)
            recovered.append(record)
    if output:
        _sync_streams(output, selected)
    order = list(selected)
    random.Random(order_seed).shuffle(order)
    records: list[dict[str, Any]] = list(recovered)
    for index, claim in enumerate(claims):
        case_id = claim["case_id"]
        if all((case_id, condition) in done for condition in selected):
            continue
        # A claim's closed-world corpus is a shared input for every condition.
        # Do not turn a corrupt/missing corpus into a plausible N×K result.
        corpus = _corpus(archive, claim, blocked_url_families, limits)
        rotation = index % len(order)
        for condition in order[rotation:] + order[:rotation]:
            if (case_id, condition) in done:
                continue
            if output and manifest:
                begin_pair(output, manifest=manifest, case_id=case_id, condition=condition)
            # Query-selected BM25 candidates from one condition must not warm
            # another condition. Corpus text/model weights may remain shared.
            LazyDenseEmbedder.clear_case_cache()
            ledger = CaseLedger(limits)
            try:
                ledger = _ledger(limits, private_trace_root=private_trace_root, condition=condition, case_id=case_id, phase="main")
                ledger.event("cache_policy", document_cache="cold_per_condition", model_weights="shared_resident")
                result = infer_condition(condition, case_id=case_id, claim=claim["claim"], corpus=corpus, blocked_url_families=blocked_url_families, complete=complete, controller_complete=controller_complete, verdict_complete=verdict_complete, ledger=ledger, embed=embed)
            except Exception as error:
                result = _failure(_error(error), status="budget_exhausted" if isinstance(error, BudgetExceeded) else "failed")
            if result.get("status") == "succeeded":
                # A returned success must satisfy shared evidence/provenance
                # postconditions.  Breaching them invalidates the run rather
                # than being relabelled as one condition-local observation.
                evidence = result["prediction"]["evidence"]
                if validate_package({"case_id": case_id, "evidence": evidence, "readiness": result["prediction"].get("readiness")}) == INVALID_ARTIFACT:
                    raise ValueError("invalid_artifact")
                _verify_sources(evidence, corpus)
            record = {"case_id": case_id, "condition": condition, "run_sha256": manifest["sha256"] if manifest else None, **result,
                      "usage": ledger.usage(), "usage_status": "observed",
                      "process_checkpoints": ledger.process_checkpoints,
                      "process_trace_status": "available" if ledger.process_checkpoints else "no_checkpoint_observed",
                      "trace": ledger.events}
            records.append(record)
            if output:
                append(output, record)
                finish_pair(output, manifest=manifest, record=record)
                _sync_streams(output, selected)
        LazyDenseEmbedder.clear_case_cache()
    return records


def write_fixed_evidence_stage(*, claims: list[dict[str, str]], archive: SourceCorpora, blocked_url_families: set[str], complete: Completion, artifact: Path, embed: Callable[[list[str]], Any] | None = None, caps: dict[str, Any] | None = None, input_fingerprints: dict[str, str] | None = None, private_trace_root: Path | None = None) -> dict[str, Any]:
    if artifact.exists():
        raise ValueError("stage_artifact_already_exists")
    limits = dict(DEFAULT_CAPS if caps is None else caps)
    inputs = _inputs(claims, archive, blocked_url_families, limits, input_fingerprints)
    packages = []
    for case in claims:
        ledger = _ledger(limits, private_trace_root=private_trace_root, condition="fixed_flow", case_id=case["case_id"], phase="evidence")
        try:
            corpus = _corpus(archive, case, blocked_url_families, limits)
            evidence = fixed_evidence_policy(case_id=case["case_id"], claim=case["claim"], corpus=corpus, blocked_url_families=blocked_url_families, complete=complete, ledger=ledger, embed=embed)
            readiness = READY_WITH_EVIDENCE if evidence else READY_INSUFFICIENT_EVIDENCE
            package = {"case_id": case["case_id"], "readiness": readiness, "evidence": evidence}
            if validate_package(package) == INVALID_ARTIFACT:
                raise ValueError("invalid_artifact")
            _verify_sources(evidence, corpus)
        except Exception as error:
            package = {"case_id": case["case_id"], "readiness": INVALID_ARTIFACT, "invalid_reason": _error(error), "evidence": []}
        packages.append({**package, "usage": ledger.usage(), "usage_status": "observed",
                         "process_checkpoints": ledger.process_checkpoints,
                         "process_trace_status": "available" if ledger.process_checkpoints else "no_checkpoint_observed",
                         "trace": ledger.events})
        LazyDenseEmbedder.clear_case_cache()
    payload = {"schema": STAGE_SCHEMA, "binding": {"case_ids": [case["case_id"] for case in claims], "condition": "fixed_flow", "inputs": inputs}, "packages": packages}
    payload["sha256"] = _digest(payload)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(artifact.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, artifact)
    return payload


def validate_stage(*, claims: list[dict[str, str]], archive: SourceCorpora, blocked_url_families: set[str], artifact: Path, caps: dict[str, Any], input_fingerprints: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate every binding and package before any verdict callback is invoked."""
    expected_inputs = _inputs(claims, archive, blocked_url_families, caps, input_fingerprints)
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("stage_artifact_missing_or_invalid") from error
    if not isinstance(payload, dict) or payload.get("schema") != STAGE_SCHEMA:
        raise ValueError("stage_artifact_schema")
    unsigned = dict(payload)
    digest = unsigned.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(unsigned):
        raise ValueError("stage_artifact_integrity_failed")
    expected_ids = [case["case_id"] for case in claims]
    if payload.get("binding") != {"case_ids": expected_ids, "condition": "fixed_flow", "inputs": expected_inputs}:
        raise ValueError("stage_artifact_binding_failed")
    packages = payload.get("packages")
    if not isinstance(packages, list) or len(packages) != len(expected_ids) or any(not isinstance(item, dict) for item in packages):
        raise ValueError("stage_case_accounting")
    if [item.get("case_id") for item in packages] != expected_ids:
        raise ValueError("stage_case_accounting")
    problems: dict[str, str] = {}
    for claim, package in zip(claims, packages):
        try:
            if validate_package(package) == INVALID_ARTIFACT:
                raise ValueError(_stage_problem(package))
            _verify_sources(package["evidence"], _corpus(archive, claim, blocked_url_families, caps))
            probe = CaseLedger(caps)
            probe.restore_usage(package.get("usage", {}))
            if probe.verdict_calls != 0:
                raise ValueError("evidence_stage_contains_verdict_call")
            if not isinstance(package.get("trace"), list) or any(not isinstance(item, dict) for item in package["trace"]):
                raise ValueError("stage_trace_schema")
            if (not isinstance(package.get("process_checkpoints"), list)
                    or any(not isinstance(item, dict) for item in package["process_checkpoints"])):
                raise ValueError("stage_process_checkpoint_schema")
        except Exception as error:
            problems[claim["case_id"]] = _error(error)
    return payload, problems


def replay_fixed_evidence_stage(*, claims: list[dict[str, str]], archive: SourceCorpora, blocked_url_families: set[str], artifact: Path, verdict_complete: Completion | None, output: Path, caps: dict[str, Any] | None = None, input_fingerprints: dict[str, str] | None = None, private_trace_root: Path | None = None) -> list[dict[str, Any]]:
    limits = dict(DEFAULT_CAPS if caps is None else caps)
    payload, problems = validate_stage(claims=claims, archive=archive, blocked_url_families=blocked_url_families, artifact=artifact, caps=limits, input_fingerprints=input_fingerprints)
    inputs = dict(payload["binding"]["inputs"], stage_sha256=sha256(artifact), execution_mode="staged_qualification")
    manifest = bind_run(output, cases=claims, conditions=["fixed_flow"], inputs=inputs)
    done = completed(output, manifest=manifest)
    _sync_streams(output, ["fixed_flow"])
    if not problems and verdict_complete is None:
        raise ValueError("hero_verdict_completion_required")
    records = []
    for case, package in zip(claims, payload["packages"]):
        if (case["case_id"], "fixed_flow") in done:
            continue
        ledger = _ledger(limits, private_trace_root=private_trace_root, condition="fixed_flow", case_id=case["case_id"], phase="verdict")
        try:
            ledger.restore_usage(package.get("usage", {}))
            ledger.events.extend(package.get("trace", []))
            ledger.process_checkpoints.extend(package.get("process_checkpoints", []))
        except (ValueError, TypeError):
            ledger = CaseLedger(limits)
            ledger.event("evidence_stage_usage_unavailable")
        if problems:
            result = _failure(problems.get(case["case_id"], "staged_verdict_gate_blocked"))
        else:
            try:
                prediction = fixed_verdict_policy(claim=case["claim"], evidence=package["evidence"], verdict_complete=verdict_complete, ledger=ledger)
                result = {"status": "succeeded", "prediction": prediction}
            except Exception as error:
                result = _failure(_error(error), status="budget_exhausted" if isinstance(error, BudgetExceeded) else "failed")
        record = {"case_id": case["case_id"], "condition": "fixed_flow", "run_sha256": manifest["sha256"], **result,
                  "usage": ledger.usage(), "usage_status": package.get("usage_status", "unavailable"),
                  "process_checkpoints": ledger.process_checkpoints,
                  "process_trace_status": package.get("process_trace_status", "unavailable"),
                  "trace": ledger.events}
        append(output, record)
        records.append(record)
        _sync_streams(output, ["fixed_flow"])
    return records
