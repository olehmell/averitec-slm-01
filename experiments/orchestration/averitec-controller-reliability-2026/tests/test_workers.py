from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workers


CASE = {"case_id": "averitec-dev-0000", "claim": "A test claim.", "split": "dev"}


class _Span:
    def __init__(self, record: dict) -> None:
        self.record = record

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def update(self, **kwargs: object) -> None:
        self.record.setdefault("updates", []).append(kwargs)


class _Tracer:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def span(self, name: str, **kwargs: object) -> _Span:
        record = {"name": name, **kwargs}
        self.records.append(record)
        return _Span(record)


def _corpus(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "dev.zip"
    with ZipFile(archive, "w") as target:
        target.writestr("output_dev/0.json", json.dumps({"url": "https://source.example/a", "url2text": ["Evidence about a test claim."]}))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"schema": "averitec-source-corpora/v1", "stores": {"dev": {"archives": [{"path": "dev.zip", "sha256": digest, "first_index": 0, "last_index": 0, "member_template": "output_dev/{index}.json"}]}}}))
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"blocked_url_families": []}))
    return manifest


def _transport(_endpoint: str, _model: str, **_kwargs: object):
    def complete(prompt: str, _maximum: int, *_args: object, **_more: object) -> str:
        if "Decompose factual facets" in prompt:
            return '{"facets":[{"id":"f1","text":"test fact"}]}'
        if "retrieval hypotheses" in prompt:
            return '{"hyde":["test claim evidence"]}'
        if '"contract_version":"semantic-span-fidelity/v4-segments"' in prompt:
            sources = json.loads(prompt.rsplit("Sources: ", 1)[1])
            return json.dumps({"contract_version": "semantic-span-fidelity/v4-segments", "qas": [{"facet_ids": ["f1"], "question": "What evidence exists?", "segment_ids": [sources[0]["segments"][0]["segment_id"]], "support_validated": True, "support_score": 1.0, "relevance_score": 1.0}]})
        if prompt.startswith('Return JSON {"covered_facet_ids"'):
            return '{"covered_facet_ids":["f1"],"gaps":[]}'
        return "The supplied evidence addresses the claim.\nVerdict: Supported"
    complete.accepts_timeout = True
    complete.accepts_json_schema = True
    return complete


def _tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> workers.PipelineTools:
    manifest = _corpus(tmp_path)
    monkeypatch.setattr(workers, "invoke_openai", _transport)
    monkeypatch.setattr(workers, "_fetch_model_ids", lambda _endpoint, _timeout: ("Qwen/Qwen3.5-4B",))
    return workers.PipelineTools(CASE, endpoint="http://mock/v1", model="Qwen/Qwen3.5-4B", source_corpora=manifest, exclusions=tmp_path / "exclusions.json", root=tmp_path)


def test_gold_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid_case"):
        workers.SyntheticTools({**CASE, "label": "Supported"})


def test_pipeline_returns_only_shared_metric_contract_and_never_dense(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool(tmp_path, monkeypatch)
    monkeypatch.setattr(workers.LazyDenseEmbedder, "encode_queries", lambda *_: pytest.fail("dense embedding called"))
    outputs = [tool.run(action) for action in ("decompose", "queries", "retrieve", "qa", "coverage", "select", "verdict")]
    for output in outputs:
        assert set(output) == {"status", "metrics", "error_code"}
        assert set(output["metrics"]) == {"facet_count", "query_count", "candidate_count", "qa_count", "selected_count"}
        assert all(isinstance(value, int) and value >= 0 for value in output["metrics"].values())
        assert "A test claim" not in json.dumps(output)
        assert "averitec-dev-0000" not in json.dumps(output)
    assert outputs[-1]["status"] == "ok"
    assert tool._state.coverage == {"covered_facet_ids": ["f1"], "score": 1.0}
    receipt = tool.preparation_receipt()
    assert receipt["model_binding"] == {"requested": "Qwen/Qwen3.5-4B", "reported": "Qwen/Qwen3.5-4B"}
    assert receipt["endpoint_health"] == {"checked": True, "identity_verified": True}
    assert receipt["corpus"]["member_verified"] is True
    assert len(receipt["stages"]) == 7
    assert all(set(row) == {"stage", "status", "error_code", "model_calls", "output_content_sha256"} for row in receipt["stages"])
    assert receipt["ledger"]["verdict_calls"] == 1
    assert "A test claim" not in json.dumps(receipt)
    assert tool.run("verdict") == {"status": "invalid", "metrics": outputs[-1]["metrics"], "error_code": "terminal_already_called"}


def test_empty_and_failed_steps_stay_worker_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool(tmp_path, monkeypatch)
    assert tool.run("qa")["status"] == "empty"
    assert tool.run("retrieve") == {"status": "invalid", "metrics": {"facet_count": 0, "query_count": 0, "candidate_count": 0, "qa_count": 0, "selected_count": 0}, "error_code": "worker_failure"}
    def timeout_transport(*_args: object, **_kwargs: object):
        def complete(*_call_args: object, **_call_kwargs: object) -> str:
            raise TimeoutError()
        return complete
    monkeypatch.setattr(workers, "invoke_openai", timeout_transport)
    monkeypatch.setattr(workers, "_fetch_model_ids", lambda _endpoint, _timeout: ("Qwen/Qwen3.5-4B",))
    timeout_tool = workers.PipelineTools(CASE, endpoint="http://mock/v1", model="Qwen/Qwen3.5-4B", source_corpora=_corpus(tmp_path / "timeout"), exclusions=tmp_path / "timeout" / "exclusions.json", root=tmp_path / "timeout")
    assert timeout_tool.run("decompose")["status"] == "timeout"


def test_synthetic_tools_are_deterministic_smoke_only() -> None:
    tool = workers.SyntheticTools(CASE)
    outputs = [tool.run(action) for action in ("decompose", "queries", "retrieve", "qa", "coverage", "select", "verdict")]
    assert [item["status"] for item in outputs] == ["ok", "ok", "empty", "empty", "ok", "empty", "ok"]


def test_terminal_malformed_output_allows_one_engine_visible_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"terminal": 0}

    def retry_transport(_endpoint: str, _model: str, **_kwargs: object):
        base = _transport(_endpoint, _model)
        def complete(prompt: str, maximum: int, *args: object, **kwargs: object) -> str:
            if "numbered Q/A evidence" in prompt:
                attempts["terminal"] += 1
                return "garbled" if attempts["terminal"] == 1 else "Evidence is sufficient.\nVerdict: Supported"
            return base(prompt, maximum, *args, **kwargs)
        return complete

    monkeypatch.setattr(workers, "invoke_openai", retry_transport)
    monkeypatch.setattr(workers, "_fetch_model_ids", lambda _endpoint, _timeout: ("Qwen/Qwen3.5-4B",))
    tool = workers.PipelineTools(CASE, endpoint="http://mock/v1", model="Qwen/Qwen3.5-4B", source_corpora=_corpus(tmp_path), exclusions=tmp_path / "exclusions.json", root=tmp_path)
    for action in ("decompose", "queries", "retrieve", "qa", "coverage", "select"):
        assert tool.run(action)["status"] == "ok"
    assert tool.run("verdict")["status"] == "invalid"
    assert tool.run("verdict")["status"] == "ok"
    assert attempts["terminal"] == tool._terminal_attempts == 2


def test_endpoint_health_identity_is_checked_before_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _corpus(tmp_path)
    order: list[str] = []
    monkeypatch.setattr(workers, "_fetch_model_ids", lambda _endpoint, _timeout: order.append("health") or ("Qwen/Qwen3.5-4B",))
    monkeypatch.setattr(workers, "invoke_openai", lambda *_args, **_kwargs: order.append("transport") or _transport(*_args, **_kwargs))
    workers.PipelineTools(CASE, endpoint="http://mock/v1/chat/completions", model="Qwen/Qwen3.5-4B", source_corpora=manifest, exclusions=tmp_path / "exclusions.json", root=tmp_path)
    assert order == ["health", "transport"]
    monkeypatch.setattr(workers, "_fetch_model_ids", lambda *_args: ("other-model",))
    with pytest.raises(ValueError, match="model_endpoint_identity"):
        workers.PipelineTools(CASE, endpoint="http://mock/v1", model="Qwen/Qwen3.5-4B", source_corpora=manifest, exclusions=tmp_path / "exclusions.json", root=tmp_path)


def test_tracer_captures_tool_state_and_nested_generation_without_transport_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _Tracer()
    manifest = _corpus(tmp_path)
    monkeypatch.setattr(workers, "invoke_openai", _transport)
    monkeypatch.setattr(workers, "_fetch_model_ids", lambda _endpoint, _timeout: ("Qwen/Qwen3.5-4B",))
    tool = workers.PipelineTools(CASE, endpoint="http://endpoint.example/v1", model="Qwen/Qwen3.5-4B", source_corpora=manifest, exclusions=tmp_path / "exclusions.json", root=tmp_path, tracer=tracer)
    assert tool.run("decompose")["status"] == "ok"
    tool_span = tracer.records[0]
    generation = tracer.records[1]
    assert tool_span["name"] == "averitec.worker.decompose"
    assert tool_span["kind"] == "tool"
    assert set(tool_span["input"]["state"]) == {"facets", "query_plan", "candidate_passages", "qa_candidates", "coverage", "selected_evidence", "readiness"}
    candidates = tool_span["input"]["state"]["candidate_passages"]
    assert set(candidates) == {"total_count", "ordered_ids_sha256", "qa_window"}
    assert candidates["total_count"] == 0 and candidates["qa_window"] == []
    assert generation["name"] == "averitec.worker.generation"
    assert generation["kind"] == "generation"
    assert set(generation["input"]) == {"prompt", "max_tokens", "json_schema"}
    assert generation["model"] == "Qwen/Qwen3.5-4B"
    assert generation["updates"][-1]["output"]["completion"].startswith('{"facets"')
    trace_json = json.dumps(tracer.records)
    assert "endpoint.example" not in trace_json
    assert "TYPESAFE_API_KEY" not in trace_json


def test_trace_snapshot_bounds_candidate_text_and_binds_full_ranking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool(tmp_path, monkeypatch)
    tool.run("decompose")
    tool.run("queries")
    tool.run("retrieve")
    candidate = tool._state.candidate_passages[0]
    tool._state.candidate_passages = [candidate] * 12
    snapshot = tool.trace_snapshot()["candidate_passages"]
    assert snapshot["total_count"] == 12
    assert len(snapshot["qa_window"]) == 10
    assert len(snapshot["ordered_ids_sha256"]) == 64
    assert "source_text" not in snapshot["qa_window"][0]
    assert snapshot["qa_window"][0]["source_text_length"] == len(candidate.source_text)
