from __future__ import annotations

import base64
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tracing


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_completed_span_reader_streams_without_loading_whole_journal(tmp_path, monkeypatch):
    journal = tmp_path / 'stream.traces.jsonl'
    recorder = tracing.TraceRecorder(journal, 'stream-run', 'stream-trial')
    with recorder.span('trial'):
        pass
    monkeypatch.setattr(Path, 'read_text', lambda *args, **kwargs: pytest.fail('whole journal read'))
    assert len(list(tracing._iter_completed(journal))) == 1


def test_recorder_writes_durable_nested_spans_and_redacts_credentials(tmp_path: Path) -> None:
    journal = tmp_path / "trace.jsonl"
    recorder = tracing.TraceRecorder(journal, "run-a", "trial-a", {"condition": "nominal", "api_key": "nope"})
    with recorder.span("trial", input={"prompt": "safe", "secret": "nope"}) as trial:
        with recorder.span("controller.request", kind="generation", model="model-a", input={"token": "nope"}) as generation:
            generation.update(output={"choice": "search"}, usage={"input": 3, "output": 1})
        trial.update(output={"outcome": "ok"})

    rows = _rows(journal)
    assert [row["event"] for row in rows] == ["span_start", "span_start", "span_end", "span_end"]
    assert rows[1]["parent_span_id"] == rows[0]["span_id"]
    assert rows[0]["trace_id"] == recorder.trace_id and len(recorder.trace_id) == 32
    assert rows[1]["kind"] == "generation" and len(rows[1]["span_id"]) == 16
    assert rows[0]["input"]["secret"] == "[REDACTED]"
    assert rows[1]["metadata"]["api_key"] == "[REDACTED]"
    assert rows[2]["usage"] == {"input": 3, "output": 1}


def test_exception_is_closed_with_bounded_error_and_start_survives_interruption(tmp_path: Path) -> None:
    journal = tmp_path / "trace.jsonl"
    recorder = tracing.TraceRecorder(journal, "run-a", "trial-a")
    with pytest.raises(RuntimeError):
        with recorder.span("tool"):
            raise RuntimeError("TOP_SECRET and /private/path")
    rows = _rows(journal)
    assert rows[-1]["level"] == "ERROR" and rows[-1]["status_message"] == "exception"
    assert "TOP_SECRET" not in journal.read_text()
    # A process that crashes between these records retains the begin record.
    tracing._append(journal, {"event": "span_start", "timestamp": "2026-01-01T00:00:00.000000Z",
                              "trace_id": recorder.trace_id, "span_id": "a" * 16, "name": "unfinished",
                              "kind": "span", "parent_span_id": None, "input": None, "model": None, "metadata": {}})
    assert len(tracing._completed_spans(journal)) == 1


def test_recorder_rejects_oversized_event_before_writing(tmp_path: Path) -> None:
    journal = tmp_path / "oversized.traces.jsonl"
    recorder = tracing.TraceRecorder(journal, "run-a", "trial-a")
    with pytest.raises(ValueError, match="trace_event_too_large"):
        with recorder.span("oversized", input={"payload": "x" * tracing.TRACE_MAX_EVENT_BYTES}):
            pass
    assert not journal.exists()


def test_export_uses_official_otlp_path_basic_auth_and_historical_timestamps(tmp_path: Path, monkeypatch) -> None:
    journal, dotenv = tmp_path / "trace.jsonl", tmp_path / ".env"
    dotenv.write_text("LANGFUSE_PUBLIC_KEY=PUBLIC\nLANGFUSE_SECRET_KEY=SECRET\nLANGFUSE_BASE_URL=https://example.invalid\n")
    recorder = tracing.TraceRecorder(journal, "run-a", "trial-a", {"role": "development"})
    with recorder.span("trial", input={"task": "x"}) as span:
        span.update(output={"outcome": "ok"})
    captured = {}

    class Response:
        status = 202
        def geturl(self): return "https://example.invalid/api/public/otel/v1/traces"
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *_args): return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(tracing, "_export_urlopen", fake_urlopen)
    result = tracing.export_traces(journal, dotenv_path=dotenv)
    assert result["status"] == "exported" and result["completed_spans"] == result["accepted_spans"] == 1
    assert captured["url"] == "https://example.invalid/api/public/otel/v1/traces"
    assert captured["headers"]["X-langfuse-ingestion-version"] == "4"
    assert captured["headers"]["Authorization"] == "Basic " + base64.b64encode(b"PUBLIC:SECRET").decode()
    span_payload = captured["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span_payload["traceId"] == recorder.trace_id
    assert span_payload["startTimeUnixNano"].isdigit() and span_payload["endTimeUnixNano"].isdigit()
    assert all("SECRET" not in json.dumps(value) for value in (result, captured["payload"]))


def test_tool_kind_and_token_counts_survive_redaction_and_map_to_otlp() -> None:
    value = tracing.redact({"input_tokens": 4, "output_tokens": 2, "api_key": "hidden"})
    assert value == {"input_tokens": 4, "output_tokens": 2, "api_key": "[REDACTED]"}
    span = {"trace_id": "a" * 32, "span_id": "b" * 16, "parent_span_id": None, "name": "tool",
            "kind": "tool", "started_at": "2026-01-01T00:00:00.000000Z", "ended_at": "2026-01-01T00:00:01.000000Z",
            "input": {}, "output": {}, "metadata": {}, "usage": {"input_tokens": 4, "output_tokens": 2}, "level": "DEFAULT", "model": None, "status_message": None}
    attributes = tracing._otlp_payload([span])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
    values = {item["key"]: item["value"]["stringValue"] for item in attributes}
    assert values["langfuse.observation.type"] == "tool"
    assert json.loads(values["langfuse.observation.usage_details"]) == {"input": 4, "output": 2, "total": 6}


def test_export_failure_does_not_write_receipt_or_disclose_credentials(tmp_path: Path, monkeypatch) -> None:
    journal, dotenv = tmp_path / "trace.jsonl", tmp_path / ".env"
    dotenv.write_text("LANGFUSE_PUBLIC_KEY=PUBLIC\nLANGFUSE_SECRET_KEY=SECRET\nLANGFUSE_HOST=https://example.invalid\n")
    recorder = tracing.TraceRecorder(journal, "run-a", "trial-a")
    with recorder.span("trial"):
        pass
    def fail(_request, timeout): raise OSError("SECRET")
    monkeypatch.setattr(tracing, "_export_urlopen", fail)
    with pytest.raises(tracing.TraceExportError, match="langfuse_export_transport_failure"):
        tracing.export_traces(journal, dotenv_path=dotenv)
    assert "SECRET" not in journal.read_text() and list(tmp_path.glob("*receipt*")) == []


def test_export_without_completed_span_is_local_noop(tmp_path: Path, monkeypatch) -> None:
    journal, dotenv = tmp_path / "trace.jsonl", tmp_path / ".env"
    dotenv.write_text("LANGFUSE_PUBLIC_KEY=PUBLIC\nLANGFUSE_SECRET_KEY=SECRET\nLANGFUSE_HOST=https://example.invalid\n")
    tracing._append(journal, {"event": "span_start", "timestamp": "2026-01-01T00:00:00.000000Z",
                              "trace_id": "a" * 32, "span_id": "b" * 16, "name": "unfinished",
                              "kind": "span", "parent_span_id": None, "input": None, "model": None, "metadata": {}})
    monkeypatch.setattr(tracing, "_export_urlopen", lambda *_args, **_kwargs: pytest.fail("must not export"))
    assert tracing.export_traces(journal, dotenv_path=dotenv) == {"status": "no_completed_spans", "completed_spans": 0, "trace_ids": []}


def test_export_greedily_splits_completed_spans_and_accepts_protobuf_string_zero(tmp_path: Path, monkeypatch) -> None:
    journal, dotenv = tmp_path / "trace.jsonl", tmp_path / ".env"
    dotenv.write_text("LANGFUSE_PUBLIC_KEY=PUBLIC\nLANGFUSE_SECRET_KEY=SECRET\nLANGFUSE_HOST=https://example.invalid\n")
    recorder = tracing.TraceRecorder(journal, "run-a", "trial-a")
    for index in range(3):
        with recorder.span("span-" + str(index), kind="tool") as span:
            span.update(usage={"input_tokens": index + 1, "output_tokens": 1})
    calls = []
    class Response:
        status = 200
        def geturl(self): return "https://example.invalid/api/public/otel/v1/traces"
        def read(self): return b'{"partialSuccess":{"rejectedSpans":"0"}}'
        def __enter__(self): return self
        def __exit__(self, *_args): return False
    def send(request, timeout):
        calls.append(json.loads(request.data)); return Response()
    monkeypatch.setattr(tracing, "_export_urlopen", send)
    monkeypatch.setattr(tracing, "EXPORT_MAX_SPANS", 1)
    result = tracing.export_traces(journal, dotenv_path=dotenv)
    assert result["accepted_spans"] == 3 and len(calls) == 3
    usage = calls[0]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
    assert any(item["key"] == "langfuse.observation.usage_details" for item in usage)


def test_export_rejects_single_oversized_span_without_transmission(tmp_path: Path, monkeypatch) -> None:
    journal, dotenv = tmp_path / "trace.jsonl", tmp_path / ".env"
    dotenv.write_text("LANGFUSE_PUBLIC_KEY=PUBLIC\nLANGFUSE_SECRET_KEY=SECRET\nLANGFUSE_HOST=https://example.invalid\n")
    recorder = tracing.TraceRecorder(journal, "run-a", "trial-a")
    with recorder.span("large", input={"payload": "x" * 2000}):
        pass
    monkeypatch.setattr(tracing, "EXPORT_MAX_BYTES", 100)
    monkeypatch.setattr(tracing, "_export_urlopen", lambda *_args, **_kwargs: pytest.fail("must not export"))
    with pytest.raises(tracing.TraceExportError, match="single_span_too_large"):
        tracing.export_traces(journal, dotenv_path=dotenv)
