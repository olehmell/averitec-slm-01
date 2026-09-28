"""Durable local-first traces with optional, explicit Langfuse OTLP export.

No network activity occurs while a :class:`TraceRecorder` is recording.  The
separate ``export_traces`` operation sends completed spans only, using
Langfuse's current OTLP/HTTP endpoint rather than its deprecated ingestion API.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Literal
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


LANGFUSE_OTLP_TRACES_PATH = "/api/public/otel/v1/traces"
# The current official Python SDK is v4; direct OTLP keeps historical timestamp
# and stable span-id control without adding tracing overhead to measurements.
LANGFUSE_PYTHON_SDK_RECOMMENDATION = "langfuse>=4.7.0"
EXPORT_MAX_SPANS = 100
EXPORT_MAX_BYTES = 2_000_000
TRACE_MAX_EVENT_BYTES = 2_000_000
_SENSITIVE = {"api_key", "apikey", "authorization", "credential", "credentials", "password", "secret", "secret_key", "token", "access_token", "bearer_token", "typesafe_api_key", "langfuse_secret_key", "langfuse_public_key"}


class TraceExportError(RuntimeError):
    """Public-safe exporter failure; its text never contains response details."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _export_urlopen(request: Request, timeout: int):
    """Credential-bearing export requests must fail on every HTTP redirect."""
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _nanoseconds(value: str) -> str:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return str(int(timestamp.timestamp() * 1_000_000_000))


def _identifier(seed: str, length: int) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:length]


def redact(value: Any) -> Any:
    """Recursively remove credential-shaped fields without rejecting safe data."""
    if isinstance(value, dict):
        return {str(key): ("[REDACTED]" if str(key).lower() in _SENSITIVE
                           else redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(redact(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _append(path: Path, value: dict[str, Any]) -> None:
    payload = (_json(value) + "\n").encode("utf-8")
    if len(payload) > TRACE_MAX_EVENT_BYTES:
        raise ValueError("trace_event_too_large")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


@dataclass
class TraceSpan:
    recorder: "TraceRecorder"
    name: str
    kind: Literal["span", "generation", "tool"]
    span_id: str
    parent_span_id: str | None
    started_at: str
    input: Any = None
    model: str | None = None
    metadata: dict[str, Any] | None = None
    output: Any = None
    usage: dict[str, int] | None = None
    level: str = "DEFAULT"
    status_message: str | None = None
    ended: bool = False

    def update(self, *, output: Any = None, usage: dict[str, int] | None = None,
               metadata: dict[str, Any] | None = None, level: str | None = None,
               status_message: str | None = None) -> "TraceSpan":
        if output is not None:
            self.output = output
        if usage is not None:
            self.usage = usage
        if metadata is not None:
            self.metadata = {**(self.metadata or {}), **metadata}
        if level is not None:
            self.level = level if level in {"DEBUG", "DEFAULT", "WARNING", "ERROR"} else "ERROR"
        if status_message is not None:
            self.status_message = status_message[:200]
        return self

    def end(self, *, exception: bool = False) -> None:
        if self.ended:
            return
        self.ended = True
        if exception:
            self.level, self.status_message = "ERROR", "exception"
        _append(self.recorder.path, {
            "event": "span_end", "timestamp": _utc_now(), "trace_id": self.recorder.trace_id,
            "span_id": self.span_id, "output": self.output, "usage": self.usage,
            "metadata": self.metadata, "level": self.level, "status_message": self.status_message,
        })


@dataclass
class TraceRecorder:
    path: Path | str
    run_id: str
    trial_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(init=False)
    _counter: int = field(default=0, init=False)
    _local: threading.local = field(default_factory=threading.local, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.trace_id = _identifier("langfuse-trace:" + self.run_id + ":" + self.trial_id, 32)

    def _stack(self) -> list[TraceSpan]:
        if not hasattr(self._local, "spans"):
            self._local.spans = []
        return self._local.spans

    @contextmanager
    def span(self, name: str, kind: Literal["span", "generation", "tool"] = "span", input: Any = None,
             model: str | None = None, metadata: dict[str, Any] | None = None) -> Iterator[TraceSpan]:
        if not isinstance(name, str) or not name or kind not in {"span", "generation", "tool"}:
            raise ValueError("trace_span_arguments")
        self._counter += 1
        stack = self._stack()
        parent = stack[-1].span_id if stack else None
        span = TraceSpan(self, name, kind, _identifier(f"{self.trace_id}:{self._counter}:{name}", 16), parent,
                         _utc_now(), input=input, model=model,
                         metadata={**self.metadata, **(metadata or {})})
        _append(self.path, {"event": "span_start", "timestamp": span.started_at, "trace_id": self.trace_id,
                            "span_id": span.span_id, "parent_span_id": parent, "name": name, "kind": kind,
                            "input": input, "model": model, "metadata": span.metadata})
        stack.append(span)
        try:
            yield span
        except BaseException:
            span.end(exception=True)
            raise
        else:
            span.end()
        finally:
            if stack and stack[-1] is span:
                stack.pop()


def _dotenv_values(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    allowed = {"LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST"}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key.strip() in allowed:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key.strip()] = value
    return values


def _credentials(dotenv_path: Path | str | None) -> tuple[str, str, str]:
    dotenv = _dotenv_values(Path(dotenv_path) if dotenv_path is not None else None)
    def take(name: str) -> str | None:
        return os.environ.get(name) or dotenv.get(name)
    public, secret = take("LANGFUSE_PUBLIC_KEY"), take("LANGFUSE_SECRET_KEY")
    base = take("LANGFUSE_BASE_URL") or take("LANGFUSE_HOST")
    if not public or not secret or not base or not base.startswith(("https://", "http://")):
        raise TraceExportError("langfuse_credentials_missing")
    return public, secret, base.rstrip("/")


def _iter_completed(path: Path) -> Iterator[dict[str, Any]]:
    starts: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        raise TraceExportError("trace_journal_missing")
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceExportError("trace_journal_invalid") from error
            if not isinstance(row, dict):
                raise TraceExportError("trace_journal_invalid")
            key = (row.get("trace_id"), row.get("span_id"))
            if row.get("event") == "span_start" and all(isinstance(item, str) for item in key):
                starts[key] = row
            elif row.get("event") == "span_end" and key in starts:
                start = starts.pop(key)
                yield {**start, **row, "started_at": start["timestamp"], "ended_at": row["timestamp"]}


def _completed_spans(path: Path) -> list[dict[str, Any]]:
    """Compatibility helper for small local callers and tests."""
    return list(_iter_completed(path))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attribute(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": _json(value)}}


def _string_attribute(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _usage_details(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    for target, names in (("input", ("input", "input_tokens", "prompt_tokens")),
                          ("output", ("output", "output_tokens", "completion_tokens"))):
        for name in names:
            item = value.get(name)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                result[target] = item
                break
    if result:
        result.setdefault("total", sum(result.values()))
    return result or None


def _otlp_payload(spans: list[dict[str, Any]]) -> dict[str, Any]:
    payload = []
    for row in spans:
        attributes = [_string_attribute("langfuse.observation.type", row["kind"]),
                      _attribute("langfuse.observation.input", row.get("input")),
                      _attribute("langfuse.observation.output", row.get("output")),
                      _string_attribute("langfuse.observation.level", row.get("level", "DEFAULT"))]
        for key, value in (row.get("metadata") or {}).items():
            safe_key = "".join(char if char.isalnum() or char in "_-" else "_" for char in str(key))[:80]
            if safe_key:
                attributes.append(_attribute("langfuse.observation.metadata." + safe_key, value))
        if row.get("model"):
            attributes.append(_string_attribute("langfuse.observation.model.name", row["model"]))
        usage = _usage_details(row.get("usage"))
        if usage is not None:
            attributes.append(_attribute("langfuse.observation.usage_details", usage))
        if row.get("status_message"):
            attributes.append(_string_attribute("langfuse.observation.status_message", row["status_message"]))
        payload.append({"traceId": row["trace_id"], "spanId": row["span_id"], "parentSpanId": row.get("parent_span_id") or "",
                        "name": row["name"], "startTimeUnixNano": _nanoseconds(row["started_at"]),
                        "endTimeUnixNano": _nanoseconds(row["ended_at"]), "attributes": attributes})
    return {"resourceSpans": [{"scopeSpans": [{"scope": {"name": "averitec-controller-reliability"}, "spans": payload}]}]}


def export_traces(path: Path | str, *, dotenv_path: Path | str | None, smoke: bool = False) -> dict[str, Any]:
    """Explicitly export completed journal spans; no success receipt is written.

    Stable trace/span IDs make reruns address the same OTLP observations.  The
    caller owns retries and any separate remote-import smoke verification.
    """
    public, secret, base = _credentials(dotenv_path)
    accepted = 0
    completed = 0
    trace_ids: set[str] = set()
    encoded = base64.b64encode(f"{public}:{secret}".encode("utf-8")).decode("ascii")
    def send(chunk: list[dict[str, Any]], wire: bytes) -> None:
        nonlocal accepted
        request = Request(base + LANGFUSE_OTLP_TRACES_PATH, data=wire, method="POST",
                          headers={"Content-Type": "application/json", "Authorization": "Basic " + encoded,
                                   "x-langfuse-ingestion-version": "4"})
        try:
            with _export_urlopen(request, timeout=10 if smoke else 30) as response:  # nosec B310: explicit user-configured host
                if response.geturl() != base + LANGFUSE_OTLP_TRACES_PATH or not 200 <= response.status < 300:
                    raise TraceExportError("langfuse_export_http_status")
                body = response.read()
        except HTTPError as error:
            raise TraceExportError("langfuse_export_http_" + str(error.code)) from None
        except (URLError, OSError):
            raise TraceExportError("langfuse_export_transport_failure") from None
        try:
            acknowledgement = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise TraceExportError("langfuse_export_invalid_ack") from None
        partial = acknowledgement.get("partialSuccess", {}) if isinstance(acknowledgement, dict) else None
        rejected = partial.get("rejectedSpans", 0) if isinstance(partial, dict) else None
        if isinstance(rejected, str) and rejected.isdigit():
            rejected = int(rejected)
        if not isinstance(rejected, int) or isinstance(rejected, bool) or rejected:
            raise TraceExportError("langfuse_export_partial_rejection")
        accepted += len(chunk)

    batch: list[dict[str, Any]] = []
    for span in _iter_completed(Path(path)):
        completed += 1
        trace_ids.add(span["trace_id"])
        candidate = batch + [span]
        wire = _json(_otlp_payload(candidate)).encode("utf-8")
        if len(candidate) <= EXPORT_MAX_SPANS and len(wire) <= EXPORT_MAX_BYTES:
            batch = candidate
            continue
        if not batch:
            raise TraceExportError("langfuse_export_single_span_too_large")
        send(batch, _json(_otlp_payload(batch)).encode("utf-8"))
        single_wire = _json(_otlp_payload([span])).encode("utf-8")
        if len(single_wire) > EXPORT_MAX_BYTES:
            raise TraceExportError("langfuse_export_single_span_too_large")
        batch = [span]
    if batch:
        send(batch, _json(_otlp_payload(batch)).encode("utf-8"))
    if not completed:
        return {"status": "no_completed_spans", "completed_spans": 0, "trace_ids": []}
    return {"status": "exported", "completed_spans": completed, "accepted_spans": accepted,
            "trace_ids": sorted(trace_ids), "journal_sha256": _file_hash(Path(path))}
