from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemini_count_preflight as count  # noqa: E402


def test_count_sends_full_generate_request_and_reads_total(monkeypatch):
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"totalTokens":423}'

    class Opener:
        def open(self, request, timeout):
            sent["url"] = request.full_url
            sent["body"] = json.loads(request.data)
            sent["timeout"] = timeout
            sent["key"] = request.get_header("X-goog-api-key")
            return Response()

    monkeypatch.setattr(count, "build_opener", lambda handler: Opener())
    body = {"systemInstruction": {"parts": [{"text": "Rule"}]},
            "contents": [{"role": "user", "parts": [{"text": "Evidence"}]}],
            "generationConfig": {"maxOutputTokens": 2048}}
    assert count._count(body, "hidden-key", 30) == 423
    assert sent == {"url": count.URL, "body": {"generateContentRequest": {
        "model": "models/gemini-3.1-flash-lite", **body}}, "timeout": 30, "key": "hidden-key"}


def test_failed_count_is_recorded_and_cannot_retry(tmp_path: Path, monkeypatch):
    manifest, candidates, gate = (tmp_path / name for name in ("manifest", "candidates", "gate"))
    for path in (manifest, candidates, gate):
        path.write_text("{}", encoding="utf-8")
    policy = {"selected_ordinal": 4, "generate_request_sha256": "a" * 64,
              "maximum_wall_seconds": 120, "timeout_seconds": 30}
    monkeypatch.setattr(count, "check", Mock(return_value=(policy, {"contents": []})))
    monkeypatch.setattr(count, "_count", Mock(side_effect=TimeoutError("unknown")))
    output = tmp_path / "output"
    receipt = count.execute(manifest, candidates, output, gate, key="hidden-key")
    assert receipt["requests_attempted"] == 1 and receipt["complete"] is False
    assert json.loads((output / "result.json").read_text())["status"] == "unknown_submission_outcome"
    assert "hidden-key" not in (output / "result.json").read_text()
    with pytest.raises(ValueError, match="output_or_credential"):
        count.execute(manifest, candidates, output, gate, key="hidden-key")
    assert count._count.call_count == 1
