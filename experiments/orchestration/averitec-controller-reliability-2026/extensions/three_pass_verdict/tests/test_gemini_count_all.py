from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemini_count_all as collector  # noqa: E402


def _setup(tmp_path: Path, monkeypatch, counts):
    manifest, candidates, smoke_receipt, gate = (tmp_path / name for name in ("manifest", "candidates", "smoke", "gate"))
    for path in (manifest, candidates, smoke_receipt, gate):
        path.write_text("{}", encoding="utf-8")
    policy = {"smoke_receipt_sha256": "a" * 64, "maximum_wall_seconds": 120,
              "timeout_seconds": 30, "input_margin_tokens": 128,
              "output_request_limit_tokens": 2048, "context_limit_tokens": 32768,
              "inter_call_delay_seconds": 0, "request_count": 2,
              "model": collector.smoke.MODEL, "endpoint": collector.smoke.URL,
              "source_commit": "b" * 40, "code_sha256": {"runner": "c" * 64}}
    records = [{"observation_sha256": str(i), "request_sha256": str(i), "serialized_bytes": 5}
               for i in range(2)]
    monkeypatch.setattr(collector, "check", Mock(return_value=(policy, [{"i": i} for i in range(2)], records)))
    monkeypatch.setattr(collector.smoke, "_count", Mock(side_effect=counts))
    return manifest, candidates, smoke_receipt, tmp_path / "output", gate


def test_full_count_writes_bound_report(tmp_path: Path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch, [101, 202])
    receipt = collector.execute(*paths, key="hidden")
    assert receipt["complete"] is True and receipt["requests_attempted"] == 2
    assert receipt["successful_counts"] == 2 and receipt["total_count"] == 303
    report = json.loads((paths[3] / "token-preflight.json").read_text())
    assert report["method"] == "provider_count_tokens"
    assert [row["input_tokens"] for row in report["records"]] == [101, 202]
    assert receipt["token_preflight_sha256"] == collector.runner.digest(paths[3] / "token-preflight.json")
    assert "hidden" not in (paths[3] / "results.jsonl").read_text()


def test_uncertain_count_stops_without_retry(tmp_path: Path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch, [TimeoutError("unknown"), 202])
    receipt = collector.execute(*paths, key="hidden")
    assert receipt["complete"] is False and receipt["requests_attempted"] == 1
    assert receipt["stop_reason"] == "uncertain_count_outcome"
    assert not (paths[3] / "token-preflight.json").exists()
    assert collector.smoke._count.call_count == 1
    with pytest.raises(ValueError, match="output_or_credential"):
        collector.execute(*paths, key="hidden")
