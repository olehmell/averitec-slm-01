from __future__ import annotations

import json
from pathlib import Path
import signal
import sys
import time

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemini_generation_smoke as smoke  # noqa: E402


def test_transport_checks_exact_request_before_network_and_restores():
    original = smoke.runner.recovery_module._post
    policy = {"request_sha256": "0" * 64, "request_timeout_seconds": 120,
              "maximum_wall_seconds": 2}
    with pytest.raises(ValueError, match="request_drift"):
        with smoke._single_request(policy):
            smoke.runner.recovery_module._post(
                smoke.counter.URL.replace(":countTokens", ":generateContent"),
                {"contents": []}, {"x-goog-api-key": "hidden"}, 120)
    assert smoke.runner.recovery_module._post is original
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_wall_deadline_is_hard_and_restores_transport():
    original = smoke.runner.recovery_module._post
    with pytest.raises(TimeoutError, match="wall_deadline"):
        with smoke._single_request({"maximum_wall_seconds": 0.05}):
            time.sleep(0.2)
    assert smoke.runner.recovery_module._post is original
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_count_receipt_requires_matching_successful_result(tmp_path: Path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"status": "ok", "total_tokens": 279}), encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"schema": "averitec-gemini-count-smoke-receipt/v1",
                                        "complete": True, "requests_attempted": 1,
                                        "result_sha256": smoke.runner.digest(result_path)}), encoding="utf-8")
    assert smoke._count_receipt(receipt_path)[1]["total_tokens"] == 279
    result_path.write_text(json.dumps({"status": "ok", "total_tokens": 280}), encoding="utf-8")
    with pytest.raises(ValueError, match="count_smoke_invalid"):
        smoke._count_receipt(receipt_path)
