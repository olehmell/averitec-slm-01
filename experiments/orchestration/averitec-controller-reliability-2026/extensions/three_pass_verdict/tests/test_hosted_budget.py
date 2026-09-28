from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hosted_budget as budget  # noqa: E402


def _ledger(tmp_path: Path) -> Path:
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"schema": budget.SCHEMA, "experiment_id": budget.EXPERIMENT_ID,
                                "calls": [], "count_tokens_requests": {"jev": 0, "gemini": 1001}}),
                    encoding="utf-8")
    path = tmp_path / "ledger.sqlite3"
    result = budget.initialize(path, seed, strict_path=False)
    assert result["count_tokens_requests"]["gemini"] == 1001
    return path


def _reserve(path: Path, request_id: str, arm: str = "jev", **changes: int) -> None:
    values = {"input_tokens": 2459, "output_tokens": 1024, "wall_seconds": 30}
    values.update(changes)
    budget.reserve(path, request_id=request_id, arm=arm, block="selector",
                   manifest_sha256="a" * 64, request_sha256="b" * 64,
                   strict_path=False, **values)


def test_reserve_is_durable_and_settled_usage_releases_only_confirmed_difference(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    with pytest.raises(FileExistsError):
        budget.initialize(path, tmp_path / "seed.json", strict_path=False)
    _reserve(path, "selector/jev/1/1")
    with pytest.raises(budget.BudgetStop, match="pending"):
        _reserve(path, "selector/gemini/1/1", arm="gemini")
    assert budget.snapshot(path, strict_path=False)["by_arm"]["jev"]["output_tokens"] == 1024
    budget.settle(path, "selector/jev/1/1", input_tokens=555, output_tokens=32,
                  elapsed_seconds=2.1, strict_path=False)
    state = budget.snapshot(path, strict_path=False)
    assert state["pending_request_ids"] == []
    assert state["by_arm"]["jev"] == {"requests": 1, "input_tokens": 555,
                                       "output_tokens": 32, "wall_seconds": 3}
    _reserve(path, "selector/gemini/1/1", arm="gemini")
    assert budget.snapshot(path, strict_path=False)["pending_request_ids"] == ["selector/gemini/1/1"]


def test_duplicate_and_unconfirmed_usage_leave_reservation_pending(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    _reserve(path, "selector/jev/1/1")
    with pytest.raises(budget.BudgetStop, match="unconfirmed"):
        budget.settle(path, "selector/jev/1/1", input_tokens=2460, output_tokens=32,
                      elapsed_seconds=1, strict_path=False)
    assert budget.snapshot(path, strict_path=False)["pending_request_ids"] == ["selector/jev/1/1"]
    budget.settle(path, "selector/jev/1/1", input_tokens=555, output_tokens=32,
                  elapsed_seconds=1, strict_path=False)
    with pytest.raises(Exception):
        _reserve(path, "selector/jev/1/1")


def test_global_cap_checked_before_dispatch(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    with pytest.raises(budget.BudgetStop, match="global_cap_before_request"):
        _reserve(path, "selector/gemini/1/1", arm="gemini", output_tokens=200001)
    # A cap rejection must not insert a pending request.
    assert budget.snapshot(path, strict_path=False)["by_arm"]["gemini"]["requests"] == 0
    assert budget.snapshot(path, strict_path=False)["pending_request_ids"] == []


def test_recovery_config_binding_preserves_existing_rows_and_rejects_other_hashes(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    _reserve(path, "selector/jev/1/1")
    budget.settle(path, "selector/jev/1/1", input_tokens=555, output_tokens=32,
                  elapsed_seconds=2, strict_path=False)
    assert budget.sha256(budget.CONFIG) == budget.PUBLIC_SUPPORT_RELOCATION_CONFIG_SHA256
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE metadata SET value=? WHERE key='config_sha256'",
                           (json.dumps(budget.PRE_RECOVERY_CONFIG_SHA256),))
    state = budget.snapshot(path, strict_path=False)
    assert state["by_arm"]["jev"] == {"requests": 1, "input_tokens": 555,
                                      "output_tokens": 32, "wall_seconds": 2}
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE metadata SET value=? WHERE key='config_sha256'",
                           (json.dumps("0" * 64),))
    with pytest.raises(ValueError, match="hosted_budget_identity"):
        budget.snapshot(path, strict_path=False)
