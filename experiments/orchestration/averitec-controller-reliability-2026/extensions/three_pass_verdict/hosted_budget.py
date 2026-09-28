"""One durable, shared admission ledger for hosted workflow and selector calls.

Initialize once from reviewed smoke receipts. A reservation is committed before
HTTP dispatch; an uncertain call remains reserved and prevents later calls.
The database path is fixed in the main checkout so isolated worktrees cannot
silently create separate budgets. This module never sends a model request.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any


SCHEMA = "averitec-three-pass-hosted-budget/v1"
EXPERIMENT_ID = "averitec-controller-three-pass-verdict-2026"
LEDGER = Path(".private-artifacts/averitec-controller-reliability-2026/three-pass-verdict-2026/hosted-budget.sqlite3").resolve()
CONFIG = Path(__file__).resolve().parent / "config.yaml"
ARMS = ("jev", "gemini")
CAPS = {
    "jev": {"requests": 9910, "input_tokens": 9000000, "output_tokens": 1000000,
            "wall_seconds": 43200, "count_tokens_requests": 0},
    "gemini": {"requests": 9910, "input_tokens": 6000000, "output_tokens": 200000,
               "wall_seconds": 43200, "count_tokens_requests": 9910},
}
PRE_RECOVERY_CONFIG_SHA256 = "5eb91ab7ebd765b4f01ee5c1f036a707c3a7e6dfa3e19d357847dbc22c0a4bf7"
SELECTOR_RECOVERY_CONFIG_SHA256 = "d2e51e6da830e2f75db4889e81f97f1dc779b8ba509ad29d32d5c7c07316d3f0"
NATIVE_RECOVERY_CONFIG_SHA256 = "e92698a8a2f7e4214c1fb1985b7a8df17e2bde92e542f5983ff403bc95f52597"
JEFF_WARMUP_RECOVERY_CONFIG_SHA256 = "b4709f2c6b46574f0319b2ca64c76a3774b9ff28dddf998c34d7e2e0a0fe627e"
GPU_ACCOUNTING_AMENDMENT_CONFIG_SHA256 = "71e00ca90c6477ec1cde67496a846ba7fe31dead053eb6c0d40efa6728e1181a"
NATIVE_SMOKE_TRANSFER_CONFIG_SHA256 = "ac4cc7a049846b4ebb27bf33f0f9cf836fb395311b61fc84c024ff3d02986468"
PUBLIC_SUPPORT_RELOCATION_CONFIG_SHA256 = "a5e2edc8885277e54f776263e5eeccf580563e02a4092c8e7529a45900631eb1"


class BudgetStop(RuntimeError):
    """A reviewed global ceiling or pending request stops further dispatch."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=10000")
    return connection


def _totals(connection: sqlite3.Connection, arm: str) -> dict[str, int]:
    row = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='settled' THEN input_actual ELSE input_reserved END),0), "
        "COALESCE(SUM(CASE WHEN status='settled' THEN output_actual ELSE output_reserved END),0), "
        "COALESCE(SUM(CASE WHEN status='settled' THEN wall_actual ELSE wall_reserved END),0) "
        "FROM calls WHERE arm=?", (arm,)).fetchone()
    return dict(zip(("requests", "input_tokens", "output_tokens", "wall_seconds"), map(int, row)))


def _check_config() -> None:
    import yaml
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    hosted = config["resources"]["api_caps_per_hosted_arm"]
    expected = {arm: {"requests": hosted[arm]["maximum_generation_requests"],
                      "input_tokens": hosted[arm]["maximum_input_tokens"],
                      "output_tokens": hosted[arm]["maximum_output_tokens"],
                      "wall_seconds": hosted[arm]["maximum_wall_seconds"],
                      "count_tokens_requests": hosted[arm].get("maximum_count_tokens_requests", 0)}
                for arm in ARMS}
    if expected != CAPS:
        raise ValueError("hosted_budget_config_cap_drift")


def initialize(path: Path, seed_file: Path, *, strict_path: bool = True) -> dict[str, Any]:
    """Create once from a reviewed seed file; never reinitialize an existing DB."""
    if strict_path and path.resolve() != LEDGER:
        raise ValueError("hosted_budget_canonical_path")
    seed = json.loads(seed_file.read_text(encoding="utf-8"))
    if (not isinstance(seed, dict) or set(seed) != {"schema", "experiment_id", "calls", "count_tokens_requests"}
            or seed["schema"] != SCHEMA or seed["experiment_id"] != EXPERIMENT_ID
            or not isinstance(seed["calls"], list)
            or not isinstance(seed["count_tokens_requests"], dict)
            or set(seed["count_tokens_requests"]) != set(ARMS)):
        raise ValueError("hosted_budget_seed_shape")
    for arm in ARMS:
        amount = seed["count_tokens_requests"][arm]
        if type(amount) is not int or not 0 <= amount <= CAPS[arm]["count_tokens_requests"]:
            raise ValueError("hosted_budget_seed_count")
    if not path.parent.is_dir():
        raise ValueError("hosted_budget_parent_missing")
    _check_config()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(descriptor)
    try:
        connection = _connect(path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE calls (request_id TEXT PRIMARY KEY, arm TEXT NOT NULL, block TEXT NOT NULL, "
                               "manifest_sha256 TEXT NOT NULL, request_sha256 TEXT NOT NULL, status TEXT NOT NULL, "
                               "input_reserved INTEGER NOT NULL, output_reserved INTEGER NOT NULL, wall_reserved INTEGER NOT NULL, "
                               "input_actual INTEGER, output_actual INTEGER, wall_actual INTEGER, source_receipt_sha256 TEXT)")
            metadata = {"schema": SCHEMA, "experiment_id": EXPERIMENT_ID,
                        "seed_sha256": sha256(seed_file), "config_sha256": sha256(CONFIG), "caps": CAPS,
                        "count_tokens_requests": seed["count_tokens_requests"]}
            for key, value in metadata.items():
                connection.execute("INSERT INTO metadata VALUES (?,?)", (key, json.dumps(value, sort_keys=True)))
            for call in seed["calls"]:
                if (not isinstance(call, dict) or set(call) != {"request_id", "arm", "block", "manifest_sha256",
                        "request_sha256", "input_tokens", "output_tokens", "wall_seconds", "source_receipt_sha256"}
                        or call["arm"] not in ARMS or call["block"] != "smoke"
                        or any(type(call[key]) is not int or call[key] < 0 for key in
                               ("input_tokens", "output_tokens", "wall_seconds"))
                        or any(not isinstance(call[key], str) or not call[key] for key in
                               ("request_id", "manifest_sha256", "request_sha256", "source_receipt_sha256"))):
                    raise ValueError("hosted_budget_seed_call")
                connection.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                   (call["request_id"], call["arm"], "smoke", call["manifest_sha256"],
                                    call["request_sha256"], "settled", call["input_tokens"], call["output_tokens"],
                                    call["wall_seconds"], call["input_tokens"], call["output_tokens"],
                                    call["wall_seconds"], call["source_receipt_sha256"]))
            for arm in ARMS:
                if any(_totals(connection, arm)[key] > CAPS[arm][key]
                       for key in ("requests", "input_tokens", "output_tokens", "wall_seconds")):
                    raise ValueError("hosted_budget_seed_exceeds_cap")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return snapshot(path, strict_path=strict_path)


def _validate(connection: sqlite3.Connection) -> None:
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key,value FROM metadata")}
    _check_config()
    current_config_sha256 = sha256(CONFIG)
    # Exact GPU-accounting amendments retain the reviewed hosted cap and all
    # existing ledger rows; no other configuration drift is accepted.
    config_bound = metadata.get("config_sha256") == current_config_sha256 or (
        current_config_sha256 in {SELECTOR_RECOVERY_CONFIG_SHA256, NATIVE_RECOVERY_CONFIG_SHA256,
                                  JEFF_WARMUP_RECOVERY_CONFIG_SHA256,
                                  GPU_ACCOUNTING_AMENDMENT_CONFIG_SHA256,
                                  NATIVE_SMOKE_TRANSFER_CONFIG_SHA256,
                                  PUBLIC_SUPPORT_RELOCATION_CONFIG_SHA256}
        and metadata.get("config_sha256") == PRE_RECOVERY_CONFIG_SHA256
    )
    if (set(metadata) != {"schema", "experiment_id", "seed_sha256", "config_sha256", "caps", "count_tokens_requests"}
            or metadata["schema"] != SCHEMA or metadata["experiment_id"] != EXPERIMENT_ID
            or not config_bound or metadata["caps"] != CAPS):
        raise ValueError("hosted_budget_identity")


def reserve(path: Path, *, request_id: str, arm: str, block: str, manifest_sha256: str,
            request_sha256: str, input_tokens: int, output_tokens: int, wall_seconds: int,
            strict_path: bool = True) -> None:
    if strict_path and path.resolve() != LEDGER:
        raise ValueError("hosted_budget_canonical_path")
    if not path.is_file():
        raise ValueError("hosted_budget_not_initialized")
    if (arm not in ARMS or block not in {"workflow", "selector"} or not request_id
            or any(type(value) is not int or value < 1 for value in (input_tokens, output_tokens, wall_seconds))
            or any(not isinstance(value, str) or len(value) != 64 for value in (manifest_sha256, request_sha256))):
        raise ValueError("hosted_budget_request_shape")
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _validate(connection)
        if connection.execute("SELECT 1 FROM calls WHERE status='pending' LIMIT 1").fetchone():
            raise BudgetStop("hosted_budget_pending_unknown_or_inflight")
        totals = _totals(connection, arm)
        increments = {"requests": 1, "input_tokens": input_tokens,
                      "output_tokens": output_tokens, "wall_seconds": wall_seconds}
        if any(totals[key] + increments[key] > CAPS[arm][key] for key in increments):
            raise BudgetStop("hosted_budget_global_cap_before_request")
        connection.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (request_id, arm, block, manifest_sha256, request_sha256, "pending",
                            input_tokens, output_tokens, wall_seconds, None, None, None, None))
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def settle(path: Path, request_id: str, *, input_tokens: int, output_tokens: int,
           elapsed_seconds: float, strict_path: bool = True) -> None:
    if strict_path and path.resolve() != LEDGER:
        raise ValueError("hosted_budget_canonical_path")
    if not path.is_file():
        raise ValueError("hosted_budget_not_initialized")
    if (type(input_tokens) is not int or input_tokens < 0 or type(output_tokens) is not int or output_tokens < 0
            or not isinstance(elapsed_seconds, (int, float)) or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0):
        raise ValueError("hosted_budget_settlement_shape")
    wall = math.ceil(elapsed_seconds)
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _validate(connection)
        row = connection.execute("SELECT status,input_reserved,output_reserved,wall_reserved FROM calls WHERE request_id=?",
                                 (request_id,)).fetchone()
        if row is None or row[0] != "pending" or input_tokens > row[1] or output_tokens > row[2] or wall > row[3]:
            raise BudgetStop("hosted_budget_settlement_unconfirmed")
        connection.execute("UPDATE calls SET status='settled', input_actual=?, output_actual=?, wall_actual=? WHERE request_id=?",
                           (input_tokens, output_tokens, wall, request_id))
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def snapshot(path: Path, *, strict_path: bool = True) -> dict[str, Any]:
    if strict_path and path.resolve() != LEDGER:
        raise ValueError("hosted_budget_canonical_path")
    if not path.is_file():
        raise ValueError("hosted_budget_not_initialized")
    connection = _connect(path)
    try:
        _validate(connection)
        return {"schema": SCHEMA, "experiment_id": EXPERIMENT_ID,
                "seed_sha256": json.loads(connection.execute(
                    "SELECT value FROM metadata WHERE key='seed_sha256'").fetchone()[0]),
                "by_arm": {arm: _totals(connection, arm) for arm in ARMS},
                "pending_request_ids": [row[0] for row in connection.execute(
                    "SELECT request_id FROM calls WHERE status='pending' ORDER BY request_id")],
                "count_tokens_requests": json.loads(connection.execute(
                    "SELECT value FROM metadata WHERE key='count_tokens_requests'").fetchone()[0])}
    finally:
        connection.close()
