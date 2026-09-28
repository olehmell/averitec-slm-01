"""Append-only, manifest-bound case-output checkpointing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager
import fcntl
from typing import Any, Iterable


INFLIGHT_SCHEMA = "averitec-inference-inflight/v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("atomic_json_write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@contextmanager
def run_lock(output: Path):
    """Hold an exclusive non-blocking lock for one public run journal."""
    path = output.with_suffix(output.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("run_lock_held") from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _inflight_path(output: Path, case_id: str, condition: str) -> Path:
    key = hashlib.sha256((case_id + "\0" + condition).encode("utf-8")).hexdigest()
    return output.with_suffix(output.suffix + ".inflight") / (key + ".json")


def begin_pair(output: Path, *, manifest: dict[str, Any], case_id: str, condition: str) -> None:
    """Persist intent before a case-condition can dispatch its first model call."""
    if case_id not in manifest["case_ids"] or condition not in manifest["conditions"]:
        raise ValueError("inflight_pair_outside_manifest")
    path = _inflight_path(output, case_id, condition)
    if path.exists():
        raise ValueError("inflight_pair_already_claimed")
    _atomic_json(path, {
        "schema": INFLIGHT_SCHEMA,
        "run_sha256": manifest["sha256"],
        "case_id": case_id,
        "condition": condition,
        "state": "claimed",
    })


def finish_pair(output: Path, *, manifest: dict[str, Any], record: dict[str, Any]) -> None:
    case_id, condition = record.get("case_id"), record.get("condition")
    path = _inflight_path(output, case_id, condition)
    if not path.is_file():
        raise ValueError("inflight_pair_missing")
    _atomic_json(path, {
        "schema": INFLIGHT_SCHEMA,
        "run_sha256": manifest["sha256"],
        "case_id": case_id,
        "condition": condition,
        "state": "finalized",
        "record_sha256": _digest(record),
    })


def inflight_pairs(output: Path, *, manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    root = output.with_suffix(output.suffix + ".inflight")
    if not root.exists():
        return {}
    allowed_cases, allowed_conditions = set(manifest["case_ids"]), set(manifest["conditions"])
    values: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("inflight_record_invalid") from error
        case_id, condition = value.get("case_id"), value.get("condition")
        key = (case_id, condition)
        if (value.get("schema") != INFLIGHT_SCHEMA
                or value.get("run_sha256") != manifest["sha256"]
                or case_id not in allowed_cases or condition not in allowed_conditions
                or value.get("state") not in {"claimed", "finalized"}
                or key in values
                or path != _inflight_path(output, case_id, condition)):
            raise ValueError("inflight_record_incompatible")
        if value["state"] == "finalized" and not isinstance(value.get("record_sha256"), str):
            raise ValueError("inflight_record_invalid")
        values[key] = value
    return values


def bind_run(output: Path, *, cases: Iterable[dict[str, str]], conditions: list[str], inputs: dict[str, str]) -> dict[str, Any]:
    manifest = {"schema": "averitec-inference-run/v1", "case_ids": [case["case_id"] for case in cases], "conditions": conditions, "inputs": inputs}
    manifest["sha256"] = _digest(manifest)
    path = output.with_suffix(output.suffix + ".run.json")
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != manifest:
            raise ValueError("incompatible_resume")
    else:
        _atomic_json(path, manifest)
    return manifest


def output_records(output: Path, *, manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if not output.exists():
        return {}
    allowed_cases = set(manifest["case_ids"])
    allowed_conditions = set(manifest["conditions"])
    found: dict[tuple[str, str], dict[str, Any]] = {}
    raw = output.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("partial_final_output_line")
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
            key = (row["case_id"], row["condition"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid_resume_output:{number}") from error
        if row.get("run_sha256") != manifest["sha256"] or not isinstance(row.get("status"), str) or not isinstance(row.get("prediction"), dict):
            raise ValueError("incompatible_resume")
        if key[0] not in allowed_cases or key[1] not in allowed_conditions or key in found:
            raise ValueError("incompatible_resume")
        found[key] = row
    return found


def completed(output: Path, *, manifest: dict[str, Any]) -> set[tuple[str, str]]:
    return set(output_records(output, manifest=manifest))


def append(output: Path, record: dict[str, Any]) -> None:
    with output.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
