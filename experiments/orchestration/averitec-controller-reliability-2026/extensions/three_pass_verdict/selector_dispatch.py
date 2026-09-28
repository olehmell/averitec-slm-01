"""One-shot, manifest-bound selector dispatch claims; no model or job calls.

The coordinator runs on the submitting host. Claims are consumed before a
hosted call or Slurm submission, and exported into the immutable GPU bundle.
An uncertain submission keeps its claim: there is deliberately no reset API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

SCHEMA = "averitec-three-pass-selector-dispatch/v1"
CLAIM_SCHEMA = "averitec-three-pass-selector-claim/v1"
ARMS = frozenset({"jev", "lfm", "laya_typed", "gemini", "qwen", "lfm26", "jeff"})
CANONICAL_LEDGER = Path(".private-artifacts/averitec-controller-reliability-2026/three-pass-verdict-2026/selector-dispatch-v14.sqlite3").resolve()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA busy_timeout=10000")
    return db


def initialize(path: Path, manifest: Path, *, strict_path: bool = True) -> None:
    """Create the sole ledger before any bundle is staged or call dispatched."""
    if strict_path and path.resolve() != CANONICAL_LEDGER:
        raise ValueError("selector_dispatch_canonical_ledger_required")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if (value.get("schema") != "averitec-three-pass-selector-launch/v1"
            or set(value.get("arms", {})) != ARMS or value.get("passes") != [1, 2, 3]
            or value.get("total_measured_calls") != 21000):
        raise ValueError("selector_dispatch_manifest_shape")
    if not path.parent.is_dir():
        raise ValueError("selector_dispatch_parent_missing")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        db = _connect(path)
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE metadata (manifest_sha256 TEXT NOT NULL, source_commit TEXT NOT NULL)")
            db.execute("INSERT INTO metadata VALUES (?,?)", (sha(manifest), value["source_commit"]))
            db.execute("CREATE TABLE claims (arm TEXT NOT NULL, pass_number INTEGER NOT NULL, output TEXT NOT NULL, "
                       "claim_sha256 TEXT NOT NULL, PRIMARY KEY (arm,pass_number), UNIQUE (output))")
            db.execute("COMMIT")
        finally:
            db.close()
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def claim(path: Path, manifest: Path, arm: str, pass_number: int, output: str, receipt: Path,
          *, strict_path: bool = True) -> dict:
    """Atomically spend one of the exact 21 arm/pass slots, before dispatch."""
    if strict_path and path.resolve() != CANONICAL_LEDGER:
        raise ValueError("selector_dispatch_canonical_ledger_required")
    if arm not in ARMS or type(pass_number) is not int or pass_number not in (1, 2, 3):
        raise ValueError("selector_dispatch_arm_pass")
    if (not output or not output.startswith("/") or "\n" in output or "\r" in output
            or Path(output).name != f"{arm}-pass-{pass_number}" or ".." in Path(output).parts):
        raise ValueError("selector_dispatch_output")
    if receipt.exists() or not receipt.parent.is_dir():
        raise ValueError("selector_dispatch_receipt_exists")
    manifest_hash = sha(manifest)
    row = {"schema": CLAIM_SCHEMA, "manifest_sha256": manifest_hash, "arm": arm,
           "pass": pass_number, "output": output, "ledger": str(path.resolve())}
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    claim_hash = hashlib.sha256(data).hexdigest()
    db = _connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        identity = db.execute("SELECT manifest_sha256 FROM metadata").fetchall()
        if identity != [(manifest_hash,)]:
            raise ValueError("selector_dispatch_manifest_drift")
        if db.execute("SELECT 1 FROM claims WHERE arm=? AND pass_number=?", (arm, pass_number)).fetchone():
            raise ValueError("selector_dispatch_slot_already_claimed")
        if db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] >= 21:
            raise ValueError("selector_dispatch_full")
        # A receipt is written before commit. An interrupted transaction leaves
        # an orphan, which must be investigated, never silently reused.
        descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        db.execute("INSERT INTO claims VALUES (?,?,?,?)", (arm, pass_number, output, claim_hash))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return row


def verify_claim(receipt: Path, manifest: Path, arm: str, pass_number: int, output: Path) -> dict:
    row = json.loads(receipt.read_text(encoding="utf-8"))
    if row != {"schema": CLAIM_SCHEMA, "manifest_sha256": sha(manifest), "arm": arm,
               "pass": pass_number, "output": str(output), "ledger": row.get("ledger")}:
        raise ValueError("selector_dispatch_claim_binding")
    if not isinstance(row["ledger"], str) or not row["ledger"].startswith("/"):
        raise ValueError("selector_dispatch_ledger_identity")
    return row


def verify_registered_claim(receipt: Path, manifest: Path, arm: str, pass_number: int,
                            output: Path, *, strict_path: bool = True) -> dict:
    """Check the coordinating ledger while still on the submitting host."""
    row = verify_claim(receipt, manifest, arm, pass_number, output)
    if strict_path and Path(row["ledger"]).resolve() != CANONICAL_LEDGER:
        raise ValueError("selector_dispatch_canonical_ledger_required")
    if not Path(row["ledger"]).is_file():
        raise ValueError("selector_dispatch_ledger_missing")
    db = _connect(Path(row["ledger"]))
    try:
        identity = db.execute("SELECT manifest_sha256 FROM metadata").fetchall()
        registered = db.execute("SELECT output,claim_sha256 FROM claims WHERE arm=? AND pass_number=?",
                                (arm, pass_number)).fetchall()
        if identity != [(sha(manifest),)] or registered != [(str(output), sha(receipt))]:
            raise ValueError("selector_dispatch_unregistered_claim")
    finally:
        db.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--claim", action="store_true")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--pass-number", type=int, choices=(1, 2, 3))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.initialize:
        initialize(args.ledger, args.manifest)
        print(json.dumps({"schema": SCHEMA, "manifest_sha256": sha(args.manifest), "claims": 0}))
    else:
        if not all((args.arm, args.pass_number, args.output, args.receipt)):
            parser.error("claim needs --arm, --pass-number, --output, --receipt")
        print(json.dumps(claim(args.ledger, args.manifest, args.arm, args.pass_number,
                               str(args.output), args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
