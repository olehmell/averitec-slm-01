"""Seal evaluator-only selector choices from the audited composite sources.

The output is kept outside the HerO inference bundle. It contains no relevance
grades; those are joined later by evaluate_selection.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import assemble_packages


def seal_choices(manifest_path: Path, *, expected_cases: int = 100) -> dict:
    audited = assemble_packages.assemble(manifest_path, expected_cases=expected_cases)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["entries"]:
        if entry.get("status") == "unavailable":
            continue
        source = Path(entry["selector_output"]["path"]) / "results.jsonl"
        rows.extend({"arm": entry["arm"], "pass": entry["pass"], **row}
                    for row in assemble_packages._measured(source))
    if len(rows) > len(assemble_packages.ARMS) * len(assemble_packages.PASSES) * expected_cases * 10:
        raise ValueError("sealed_selection_exceeds_plan")
    return {"schema": "three-pass-sealed-selection-choices/v1", "sealed": True,
            "composite_manifest_sha256": audited["composite_manifest_sha256"],
            "source_receipts": audited["selector_receipts"], "rows": rows}


def write_once(output: Path, value: dict) -> str:
    body = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(body).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = seal_choices(args.composite_manifest)
    print(json.dumps({"schema": value["schema"], "output_sha256": write_once(args.output, value)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
