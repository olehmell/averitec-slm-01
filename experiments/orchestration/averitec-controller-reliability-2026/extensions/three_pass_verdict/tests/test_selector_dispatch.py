from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import selector_dispatch as dispatch  # noqa: E402


def test_21_unique_claims_and_manifest_output_binding(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"schema": "averitec-three-pass-selector-launch/v1",
                                  "arms": {arm: {} for arm in dispatch.ARMS},
                                  "passes": [1, 2, 3], "total_measured_calls": 21000,
                                  "source_commit": "a" * 40}), encoding="utf-8")
    ledger = tmp_path / "ledger.sqlite3"
    dispatch.initialize(ledger, launch, strict_path=False)
    claims = tmp_path / "claims"
    claims.mkdir()
    for arm in sorted(dispatch.ARMS):
        for number in (1, 2, 3):
            output = tmp_path / "outputs" / f"{arm}-pass-{number}"
            receipt = claims / f"{arm}-{number}.json"
            dispatch.claim(ledger, launch, arm, number, str(output), receipt, strict_path=False)
            assert dispatch.verify_registered_claim(receipt, launch, arm, number, output, strict_path=False)["arm"] == arm
    with pytest.raises(Exception):
        dispatch.claim(ledger, launch, "jev", 1, str(tmp_path / "other" / "jev-pass-1"),
                       claims / "duplicate.json", strict_path=False)
    tampered = claims / "tampered.json"
    value = json.loads((claims / "jev-1.json").read_text())
    value["output"] = str(tmp_path / "elsewhere" / "jev-pass-1")
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered_claim"):
        dispatch.verify_registered_claim(tampered, launch, "jev", 1, Path(value["output"]), strict_path=False)


def test_claim_cannot_bind_wrong_manifest_or_existing_output(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"schema": "averitec-three-pass-selector-launch/v1",
                                  "arms": {arm: {} for arm in dispatch.ARMS},
                                  "passes": [1, 2, 3], "total_measured_calls": 21000,
                                  "source_commit": "a" * 40}), encoding="utf-8")
    ledger = tmp_path / "ledger.sqlite3"
    dispatch.initialize(ledger, launch, strict_path=False)
    receipt = tmp_path / "claim.json"
    output = tmp_path / "jev-pass-1"
    dispatch.claim(ledger, launch, "jev", 1, str(output), receipt, strict_path=False)
    with pytest.raises(ValueError, match="receipt_exists"):
        dispatch.claim(ledger, launch, "jev", 1, str(output), receipt, strict_path=False)
    launch.write_text(launch.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="claim_binding"):
        dispatch.verify_claim(receipt, launch, "jev", 1, output)
