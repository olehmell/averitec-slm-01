"""Composite assembly keeps all planned slots and fails closed on source drift."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import assemble_packages as assembly  # noqa: E402
import seal_selection_choices  # noqa: E402
import verdict_runner  # noqa: E402


def _candidates() -> dict:
    return {"gold_included": False, "gold_verdict": "must not enter packages", "cases": [
        {"case_id": f"case-{case}", "group_id": f"group-{case}", "claim": f"Claim {case}",
         "candidates": [{"id": f"case-{case}:C{rank:02d}", "passage_id": f"p-{case}-{rank}",
                         "text": f"Snippet {case}-{rank}", "url": "https://example.test/source",
                         "source_start": rank, "source_text_length": 100,
                         "source_text_sha256": "a" * 64, "reference_grade": 2}
                        for rank in range(10)]}
        for case in (1, 2)]}


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _spec(path: Path) -> dict:
    return {"path": str(path), "sha256": assembly._sha256(path)}


def _fixture(tmp_path: Path) -> tuple[Path, list[dict]]:
    candidates = _write(tmp_path / "candidates.json", _candidates())
    entries = []
    for number in assembly.PASSES:
        for arm in assembly.ARMS:
            bundle = tmp_path / ("native-v15" if arm == "jeff" else
                                 "native-v13" if arm == "laya_typed" else
                                 "gpu-v12" if arm in {"qwen", "lfm", "lfm26"} else
                                 "hosted-v1")
            launch = _write(bundle / "launch.json", {"bundle": bundle.name,
                            "source_commit": "0" * 40, "code_sha256": {"code.py": "1" * 64}})
            gate = _write(bundle / "gate.json", {"bundle": bundle.name})
            source = _write(bundle / "source-receipt.json",
                            {"gold_staged": False, "manifest_sha256": assembly._sha256(launch),
                             "gate_sha256": assembly._sha256(gate),
                             "source_commit": "0" * 40, "source_sha256": {"code.py": "1" * 64},
                             "inputs_sha256": {"candidates.json": assembly._sha256(candidates)}})
            directory = bundle / "outputs" / arm / f"pass-{number}"
            directory.mkdir(parents=True)
            _write(directory / "run-metadata.json", {"arm": arm, "pass": number,
                                                     "qualification": "real_runtime"})
            _write(directory / "receipt.json", {"qualification": "real_runtime", "complete": True})
            for name in ("preflight.json", "intents.jsonl", "traces.jsonl"):
                (directory / name).write_text("", encoding="utf-8")
            with (directory / "results.jsonl").open("w", encoding="utf-8") as stream:
                for case in (1, 2):
                    for rank in range(10):
                        if arm == "jev" and number == 2 and case == 1 and rank == 0:
                            continue
                        row = {"phase": "measured", "case_id": f"case-{case}",
                               "id": f"case-{case}:C{rank:02d}", "outcome": "ok",
                               "action": ("invalid" if arm == "laya_typed" and number == 3
                                          and case == 2 and rank == 0 else
                                          "include" if rank == 1 else "exclude")}
                        stream.write(json.dumps(row) + "\n")
            if arm in {"jev", "gemini"}:
                ledger = _write(bundle / "accounting" / f"{arm}-{number}.ledger.json",
                                {"arm": arm, "pass": number, "settled": True})
                accounting_body = {"schema": "three-pass-selector-accounting/v1",
                                   "arm": arm, "pass": number, "kind": "hosted", "ledger": _spec(ledger)}
            else:
                job_id = str(100000 + number * 100 + assembly.ARMS.index(arm))
                accepted = _write(bundle / "submissions" / f"{arm}-{number}.accepted.json",
                                  {"schema": "averitec-three-pass-selector-submit-accepted/v1",
                                   "arm": arm, "pass": number, "job_id": job_id,
                                   "manifest_sha256": assembly._sha256(launch),
                                   "gate_sha256": assembly._sha256(gate)})
                sacct = bundle / "accounting" / f"{arm}-{number}.sacct"
                sacct.parent.mkdir(parents=True, exist_ok=True)
                sacct.write_text(f"{job_id}|COMPLETED|100|billing=100,gres/gpu=1\n", encoding="utf-8")
                accounting_body = {"schema": "three-pass-selector-accounting/v1",
                                   "arm": arm, "pass": number, "kind": "slurm",
                                   "accepted": _spec(accepted), "sacct": _spec(sacct)}
            accounting = _write(bundle / "accounting" / f"{arm}-{number}.json", accounting_body)
            entries.append({"arm": arm, "pass": number, "source": _spec(source),
                            "candidates": _spec(candidates), "selector_launch": _spec(launch),
                            "selector_gate": _spec(gate),
                            "selector_output": {"path": str(directory),
                                                "files": {name: assembly._sha256(directory / name)
                                                          for name in assembly.JOURNALS}},
                            "accounting": {"kind": "hosted" if arm in {"jev", "gemini"} else "slurm",
                                           "evidence": _spec(accounting)}})
    manifest = _write(tmp_path / "composite.json",
                      {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    return manifest, entries


def _unavailable_slot(tmp_path: Path, entries: list[dict], *, arm: str = "jev", number: int = 2) -> tuple[dict, Path]:
    entry = next(row for row in entries if row["arm"] == arm and row["pass"] == number)
    root = tmp_path / "no-run" / f"{arm}-{number}"
    gate = _write(root / "astra-no-go.json",
                  {"schema": "three-pass-selector-no-go/v1", "decision": "no_go",
                   "reviewer": "gpt-6-astra", "manifest_sha256": entry["selector_launch"]["sha256"],
                   "allowed_arms": [arm], "allowed_passes": [number], "reason": "provider_access_denied"})
    launch = json.loads(Path(entry["selector_launch"]["path"]).read_text(encoding="utf-8"))
    source = _write(root / "source-receipt.json",
                    {"gold_staged": False, "manifest_sha256": entry["selector_launch"]["sha256"],
                     "gate_sha256": assembly._sha256(gate), "source_commit": launch["source_commit"],
                     "source_sha256": launch["code_sha256"], "model_calls": 0, "submitted_jobs": 0,
                     "inputs_sha256": {"candidates.json": entry["candidates"]["sha256"]}})
    ledger = root / "hosted-ledger.sqlite3"
    with sqlite3.connect(ledger) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema", json.dumps("averitec-three-pass-hosted-budget/v1")),
            ("experiment_id", json.dumps("averitec-controller-three-pass-verdict-2026"))])
        connection.execute("CREATE TABLE calls (request_id TEXT, arm TEXT, block TEXT)")
    audit = _write(root / "ledger-audit.json",
                   {"schema": "three-pass-hosted-slot-audit/v1", "arm": arm, "pass": number,
                    "ledger_sha256": assembly._sha256(ledger), "selector_calls": 0,
                    "reviewer": "gpt-6-astra"})
    accounting = _write(root / "accounting.json",
                        {"schema": "three-pass-selector-no-run-accounting/v1",
                         "arm": arm, "pass": number, "kind": "hosted",
                         "ledger": _spec(ledger), "audit": _spec(audit)})
    unavailable = _write(root / "unavailable.json",
                         {"schema": "three-pass-selector-unavailable/v1", "arm": arm, "pass": number,
                          "reason": "provider_access_denied", "source_commit": launch["source_commit"],
                          "launch_sha256": entry["selector_launch"]["sha256"],
                          "candidates_sha256": entry["candidates"]["sha256"],
                          "astra_no_go_sha256": assembly._sha256(gate), "model_calls": 0,
                          "dispatch_claim": None, "ledger_audit_sha256": assembly._sha256(audit)})
    entry.clear()
    entry.update({"arm": arm, "pass": number, "status": "unavailable",
                  "source": _spec(source), "candidates": _spec(tmp_path / "candidates.json"),
                  "selector_launch": _spec(tmp_path / "hosted-v1" / "launch.json"),
                  "astra_no_go": _spec(gate), "unavailable": _spec(unavailable),
                  "accounting": {"kind": "hosted", "evidence": _spec(accounting)}})
    return entry, ledger


def test_mixed_sources_preserve_failures_and_gold_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, entries = _fixture(tmp_path)
    seen = []

    def verify(directory: Path, candidates: Path, launch: Path, gate: Path, *, frozen_source: bool) -> dict:
        assert frozen_source
        seen.append((directory, launch, gate))
        return {"qualification": "real_runtime", "complete": True}

    monkeypatch.setattr(assembly.selector_passes, "verify", verify)
    value = assembly.assemble(manifest, expected_cases=2)
    assert len(seen) == 21
    assert len(value["selector_receipts"]) == 21
    assert len(value["packages"]) == 54
    assert (value["planned_slots"], value["runnable_packages"], value["failed_slots"]) == (54, 52, 2)
    assert value["failure_reasons"] == {"invalid_action": 1, "missing_candidate_id": 1}
    assert len({row["selector_launch_sha256"] for row in value["selector_receipts"]}) == 4
    failed = [p for p in value["packages"] if p["arm"] == "jev" and p["pass"] == 2 and p["case_id"] == "case-1"]
    assert len(failed) == 1 and failed[0]["status"] == "failed"
    assert failed[0]["selected_passages"] is None
    assert "missing_candidate_id:case-1:C00" in failed[0]["errors"]
    invalid = [p for p in value["packages"] if p["arm"] == "laya_typed" and p["pass"] == 3 and p["case_id"] == "case-2"]
    assert len(invalid) == 1 and invalid[0]["status"] == "failed"
    assert "invalid_action:case-2:C00" in invalid[0]["errors"]
    assert "gold_verdict" not in json.dumps(value) and "reference_grade" not in json.dumps(value)
    output = tmp_path / "packages.json"
    assembly.seal(output, value)
    candidates = Path(entries[0]["candidates"]["path"])
    assert len(verdict_runner._load_packages(candidates, output,
               (*verdict_runner.SELECTORS, *verdict_runner.CONTROLS), (1, 2, 3), 2)) == 54
    with pytest.raises(FileExistsError):
        assembly.seal(output, value)


def test_manifest_rejects_duplicate_slot_and_hash_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, entries = _fixture(tmp_path)
    monkeypatch.setattr(assembly.selector_passes, "verify",
                        lambda *_args, **_kwargs: {"qualification": "real_runtime", "complete": True})
    entries[1]["arm"], entries[1]["pass"] = entries[0]["arm"], entries[0]["pass"]
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    with pytest.raises(ValueError, match="duplicate_or_unknown_slot"):
        assembly.assemble(manifest, expected_cases=2)
    entries[1]["arm"], entries[1]["pass"] = assembly.ARMS[1], 1
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    Path(entries[1]["accounting"]["evidence"]["path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="composite_file_drift"):
        assembly.assemble(manifest, expected_cases=2)


def test_mixed_candidate_bytes_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, entries = _fixture(tmp_path)
    monkeypatch.setattr(assembly.selector_passes, "verify",
                        lambda *_args, **_kwargs: {"qualification": "real_runtime", "complete": True})
    altered = _candidates()
    altered["cases"][0]["claim"] = "Changed claim"
    entries[1]["candidates"] = _spec(_write(tmp_path / "different-candidates.json", altered))
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    with pytest.raises(ValueError, match="composite_candidate_drift"):
        assembly.assemble(manifest, expected_cases=2)


def test_source_receipt_must_bind_frozen_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, entries = _fixture(tmp_path)
    monkeypatch.setattr(assembly.selector_passes, "verify",
                        lambda *_args, **_kwargs: {"qualification": "real_runtime", "complete": True})
    source = Path(entries[0]["source"]["path"])
    altered = json.loads(source.read_text(encoding="utf-8"))
    altered["source_sha256"]["code.py"] = "2" * 64
    _write(source, altered)
    entries[0]["source"] = _spec(source)
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    with pytest.raises(ValueError, match="composite_source_binding"):
        assembly.assemble(manifest, expected_cases=2)


def test_sealed_choices_preserve_partial_pass_for_later_evaluation(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, _ = _fixture(tmp_path)
    monkeypatch.setattr(assembly.selector_passes, "verify",
                        lambda *_args, **_kwargs: {"qualification": "real_runtime", "complete": False})
    value = seal_selection_choices.seal_choices(manifest, expected_cases=2)
    assert value["sealed"] is True and len(value["source_receipts"]) == 21
    assert len(value["rows"]) == 21 * 20 - 1
    assert len([row for row in value["rows"] if row["arm"] == "jev" and row["pass"] == 2]) == 19
    assert "gold_verdict" not in json.dumps(value) and "reference_grade" not in json.dumps(value)
    output = tmp_path / "sealed-choices.json"
    seal_selection_choices.write_once(output, value)
    with pytest.raises(FileExistsError):
        seal_selection_choices.write_once(output, value)


def test_slurm_job_id_must_match_accepted_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, entries = _fixture(tmp_path)
    monkeypatch.setattr(assembly.selector_passes, "verify",
                        lambda *_args, **_kwargs: {"qualification": "real_runtime", "complete": True})
    entry = next(row for row in entries if row["arm"] == "qwen" and row["pass"] == 1)
    evidence_path = Path(entry["accounting"]["evidence"]["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    sacct_path = Path(evidence["sacct"]["path"])
    sacct_path.write_text("999999|COMPLETED|100|billing=100,gres/gpu=1\n", encoding="utf-8")
    evidence["sacct"] = _spec(sacct_path)
    _write(evidence_path, evidence)
    entry["accounting"]["evidence"] = _spec(evidence_path)
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    with pytest.raises(ValueError, match="composite_sacct_binding"):
        assembly.assemble(manifest, expected_cases=2)


def test_explicit_no_run_slot_keeps_denominator_without_choices(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, entries = _fixture(tmp_path)
    _unavailable_slot(tmp_path, entries)
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    seen = []

    def verify(directory: Path, *_args, **_kwargs) -> dict:
        seen.append(directory)
        return {"qualification": "real_runtime", "complete": True}

    monkeypatch.setattr(assembly.selector_passes, "verify", verify)
    value = assembly.assemble(manifest, expected_cases=2)
    assert len(seen) == 20
    assert (value["planned_slots"], value["runnable_packages"], value["failed_slots"]) == (54, 51, 3)
    assert value["failure_reasons"]["selector_unavailable"] == 2
    no_run = [p for p in value["packages"] if p["arm"] == "jev" and p["pass"] == 2]
    assert len(no_run) == 2 and all(p["selected_passages"] is None for p in no_run)
    assert all(p["errors"] == ["selector_unavailable:provider_access_denied"] for p in no_run)
    source = next(r for r in value["selector_receipts"] if r["arm"] == "jev" and r["pass"] == 2)
    assert source["status"] == "unavailable" and source["model_calls"] == 0
    assert source["dispatch_claim"] is None and source["measured_results"] == 0
    sealed = seal_selection_choices.seal_choices(manifest, expected_cases=2)
    assert len(sealed["source_receipts"]) == 21 and len(sealed["rows"]) == 400
    assert not [r for r in sealed["rows"] if r["arm"] == "jev" and r["pass"] == 2]


def test_no_run_rejects_output_field_and_ledger_call(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, entries = _fixture(tmp_path)
    slot, ledger = _unavailable_slot(tmp_path, entries)
    monkeypatch.setattr(assembly.selector_passes, "verify",
                        lambda *_args, **_kwargs: {"qualification": "real_runtime", "complete": True})
    slot["selector_output"] = {"path": "/fake", "files": {}}
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    with pytest.raises(ValueError, match="composite_entry_shape"):
        assembly.assemble(manifest, expected_cases=2)
    del slot["selector_output"]
    with sqlite3.connect(ledger) as connection:
        connection.execute("INSERT INTO calls VALUES (?,?,?)", ("selector/jev/pass-2/ordinal-1", "jev", "selector"))
    account_path = Path(slot["accounting"]["evidence"]["path"])
    account = json.loads(account_path.read_text(encoding="utf-8"))
    ledger_spec = _spec(ledger)
    account["ledger"] = ledger_spec
    audit_path = Path(account["audit"]["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["ledger_sha256"] = ledger_spec["sha256"]
    _write(audit_path, audit)
    account["audit"] = _spec(audit_path)
    _write(account_path, account)
    slot["accounting"]["evidence"] = _spec(account_path)
    unavailable_path = Path(slot["unavailable"]["path"])
    unavailable = json.loads(unavailable_path.read_text(encoding="utf-8"))
    unavailable["ledger_audit_sha256"] = account["audit"]["sha256"]
    _write(unavailable_path, unavailable)
    slot["unavailable"] = _spec(unavailable_path)
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    with pytest.raises(ValueError, match="composite_unavailable_ledger_has_calls"):
        assembly.assemble(manifest, expected_cases=2)


@pytest.mark.parametrize("field,value", [("model_calls", 1), ("dispatch_claim", "claimed")])
def test_no_run_requires_zero_calls_and_no_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      field: str, value: object) -> None:
    manifest, entries = _fixture(tmp_path)
    slot, _ = _unavailable_slot(tmp_path, entries)
    monkeypatch.setattr(assembly.selector_passes, "verify",
                        lambda *_args, **_kwargs: {"qualification": "real_runtime", "complete": True})
    path = Path(slot["unavailable"]["path"])
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence[field] = value
    _write(path, evidence)
    slot["unavailable"] = _spec(path)
    _write(manifest, {"schema": assembly.MANIFEST_SCHEMA, "entries": entries})
    with pytest.raises(ValueError, match="composite_unavailable_binding"):
        assembly.assemble(manifest, expected_cases=2)
