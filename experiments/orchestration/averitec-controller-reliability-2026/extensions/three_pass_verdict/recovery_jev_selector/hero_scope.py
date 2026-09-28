"""Build a gold-free, write-once HerO scope for the complete Jev recovery cohort.

The new cohort is recorded as pass 4. A downstream comparison may explicitly
substitute it for the incomplete original Jev pass 3, but raw IDs remain distinct.
This command does no model inference and never reads evaluator gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
THREE = HERE.parent
ROOT = HERE.parents[5]
import sys
sys.path.insert(0, str(THREE))
import verdict_runner as verdict  # noqa: E402

SCHEMA = "averitec-jev-recovery-hero-scope/v1"
SOURCE_CANDIDATES_SHA256 = "219c745aac6f50501a55c4f2511d383505317fd75338b0efaed25f44bbe23b98"
ORIGINAL_HERO_MANIFEST_SHA256 = "03f39cd827bd699b614e4927f83667fcb39a4bda20edb6eb69418cb529b74e5e"
ORIGINAL_HERO_RECEIPT_SCHEMA = "three-pass-verdict-run/v1"
RECOVERY_RECEIPT_SCHEMA = "averitec-jev-selector-recovery-receipt/v1"
PASS = 4


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("object_required")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def spec(path: Path) -> dict:
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def checked_choices(candidates: dict, choices_file: Path, receipt: dict) -> list[dict]:
    cases = candidates.get("cases")
    if candidates.get("gold_included") is not False or not isinstance(cases, list) or len(cases) != 100:
        raise ValueError("candidates_not_gold_free_100")
    if len({case["case_id"] for case in cases}) != 100:
        raise ValueError("duplicate_candidate_cases")
    if any(not isinstance(case.get("candidates"), list) or len(case["candidates"]) != 10 for case in cases):
        raise ValueError("candidate_count")
    if receipt.get("schema") != RECOVERY_RECEIPT_SCHEMA or receipt.get("phase") != "pass" or receipt.get("complete") is not True:
        raise ValueError("recovery_receipt_incomplete")
    if receipt.get("measured_results") != 1000 or receipt.get("measured_ok") != 1000 or receipt.get("logical_calls") != 1001:
        raise ValueError("recovery_receipt_count")
    if receipt.get("journal_sha256", {}).get("choices.jsonl") != sha(choices_file):
        raise ValueError("choices_receipt_hash")
    rows = [json.loads(line) for line in choices_file.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1001 or rows[0].get("phase") != "warmup" or rows[0].get("outcome") != "ok":
        raise ValueError("choices_count_or_warmup")
    expected = [(case["case_id"], candidate["id"]) for case in cases for candidate in case["candidates"]]
    measured = rows[1:]
    if [(row.get("case_id"), row.get("id")) for row in measured] != expected:
        raise ValueError("choice_order_or_identity")
    for row in measured:
        if (row.get("phase") != "measured" or row.get("outcome") != "ok"
                or row.get("action") not in {"include", "exclude"}
                or row.get("model") != "jev-1.13.0" or row.get("returned_model") != "jev-1.13.0"
                or row.get("trace_complete") is not True):
            raise ValueError("choice_unvalidated")
    return measured


def build(candidates_file: Path, choices_file: Path, recovery_receipt_file: Path,
          recovery_manifest_file: Path, recovery_gate_file: Path,
          original_hero_manifest_file: Path, original_hero_receipt_file: Path,
          original_hero_results_file: Path, original_packages_file: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("output_exists")
    candidates = read(candidates_file)
    if sha(candidates_file) != SOURCE_CANDIDATES_SHA256:
        raise ValueError("candidate_source_hash")
    receipt = read(recovery_receipt_file)
    choices = checked_choices(candidates, choices_file, receipt)
    recovery_manifest = read(recovery_manifest_file)
    if receipt.get("manifest_sha256") != sha(recovery_manifest_file):
        raise ValueError("recovery_manifest_receipt_hash")
    if recovery_manifest.get("phase") != "pass" or recovery_manifest.get("candidates", {}).get("sha256") != SOURCE_CANDIDATES_SHA256:
        raise ValueError("recovery_manifest_source")
    gate = read(recovery_gate_file)
    if (gate.get("decision") != "approved" or gate.get("manifest_sha256") != sha(recovery_manifest_file)
            or receipt.get("gate_sha256") != sha(recovery_gate_file)):
        raise ValueError("recovery_gate_binding")
    original = read(original_hero_manifest_file)
    if sha(original_hero_manifest_file) != ORIGINAL_HERO_MANIFEST_SHA256 or original.get("candidates_sha256") != SOURCE_CANDIDATES_SHA256:
        raise ValueError("original_hero_manifest_binding")
    old_receipt = read(original_hero_receipt_file)
    if (old_receipt.get("schema") != ORIGINAL_HERO_RECEIPT_SCHEMA or old_receipt.get("complete") is not True
            or old_receipt.get("manifest_sha256") != ORIGINAL_HERO_MANIFEST_SHA256
            or old_receipt.get("measured_results") != 2700
            or old_receipt.get("results_sha256") != sha(original_hero_results_file)):
        raise ValueError("original_hero_receipt_binding")
    old_results = [json.loads(line) for line in original_hero_results_file.read_text(encoding="utf-8").splitlines()]
    old_jev3_results = [row for row in old_results if row.get("phase") == "measured"
                        and row.get("arm") == "jev" and row.get("pass") == 3]
    if (len(old_jev3_results) != 100 or sum(row.get("outcome") == "ok" for row in old_jev3_results) != 9
            or sum(row.get("outcome") == "upstream_package_failure" for row in old_jev3_results) != 91):
        raise ValueError("original_jev3_verdict_status_drift")
    old_packages = read(original_packages_file)
    if sha(original_packages_file) != original.get("packages_sha256") or len(old_packages.get("packages", [])) != 2700:
        raise ValueError("original_hero_packages_binding")
    original_jev3 = [p for p in old_packages["packages"] if p.get("arm") == "jev" and p.get("pass") == 3]
    if len(original_jev3) != 100 or sum(p.get("status") == "failed" for p in original_jev3) != 91:
        raise ValueError("original_jev3_status_drift")
    grouped = [choices[i:i + 10] for i in range(0, len(choices), 10)]
    packages = []
    for case, decisions in zip(candidates["cases"], grouped):
        selected = [
            {field: candidate[field] for field in verdict.PASSAGE_FIELDS}
            for candidate, choice in zip(case["candidates"], decisions)
            if choice["action"] == "include"
        ]
        packages.append({"schema": verdict.PACKAGE_SCHEMA, "arm": "jev", "pass": PASS,
                         "case_id": case["case_id"], "group_id": case["group_id"],
                         "claim": case["claim"], "mode": "selector", "status": "ready",
                         "errors": [], "selected_passages": selected})
    output_dir.mkdir(parents=True, exist_ok=False)
    package_path = output_dir / "packages.json"
    write_new(package_path, {"schema": "three-pass-verdict-packages/v1", "packages": packages})
    manifest = dict(original)
    manifest.update({"arms": ["jev"], "passes": [PASS], "maximum_calls": 101,
                     "packages_sha256": sha(package_path), "maximum_total_input_tokens": 100000,
                     "maximum_total_output_tokens": 101 * original["max_output_tokens"]})
    manifest_path = output_dir / "verdict-manifest.json"
    write_new(manifest_path, manifest)
    provenance = {"schema": SCHEMA, "gold_included": False,
                  "raw_slot": {"arm": "jev", "pass": PASS},
                  "comparison_substitution": {"arm": "jev", "pass": 3},
                  "planned_cases": 100, "planned_model_calls": 101, "retries": 0,
                  "model": original["model"], "revision": original["revision"],
                  "candidates": spec(candidates_file), "choices": spec(choices_file),
                  "recovery_receipt": spec(recovery_receipt_file),
                  "recovery_manifest": spec(recovery_manifest_file),
                  "recovery_gate": spec(recovery_gate_file),
                  "original_hero_manifest": spec(original_hero_manifest_file),
                  "original_hero_receipt": spec(original_hero_receipt_file),
                  "original_hero_results": spec(original_hero_results_file),
                  "original_packages": spec(original_packages_file),
                  "new_packages": spec(package_path), "new_verdict_manifest": spec(manifest_path),
                  "package_order_sha256": hashlib.sha256(json.dumps(
                      [p["case_id"] for p in packages], separators=(",", ":")).encode()).hexdigest()}
    write_new(output_dir / "scope.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "choices", "recovery-receipt", "recovery-manifest", "recovery-gate",
                 "original-hero-manifest", "original-hero-receipt", "original-hero-results", "original-packages", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = vars(parser.parse_args())
    result = build(*(args[name.replace("-", "_")] for name in (
        "candidates", "choices", "recovery-receipt", "recovery-manifest", "recovery-gate",
        "original-hero-manifest", "original-hero-receipt", "original-hero-results", "original-packages", "output")))
    print(json.dumps({"schema": result["schema"], "output": str(args["output"]), "scope_sha256": sha(args["output"] / "scope.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
