"""Synthetic hash-pinned archive tests; no gold or external services."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "source_rejoin.py"
SPEC = importlib.util.spec_from_file_location("three_pass_source_rejoin", MODULE_PATH)
assert SPEC and SPEC.loader
rejoin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rejoin)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, str, str, dict, dict]:
    cases, members = [], {}
    for case_number in (0, 1):
        candidates, archive_rows = [], []
        for rank in range(10):
            url = f"https://example.test/{case_number}/{rank}"
            full_source = f"Prefix {case_number}/{rank}. Exact evidence\u2028line {case_number}/{rank}. Suffix."
            start = full_source.index("Exact")
            text = full_source[start:start + len(f"Exact evidence\u2028line {case_number}/{rank}.")]
            candidates.append({
                "id": f"C{rank + 1:02d}", "passage_id": rejoin.stable_passage_id(url, text),
                "text": text, "url": url, "source_start": start,
                "source_text_length": len(full_source),
                "source_text_sha256": digest(full_source.encode("utf-8")),
            })
            archive_rows.append({"url": url, "url2text": [full_source]})
        cases.append({"case_id": f"averitec-dev-{case_number:04d}", "group_id": f"group-{case_number}",
                      "claim": "claim is inference-only", "candidates": candidates})
        members[f"output_dev/{case_number}.json"] = b"\n".join(json.dumps(row, ensure_ascii=False).encode() for row in archive_rows)
    archive_file = tmp_path / "dev.zip"
    with ZipFile(archive_file, "w") as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    archive_sha = digest(archive_file.read_bytes())
    manifest_file = tmp_path / "source-manifest.json"
    manifest_file.write_text(json.dumps({"schema": "averitec-source-corpora/v1", "stores": {"dev": {"archives": [{
        "path": "dev.zip", "sha256": archive_sha, "first_index": 0, "last_index": 1,
        "member_template": "output_dev/{index}.json",
    }]}}}))
    document = {"gold_included": False, "cases": cases, "gold_verdict": "DO NOT COPY"}
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(json.dumps(document, ensure_ascii=False))
    return candidate_file, manifest_file, archive_file, digest(candidate_file.read_bytes()), archive_sha, document, members


def run(tmp_path: Path, candidate_file: Path, manifest_file: Path, candidate_sha: str, archive_sha: str) -> dict:
    return rejoin.rejoin_candidates(candidate_file.name, input_root=tmp_path,
                                    expected_candidate_sha256=candidate_sha,
                                    source_manifest_path=manifest_file.name,
                                    expected_archive_sha256=archive_sha,
                                    expected_cases=2)


def test_exact_full_source_rejoin_and_no_gold(tmp_path: Path) -> None:
    candidate_file, manifest_file, _, candidate_sha, archive_sha, document, _ = fixture(tmp_path)
    output = run(tmp_path, candidate_file, manifest_file, candidate_sha, archive_sha)
    assert len(output["cases"]) == 2
    assert [len(case["passages"]) for case in output["cases"]] == [10, 10]
    first = output["cases"][0]["passages"][0]
    original = document["cases"][0]["candidates"][0]
    assert {key: first[key] for key in rejoin._FIELDS} == original
    assert first["source_text"][first["source_start"]:first["source_start"] + len(first["text"])] == first["text"]
    assert first["source_text"].startswith("Prefix")
    assert "gold" not in json.dumps(output) and "DO NOT COPY" not in json.dumps(output)


@pytest.mark.parametrize("field,value,error", [
    ("url", "https://wrong.test", "candidate_passage_id"),
    ("passage_id", "passage-wrong", "candidate_passage_id"),
    ("text", "wrong evidence", "candidate_passage_id"),
    ("source_start", 0, "source_rejoin_missing"),
    ("source_text_length", 3, "candidate_fields"),
    ("source_text_sha256", "0" * 64, "source_rejoin_missing"),
])
def test_identity_or_provenance_mismatch_fails(tmp_path: Path, field: str, value, error: str) -> None:
    candidate_file, manifest_file, _, _, archive_sha, document, _ = fixture(tmp_path)
    document["cases"][0]["candidates"][0][field] = value
    candidate_file.write_text(json.dumps(document, ensure_ascii=False))
    with pytest.raises(ValueError, match=error):
        run(tmp_path, candidate_file, manifest_file, digest(candidate_file.read_bytes()), archive_sha)


def test_byte_identical_duplicate_source_is_one_provenance_match(tmp_path: Path) -> None:
    candidate_file, manifest_file, archive_file, candidate_sha, _, _, members = fixture(tmp_path)
    lines = members["output_dev/0.json"].split(b"\n")
    members["output_dev/0.json"] += b"\n" + lines[0]
    with ZipFile(archive_file, "w") as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    archive_sha = digest(archive_file.read_bytes())
    manifest = json.loads(manifest_file.read_text())
    manifest["stores"]["dev"]["archives"][0]["sha256"] = archive_sha
    manifest_file.write_text(json.dumps(manifest))
    output = run(tmp_path, candidate_file, manifest_file, candidate_sha, archive_sha)
    assert output["cases"][0]["passages"][0]["source_text"].startswith("Prefix 0/0.")


def test_distinct_full_sources_remain_ambiguous_even_if_digest_collides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_file, manifest_file, archive_file, candidate_sha, _, document, members = fixture(tmp_path)
    first = json.loads(members["output_dev/0.json"].split(b"\n")[0])
    original = first["url2text"][0]
    changed = original[:-1] + "!"
    assert changed != original and len(changed) == len(original)
    first["url2text"] = [changed]
    members["output_dev/0.json"] += b"\n" + json.dumps(first, ensure_ascii=False).encode()
    with ZipFile(archive_file, "w") as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    archive_sha = digest(archive_file.read_bytes())
    manifest = json.loads(manifest_file.read_text())
    manifest["stores"]["dev"]["archives"][0]["sha256"] = archive_sha
    manifest_file.write_text(json.dumps(manifest))

    # A real SHA-256 collision is impractical to generate. Substitute only the
    # digest used during source matching to exercise the distinct-text guard.
    expected = document["cases"][0]["candidates"][0]["source_text_sha256"]
    original_hashlib = hashlib

    def colliding_sha256(data: bytes):
        actual = original_hashlib.sha256(data)
        if data == changed.encode("utf-8"):
            return SimpleNamespace(hexdigest=lambda: expected)
        return actual

    monkeypatch.setattr(rejoin, "hashlib", SimpleNamespace(sha256=colliding_sha256))
    with pytest.raises(ValueError, match="source_rejoin_ambiguous"):
        run(tmp_path, candidate_file, manifest_file, candidate_sha, archive_sha)


def test_consistent_but_foreign_url_still_fails_archive_join(tmp_path: Path) -> None:
    candidate_file, manifest_file, _, _, archive_sha, document, _ = fixture(tmp_path)
    candidate = document["cases"][0]["candidates"][0]
    candidate["url"] = "https://foreign.test/same-text"
    candidate["passage_id"] = rejoin.stable_passage_id(candidate["url"], candidate["text"])
    candidate_file.write_text(json.dumps(document, ensure_ascii=False))
    with pytest.raises(ValueError, match="source_rejoin_missing"):
        run(tmp_path, candidate_file, manifest_file, digest(candidate_file.read_bytes()), archive_sha)


def test_candidate_and_archive_hashes_are_enforced(tmp_path: Path) -> None:
    candidate_file, manifest_file, archive_file, candidate_sha, archive_sha, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="candidate_file_hash"):
        run(tmp_path, candidate_file, manifest_file, "0" * 64, archive_sha)
    with pytest.raises(ValueError, match="dev_archive_binding_hash"):
        run(tmp_path, candidate_file, manifest_file, candidate_sha, "0" * 64)
    archive_file.write_bytes(archive_file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="source_corpora_archive_hash"):
        run(tmp_path, candidate_file, manifest_file, candidate_sha, archive_sha)
