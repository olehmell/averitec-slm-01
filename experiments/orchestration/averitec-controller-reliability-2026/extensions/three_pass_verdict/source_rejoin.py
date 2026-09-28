"""Offline, gold-free rejoin of frozen windows to exact archived source text.

Use an input root containing the ignored original dev archive (normally the
main checkout). No AVeriTeC gold, network service, or model is accessed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile


_SIBLING = Path(__file__).resolve().parents[2] / "support"
if str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))

from averitec_fixed import stable_passage_id  # noqa: E402
from source_corpora import SourceCorpora, parse_case_identity, sha256  # noqa: E402


DEFAULT_MANIFEST = Path("experiments/orchestration/averitec-controller-reliability-2026/support/manifests/source-corpora-averitec-train-dev-20260917.json")
ORIGINAL_DEV_ARCHIVE_SHA256 = "021e258cd6fb5fe6d627a4667d663e95c184c966939c15124df9206142fc2212"
_FIELDS = ("id", "passage_id", "text", "url", "source_start", "source_text_length", "source_text_sha256")


def _rooted(root: Path, path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _load_candidates(path: Path, expected_hash: str, expected_cases: int, candidates_per_case: int) -> list[dict]:
    if not path.is_file() or sha256(path) != expected_hash:
        raise ValueError("candidate_file_hash")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("gold_included") is not False:
        raise ValueError("candidates_not_gold_free")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError("candidate_case_count")
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError("invalid_candidate_case")
        case_id = case["case_id"]
        split, _ = parse_case_identity(case_id)
        if split != "dev" or case_id in seen_cases or not isinstance(case.get("group_id"), str) or not case["group_id"]:
            raise ValueError("candidate_case_identity")
        seen_cases.add(case_id)
        rows = case.get("candidates")
        if not isinstance(rows, list) or len(rows) != candidates_per_case:
            raise ValueError(f"candidate_count:{case_id}")
        seen_ids: set[str] = set()
        seen_passages: set[str] = set()
        for candidate in rows:
            if not isinstance(candidate, dict) or any(key not in candidate for key in _FIELDS):
                raise ValueError(f"candidate_fields:{case_id}")
            if (any(not isinstance(candidate[key], str) or not candidate[key] for key in ("id", "passage_id", "text", "url"))
                    or type(candidate["source_start"]) is not int or candidate["source_start"] < 0
                    or type(candidate["source_text_length"]) is not int
                    or candidate["source_start"] + len(candidate["text"]) > candidate["source_text_length"]
                    or not isinstance(candidate["source_text_sha256"], str)
                    or len(candidate["source_text_sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in candidate["source_text_sha256"])):
                raise ValueError(f"candidate_fields:{case_id}")
            if candidate["id"] in seen_ids or candidate["passage_id"] in seen_passages:
                raise ValueError(f"duplicate_candidate_id:{case_id}")
            seen_ids.add(candidate["id"])
            seen_passages.add(candidate["passage_id"])
            if stable_passage_id(candidate["url"], candidate["text"]) != candidate["passage_id"]:
                raise ValueError(f"candidate_passage_id:{case_id}:{candidate['id']}")
    return cases


def _rejoin_case(case: dict, archive: SourceCorpora) -> list[dict]:
    case_id = case["case_id"]
    _, index = parse_case_identity(case_id)
    archive_path, member = archive.locate("dev", index)
    targets: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for candidate in case["candidates"]:
        targets[(candidate["url"], candidate["source_text_length"], candidate["source_text_sha256"])].append(candidate)
    # An archive may repeat the same scraped string under one URL. Repeated
    # byte-identical source text is one provenance match; different full texts
    # remain ambiguous even if they somehow share the declared digest.
    matches: dict[str, set[str]] = {candidate["id"]: set() for candidate in case["candidates"]}
    with ZipFile(archive_path) as source:
        raw_member = source.read(member)
    # Physical LF alone separates the JSONL records. str.splitlines corrupts
    # U+2028/U+2029 inside otherwise valid JSON source strings.
    for line in raw_member.split(b"\n"):
        if not line.strip():
            continue
        row = json.loads(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise ValueError(f"archive_row_shape:{case_id}")
        url, texts = row.get("url"), row.get("url2text")
        if not isinstance(url, str) or not isinstance(texts, list):
            continue
        for source_text in texts:
            if not isinstance(source_text, str) or not source_text:
                continue
            digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            for candidate in targets.get((url, len(source_text), digest), ()):
                start, text = candidate["source_start"], candidate["text"]
                if source_text[start:start + len(text)] == text:
                    matches[candidate["id"]].add(source_text)
    result = []
    for candidate in case["candidates"]:
        found = matches[candidate["id"]]
        if len(found) != 1:
            raise ValueError(f"source_rejoin_{'missing' if not found else 'ambiguous'}:{case_id}:{candidate['id']}")
        result.append({**{key: candidate[key] for key in _FIELDS}, "source_text": next(iter(found))})
    return result


def rejoin_candidates(
    candidates_path: Path,
    *,
    input_root: Path,
    expected_candidate_sha256: str,
    source_manifest_path: Path = DEFAULT_MANIFEST,
    expected_archive_sha256: str = ORIGINAL_DEV_ARCHIVE_SHA256,
    expected_manifest_sha256: str | None = None,
    expected_cases: int = 100,
    candidates_per_case: int = 10,
) -> dict:
    """Return exact full sources and candidate identities in frozen order.

    Paths may be relative to ``input_root`` so an isolated worktree can read
    the main checkout's ignored archive. The caller must supply the frozen
    candidate file digest; the dev archive digest defaults to the pinned one.
    """
    if type(expected_cases) is not int or expected_cases < 1 or type(candidates_per_case) is not int or candidates_per_case < 1:
        raise ValueError("invalid_expected_shape")
    root = input_root.resolve()
    candidate_file = _rooted(root, candidates_path)
    manifest_file = _rooted(root, source_manifest_path)
    cases = _load_candidates(candidate_file, expected_candidate_sha256, expected_cases, candidates_per_case)
    if expected_manifest_sha256 is not None and sha256(manifest_file) != expected_manifest_sha256:
        raise ValueError("source_manifest_hash")
    archive = SourceCorpora(manifest_file, repository_root=root)
    if any(archive.binding("dev", parse_case_identity(case["case_id"])[1]).expected_sha256 != expected_archive_sha256 for case in cases):
        raise ValueError("dev_archive_binding_hash")
    verified = archive.verify_cases({"case_id": case["case_id"], "split": "dev"} for case in cases)
    result = [{"case_id": case["case_id"], "group_id": case["group_id"],
               "passages": _rejoin_case(case, archive)} for case in cases]
    return {"schema": "three-pass-source-rejoin/v1", "candidate_sha256": expected_candidate_sha256,
            "source_manifest_sha256": archive.manifest_sha256,
            "verified_archive_sha256": verified, "cases": result}
