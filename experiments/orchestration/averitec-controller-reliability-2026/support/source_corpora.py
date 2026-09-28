"""Hash-bound AVeriTeC source-corpus registry for mixed train/dev runs.

The study split keeps the original source split and array index as identity.
This module maps that identity to one declared ZIP archive and member without
opening gold data.  It supports the official three-shard train layout and the
single dev archive while keeping local paths outside tracked experiment data.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable
from zipfile import ZipFile


SCHEMA = "averitec-source-corpora/v1"
CASE_ID_RE = re.compile(r"averitec-(train|dev|test)-(\d{4})$")
SHA256_RE = re.compile(r"[0-9a-f]{64}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_repository_path(value: str, repository_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("source_corpora_path")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError("source_corpora_path_outside_repository") from error
    return resolved


def _member_name(template: str, index: int) -> str:
    if not isinstance(template, str) or template.count("{index}") != 1:
        raise ValueError("source_corpora_member_template")
    try:
        value = template.format(index=index)
    except (KeyError, ValueError) as error:
        raise ValueError("source_corpora_member_template") from error
    member = PurePosixPath(value)
    if member.is_absolute() or not member.parts or ".." in member.parts or str(member) != value:
        raise ValueError("source_corpora_member_template")
    return value


@dataclass(frozen=True)
class CorpusArchive:
    split: str
    first_index: int
    last_index: int
    path: Path
    expected_sha256: str
    member_template: str

    def contains(self, index: int) -> bool:
        return self.first_index <= index <= self.last_index

    def member(self, index: int) -> str:
        if not self.contains(index):
            raise ValueError("source_corpora_index_out_of_range")
        return _member_name(self.member_template, index)


class SourceCorpora:
    """Validated metadata plus lazy file/member verification for source ZIPs."""

    def __init__(self, manifest_path: Path, *, repository_root: Path = REPOSITORY_ROOT) -> None:
        self.manifest_path = manifest_path.resolve()
        self.repository_root = repository_root.resolve()
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("source_corpora_manifest_read") from error
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("source_corpora_manifest_schema")
        stores = payload.get("stores")
        if not isinstance(stores, dict) or not stores:
            raise ValueError("source_corpora_manifest_stores")
        archives: dict[str, tuple[CorpusArchive, ...]] = {}
        for split, store in stores.items():
            if split not in {"train", "dev", "test"} or not isinstance(store, dict):
                raise ValueError("source_corpora_split")
            rows = store.get("archives")
            if not isinstance(rows, list) or not rows:
                raise ValueError("source_corpora_archives")
            parsed: list[CorpusArchive] = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("source_corpora_archive")
                first, last = row.get("first_index"), row.get("last_index")
                expected = row.get("sha256")
                template = row.get("member_template")
                if (isinstance(first, bool) or not isinstance(first, int) or first < 0
                        or isinstance(last, bool) or not isinstance(last, int) or last < first
                        or not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None):
                    raise ValueError("source_corpora_archive")
                _member_name(template, first)
                parsed.append(CorpusArchive(
                    split=split,
                    first_index=first,
                    last_index=last,
                    path=_safe_repository_path(row.get("path"), self.repository_root),
                    expected_sha256=expected,
                    member_template=template,
                ))
            parsed.sort(key=lambda item: item.first_index)
            for left, right in zip(parsed, parsed[1:]):
                if left.last_index >= right.first_index:
                    raise ValueError("source_corpora_archive_overlap")
            archives[split] = tuple(parsed)
        self.archives = archives

    @property
    def manifest_sha256(self) -> str:
        return sha256(self.manifest_path)

    def binding(self, split: str, index: int) -> CorpusArchive:
        matches = [item for item in self.archives.get(split, ()) if item.contains(index)]
        if len(matches) != 1:
            raise ValueError(f"source_corpora_case_unmapped:{split}:{index}")
        return matches[0]

    def locate(self, split: str, index: int) -> tuple[Path, str]:
        binding = self.binding(split, index)
        return binding.path, binding.member(index)

    def fingerprints(self, *, splits: Iterable[str] | None = None) -> dict[str, str]:
        selected = set(self.archives) if splits is None else set(splits)
        values = {
            str(item.path.relative_to(self.repository_root)): item.expected_sha256
            for split in sorted(selected)
            for item in self.archives.get(split, ())
        }
        if not values:
            raise ValueError("source_corpora_selected_splits")
        return values

    def verify_cases(self, cases: Iterable[dict[str, Any]]) -> dict[str, str]:
        """Hash required archives and verify exact members for selected cases."""
        required: dict[CorpusArchive, set[str]] = {}
        for case in cases:
            case_id, split = case.get("case_id"), case.get("split")
            match = CASE_ID_RE.fullmatch(case_id) if isinstance(case_id, str) else None
            if match is None or split != match.group(1):
                raise ValueError("source_corpora_runtime_identity")
            index = int(match.group(2))
            binding = self.binding(split, index)
            required.setdefault(binding, set()).add(binding.member(index))
        verified: dict[str, str] = {}
        for binding, members in required.items():
            if not binding.path.is_file():
                raise ValueError(f"source_corpora_archive_missing:{binding.path}")
            actual = sha256(binding.path)
            if actual != binding.expected_sha256:
                raise ValueError(f"source_corpora_archive_hash:{binding.path}")
            with ZipFile(binding.path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError(f"source_corpora_duplicate_members:{binding.path}")
                missing = sorted(members - set(names))
                if missing:
                    raise ValueError("source_corpora_members_missing:" + ",".join(missing[:10]))
            verified[str(binding.path.relative_to(self.repository_root))] = actual
        return verified


def parse_case_identity(case_id: str) -> tuple[str, int]:
    match = CASE_ID_RE.fullmatch(case_id) if isinstance(case_id, str) else None
    if match is None:
        raise ValueError("canonical_case_id")
    return match.group(1), int(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-claims", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    corpora = SourceCorpora(args.manifest)
    claims: list[dict[str, Any]] = []
    if args.runtime_claims is not None:
        from data_preparation import load_runtime_claims

        claims = list(load_runtime_claims(args.runtime_claims))
    if args.check_files and not claims:
        raise ValueError("runtime_claims_required_for_file_check")
    verified = corpora.verify_cases(claims) if args.check_files else {}
    print(json.dumps({
        "schema": "averitec-source-corpora-check/v1",
        "manifest_sha256": corpora.manifest_sha256,
        "selected_case_count": len(claims),
        "expected_archives": corpora.fingerprints(splits={row["split"] for row in claims}) if claims else corpora.fingerprints(),
        "verified_archives": verified,
        "files_verified": args.check_files,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
