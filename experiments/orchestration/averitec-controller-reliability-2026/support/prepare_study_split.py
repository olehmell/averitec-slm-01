#!/usr/bin/env python3
"""Build a deterministic grouped 80/20 AVeriTeC train/dev study split.

The runtime views intentionally contain no gold fields.  Gold views are local
training/evaluation inputs; the checked-in manifest contains only identities,
hashes, grouping metadata and aggregate label counts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable
import unicodedata


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[3]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from averitec_fixed import url_family  # noqa: E402


DEFAULT_SEED = 20260917
DEFAULT_FRACTION = 0.2
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets/prepared_averitec/study-80-20-20260917"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "manifests/study-split-80-20-20260917.json"
RUNTIME_FIELDS = ("case_id", "claim", "split")
GOLD_FIELDS = ("case_id", "claim", "split", "source_index", "label", "questions", "justification")
ROLES = ("study_fit", "study_holdout")
LABELS = frozenset(("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def normalize_claim(text: str) -> tuple[str, tuple[str, ...]]:
    """Return NFKC/casefold alphanumeric claim text and its token-set key."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return " ".join(tokens), tuple(sorted(set(tokens)))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, value: int) -> int:
        while self.parents[value] != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


@dataclass(frozen=True)
class StudyRow:
    case_id: str
    claim: str
    split: str
    source_index: int
    label: str
    questions: Any
    justification: Any
    normalized_claim: str
    token_set: tuple[str, ...]
    article_url_family: str | None

    def runtime(self) -> dict[str, str]:
        return {"case_id": self.case_id, "claim": self.claim, "split": self.split}

    def gold(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "claim": self.claim,
            "split": self.split,
            "source_index": self.source_index,
            "label": self.label,
            "questions": self.questions,
            "justification": self.justification,
        }


@dataclass
class StudyGroup:
    group_id: str
    rows: list[StudyRow]
    signals: tuple[str, ...]

    @property
    def labels(self) -> Counter[str]:
        return Counter(row.label for row in self.rows)


def _load_source(split: str, path: Path) -> tuple[list[StudyRow], list[dict[str, str]], list[dict[str, str]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"source_read:{split}") from error
    if not isinstance(payload, list):
        raise ValueError(f"source_schema:{split}")
    rows: list[StudyRow] = []
    exclusions: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for index, source in enumerate(payload):
        case_id = f"averitec-{split}-{index:04d}"
        if not isinstance(source, dict):
            raise ValueError(f"source_record:{case_id}")
        claim = source.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            exclusions.append({"case_id": case_id, "reason": "empty_or_invalid_claim"})
            continue
        label = source.get("label")
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError(f"source_label:{case_id}")
        normalized_claim, token_set = normalize_claim(claim)
        article_url_family = None
        article = source.get("fact_checking_article")
        if isinstance(article, str) and article.strip():
            try:
                article_url_family = url_family(article)
            except ValueError:
                warnings.append({"case_id": case_id, "reason": "invalid_fact_checking_article_url_family_omitted"})
        rows.append(StudyRow(
            case_id=case_id,
            claim=claim,
            split=split,
            source_index=index,
            label=label,
            questions=source.get("questions"),
            justification=source.get("justification"),
            normalized_claim=normalized_claim,
            token_set=token_set,
            article_url_family=article_url_family,
        ))
    return rows, exclusions, warnings


def load_rows(train_path: Path, dev_path: Path) -> tuple[list[StudyRow], list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    train, train_exclusions, train_warnings = _load_source("train", train_path)
    dev, dev_exclusions, dev_warnings = _load_source("dev", dev_path)
    return train + dev, train_exclusions + dev_exclusions, train_warnings + dev_warnings, {
        "train": len(train) + len(train_exclusions), "dev": len(dev) + len(dev_exclusions),
    }


def group_rows(rows: list[StudyRow]) -> list[StudyGroup]:
    union_find = UnionFind(len(rows))
    signals_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)

    def add_edges(values: dict[Any, list[int]], signal: str) -> None:
        for members in values.values():
            if len(members) < 2:
                continue
            first = members[0]
            for other in members[1:]:
                union_find.union(first, other)
                signals_by_pair[(min(first, other), max(first, other))].add(signal)

    normalized: dict[str, list[int]] = defaultdict(list)
    token_sets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    article_families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.normalized_claim:
            normalized[row.normalized_claim].append(index)
        if row.token_set:
            token_sets[row.token_set].append(index)
        if row.article_url_family is not None:
            article_families[row.article_url_family].append(index)
    add_edges(normalized, "normalized_claim")
    add_edges(token_sets, "token_set")
    add_edges(article_families, "fact_checking_article_url_family")

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union_find.find(index)].append(index)
    groups: list[StudyGroup] = []
    for members in components.values():
        member_rows = sorted((rows[index] for index in members), key=lambda row: row.case_id)
        member_set = set(members)
        signals = sorted({signal for pair, values in signals_by_pair.items() if pair[0] in member_set and pair[1] in member_set for signal in values})
        group_id = "group-" + hashlib.sha256(
            ("averitec-study-group/v1:" + "\n".join(row.case_id for row in member_rows)).encode("utf-8")
        ).hexdigest()[:20]
        groups.append(StudyGroup(group_id, member_rows, tuple(signals)))
    return sorted(groups, key=lambda group: group.group_id)


def _rank(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"averitec-study-split/v1:{seed}:{group_id}".encode("utf-8")).hexdigest()


def allocate_groups(groups: list[StudyGroup], fraction: float, seed: int) -> dict[str, str]:
    if not groups:
        raise ValueError("empty_study_pool")
    if not 0 < fraction < 1:
        raise ValueError("holdout_fraction")
    total = sum(len(group.rows) for group in groups)
    labels = Counter(label for group in groups for label in group.labels.elements())
    target_total = total * fraction
    target_labels = {label: count * fraction for label, count in labels.items()}
    assigned_total = 0
    assigned_labels: Counter[str] = Counter()
    roles: dict[str, str] = {}
    # Seeded group order prevents the allocation from systematically putting
    # the largest fact-check article components into holdout. The objective
    # below uses only count and class totals, never a model outcome.
    ordered = sorted(groups, key=lambda group: (_rank(seed, group.group_id), group.group_id))

    def loss(total_count: int, label_counts: Counter[str]) -> float:
        total_loss = abs(total_count - target_total) / max(1, total)
        label_loss = sum(abs(label_counts[label] - target_labels[label]) / max(1, labels[label]) for label in labels)
        return total_loss + label_loss

    for group in ordered:
        group_labels = group.labels
        holdout_labels = assigned_labels + group_labels
        fit_loss = loss(assigned_total, assigned_labels)
        holdout_loss = loss(assigned_total + len(group.rows), holdout_labels)
        if holdout_loss < fit_loss or (holdout_loss == fit_loss and _rank(seed, group.group_id) < "8" * 64):
            roles[group.group_id] = "study_holdout"
            assigned_total += len(group.rows)
            assigned_labels = holdout_labels
        else:
            roles[group.group_id] = "study_fit"
    # A group accepted early can leave an avoidable excess once later groups
    # balance the other classes. Revisit whole groups, changing a role only
    # when the count/class objective strictly improves. Seed order stays fixed.
    for _ in range(len(groups)):
        changed = False
        for group in ordered:
            delta = -1 if roles[group.group_id] == "study_holdout" else 1
            candidate_total = assigned_total + delta * len(group.rows)
            if not 0 < candidate_total < total:
                continue
            candidate_labels = assigned_labels.copy()
            for label, count in group.labels.items():
                candidate_labels[label] += delta * count
            if loss(candidate_total, candidate_labels) < loss(assigned_total, assigned_labels) - 1e-12:
                roles[group.group_id] = "study_fit" if delta < 0 else "study_holdout"
                assigned_total, assigned_labels = candidate_total, candidate_labels
                changed = True
        if not changed:
            break
    if not any(role == "study_fit" for role in roles.values()) or not any(role == "study_holdout" for role in roles.values()):
        raise ValueError("allocation_role_empty_grouping_prevents_split")
    return roles


def _group_metadata(group: StudyGroup, role: str) -> dict[str, Any]:
    labels = dict(sorted(group.labels.items()))
    normalized_label_sets: dict[str, set[str]] = defaultdict(set)
    for row in group.rows:
        if row.normalized_claim:
            normalized_label_sets[row.normalized_claim].add(row.label)
    return {
        "group_id": group.group_id,
        "role": role,
        "case_ids": [row.case_id for row in group.rows],
        "source_indices": [
            {"case_id": row.case_id, "split": row.split, "source_index": row.source_index}
            for row in group.rows
        ],
        "mixed_label_group": len(labels) > 1,
        "contradictory_normalized_claim_labels": any(len(values) > 1 for values in normalized_label_sets.values()),
        "grouping_signals": list(group.signals),
    }


def _role_metadata(rows: list[StudyRow], groups: list[StudyGroup], role: str) -> dict[str, Any]:
    return {
        "case_ids": [row.case_id for row in rows],
        "group_ids": [group.group_id for group in groups],
        "source_indices": [
            {"case_id": row.case_id, "split": row.split, "source_index": row.source_index}
            for row in rows
        ],
        "count": len(rows),
        "label_histogram": dict(sorted(Counter(row.label for row in rows).items())),
        "role": role,
    }


def _artifact_payloads(
    *, rows: list[StudyRow], groups: list[StudyGroup], roles: dict[str, str],
    train_path: Path, dev_path: Path, output_dir: Path, manifest_path: Path,
    exclusions: list[dict[str, str]], warnings: list[dict[str, str]], source_counts: dict[str, int],
    seed: int, fraction: float,
) -> dict[Path, bytes]:
    grouped = {role: [group for group in groups if roles[group.group_id] == role] for role in ROLES}
    role_rows = {
        role: sorted((row for group in grouped[role] for row in group.rows), key=lambda row: (row.split, row.source_index))
        for role in ROLES
    }
    outputs = {
        "fit_runtime": output_dir / "fit-runtime.jsonl",
        "holdout_runtime": output_dir / "holdout-runtime.jsonl",
        "fit_gold": output_dir / "fit-gold.jsonl",
        "holdout_gold": output_dir / "holdout-gold.jsonl",
    }
    payloads: dict[Path, bytes] = {
        outputs["fit_runtime"]: _jsonl_bytes(row.runtime() for row in role_rows["study_fit"]),
        outputs["holdout_runtime"]: _jsonl_bytes(row.runtime() for row in role_rows["study_holdout"]),
        outputs["fit_gold"]: _jsonl_bytes(row.gold() for row in role_rows["study_fit"]),
        outputs["holdout_gold"]: _jsonl_bytes(row.gold() for row in role_rows["study_holdout"]),
    }
    manifest = {
        "schema": "averitec-study-split/v1",
        "study_roles": {"fit": "study_fit", "holdout": "study_holdout"},
        "inputs": {
            "train": {"path": _display_path(train_path), "sha256": sha256(train_path), "records": source_counts["train"]},
            "dev": {"path": _display_path(dev_path), "sha256": sha256(dev_path), "records": source_counts["dev"]},
        },
        "command": {
            "builder": "experiments/orchestration/averitec-adaptive-orchestration-2026/prepare_study_split.py",
            "seed": seed,
            "holdout_fraction": fraction,
        },
        "method": {
            "grouping": [
                "transitive normalized_claim equality: Unicode NFKC, casefold, alphanumeric tokens",
                "transitive alphanumeric token-set equality",
                "transitive canonical fact_checking_article URL family via averitec_fixed.url_family",
            ],
            "no_semantic_event_grouping": True,
            "allocation": "seed-ordered class-aware grouped greedy allocation with strictly improving whole-group refinement; labels are used only for stratification",
        },
        "roles": {
            role: _role_metadata(role_rows[role], grouped[role], role)
            for role in ROLES
        },
        "groups": [_group_metadata(group, roles[group.group_id]) for group in groups],
        "exclusions": exclusions,
        "source_metadata_warnings": warnings,
        "outputs": {
            key: {"path": _display_path(path), "sha256": hashlib.sha256(payload).hexdigest(), "records": len(role_rows["study_fit" if key.startswith("fit_") else "study_holdout"])}
            for key, (path, payload) in {
                "fit_runtime": (outputs["fit_runtime"], payloads[outputs["fit_runtime"]]),
                "holdout_runtime": (outputs["holdout_runtime"], payloads[outputs["holdout_runtime"]]),
                "fit_gold": (outputs["fit_gold"], payloads[outputs["fit_gold"]]),
                "holdout_gold": (outputs["holdout_gold"], payloads[outputs["holdout_gold"]]),
            }.items()
        },
        "limitations": {
            "status": "holdout-for-future-local-fitting-not-certified-unseen-to-HerO",
            "historical_usage": "historical train/dev exposure and pretrained HerO membership cannot be erased by this split",
        },
    }
    payloads[manifest_path] = _json_bytes(manifest)
    return payloads


def _write_idempotent(payloads: dict[Path, bytes]) -> None:
    for path, expected in payloads.items():
        if path.is_symlink():
            raise FileExistsError(f"artifact_symlink_rejected:{path}")
        if path.exists() and path.read_bytes() != expected:
            raise FileExistsError(f"refuse_overwrite_different_artifact:{path}")
        if path.exists() and not path.is_file():
            raise FileExistsError(f"artifact_not_regular_file:{path}")
    temporary: list[tuple[Path, Path]] = []
    try:
        for path, payload in payloads.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.append((Path(stream.name), path))
        for source, target in temporary:
            os.replace(source, target)
    except Exception:
        for source, _ in temporary:
            source.unlink(missing_ok=True)
        raise


def build_study_split(
    train_path: Path, dev_path: Path, output_dir: Path, manifest_path: Path,
    *, seed: int = DEFAULT_SEED, fraction: float = DEFAULT_FRACTION,
) -> dict[str, Any]:
    rows, exclusions, warnings, source_counts = load_rows(train_path, dev_path)
    groups = group_rows(rows)
    roles = allocate_groups(groups, fraction, seed)
    payloads = _artifact_payloads(
        rows=rows, groups=groups, roles=roles, train_path=train_path, dev_path=dev_path,
        output_dir=output_dir, manifest_path=manifest_path, exclusions=exclusions,
        warnings=warnings, source_counts=source_counts, seed=seed, fraction=fraction,
    )
    _write_idempotent(payloads)
    return json.loads(payloads[manifest_path])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"artifact_jsonl:{path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"artifact_jsonl_schema:{path}")
    return rows


def check_study_split(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("manifest_read") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != "averitec-study-split/v1":
        raise ValueError("manifest_schema")
    inputs, outputs, roles, groups = (manifest.get(key) for key in ("inputs", "outputs", "roles", "groups"))
    if not isinstance(inputs, dict) or not isinstance(outputs, dict) or not isinstance(roles, dict) or not isinstance(groups, list):
        raise ValueError("manifest_structure")
    for split in ("train", "dev"):
        item = inputs.get(split)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or sha256(_resolve_path(item["path"])) != item.get("sha256"):
            raise ValueError(f"input_hash:{split}")
    source_rows, expected_exclusions, _warnings, source_counts = load_rows(
        _resolve_path(inputs["train"]["path"]), _resolve_path(inputs["dev"]["path"])
    )
    if any(inputs[name].get("records") != source_counts[name] for name in ("train", "dev")):
        raise ValueError("input_counts")
    command = manifest.get("command")
    if (
        not isinstance(command, dict) or not isinstance(command.get("seed"), int)
        or isinstance(command["seed"], bool)
        or not isinstance(command.get("holdout_fraction"), (int, float))
        or isinstance(command["holdout_fraction"], bool)
    ):
        raise ValueError("command_schema")
    expected_groups = group_rows(source_rows)
    expected_roles = allocate_groups(expected_groups, float(command["holdout_fraction"]), command["seed"])
    source_by_id = {row.case_id: row for row in source_rows}
    role_ids: dict[str, set[str]] = {}
    role_groups: dict[str, set[str]] = {}
    for role in ROLES:
        item = roles.get(role)
        if not isinstance(item, dict) or item.get("role") != role or not isinstance(item.get("case_ids"), list) or not isinstance(item.get("group_ids"), list):
            raise ValueError(f"role_schema:{role}")
        role_ids[role] = set(item["case_ids"])
        role_groups[role] = set(item["group_ids"])
        if len(role_ids[role]) != len(item["case_ids"]) or len(role_groups[role]) != len(item["group_ids"]):
            raise ValueError(f"role_duplicates:{role}")
    if role_ids["study_fit"] & role_ids["study_holdout"] or role_groups["study_fit"] & role_groups["study_holdout"]:
        raise ValueError("role_overlap")
    grouped_ids: set[str] = set()
    grouped_names: set[str] = set()
    expected_by_group = {group.group_id: group for group in expected_groups}
    for group in groups:
        if not isinstance(group, dict) or group.get("role") not in ROLES or not isinstance(group.get("group_id"), str) or not isinstance(group.get("case_ids"), list):
            raise ValueError("group_schema")
        group_id, group_role, case_ids = group["group_id"], group["role"], set(group["case_ids"])
        expected_group = expected_by_group.get(group_id)
        if (
            group_id in grouped_names or not case_ids or grouped_ids & case_ids or not case_ids <= role_ids[group_role]
            or expected_group is None or case_ids != {row.case_id for row in expected_group.rows}
            or expected_roles[group_id] != group_role
            or group != _group_metadata(expected_group, group_role)
        ):
            raise ValueError("group_assignment")
        grouped_names.add(group_id)
        grouped_ids.update(case_ids)
    if grouped_ids != role_ids["study_fit"] | role_ids["study_holdout"] or grouped_names != role_groups["study_fit"] | role_groups["study_holdout"]:
        raise ValueError("group_coverage")
    if grouped_ids != set(source_by_id):
        raise ValueError("source_coverage")
    if manifest.get("exclusions") != expected_exclusions:
        raise ValueError("exclusions")
    for key, role, fields in (
        ("fit_runtime", "study_fit", RUNTIME_FIELDS),
        ("holdout_runtime", "study_holdout", RUNTIME_FIELDS),
        ("fit_gold", "study_fit", GOLD_FIELDS),
        ("holdout_gold", "study_holdout", GOLD_FIELDS),
    ):
        item = outputs.get(key)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"output_schema:{key}")
        path = _resolve_path(item["path"])
        if sha256(path) != item.get("sha256"):
            raise ValueError(f"output_hash:{key}")
        artifact_rows = _read_jsonl(path)
        if len(artifact_rows) != item.get("records") or any(set(row) != set(fields) for row in artifact_rows):
            raise ValueError(f"output_fields:{key}")
        if {row["case_id"] for row in artifact_rows} != role_ids[role]:
            raise ValueError(f"output_identity:{key}")
        if key.endswith("runtime") and any(set(row) - set(RUNTIME_FIELDS) for row in artifact_rows):
            raise ValueError(f"runtime_gold_leak:{key}")
        expected_rows = sorted((source_by_id[case_id] for case_id in role_ids[role]), key=lambda row: (row.split, row.source_index))
        expected_payload = [row.runtime() if key.endswith("runtime") else row.gold() for row in expected_rows]
        if artifact_rows != expected_payload:
            raise ValueError(f"output_content:{key}")
        expected_histogram = dict(sorted(Counter(row["label"] for row in artifact_rows).items())) if key.endswith("gold") else None
        if expected_histogram is not None and expected_histogram != roles[role].get("label_histogram"):
            raise ValueError(f"label_histogram:{key}")
    for role in ROLES:
        expected_role = _role_metadata(
            sorted((source_by_id[case_id] for case_id in role_ids[role]), key=lambda row: (row.split, row.source_index)),
            sorted((expected_by_group[group_id] for group_id in role_groups[role]), key=lambda group: group.group_id), role,
        )
        if roles[role] != expected_role:
            raise ValueError(f"role_metadata:{role}")
    return {"ok": True, "manifest": _display_path(manifest_path), "records": len(grouped_ids)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=REPOSITORY_ROOT / "datasets/averitec/raw/train.json")
    parser.add_argument("--dev", type=Path, default=REPOSITORY_ROOT / "datasets/averitec/raw/dev.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        print(json.dumps(check_study_split(args.manifest), sort_keys=True))
        return 0
    manifest = build_study_split(args.train, args.dev, args.output_dir, args.manifest, seed=args.seed, fraction=args.holdout_fraction)
    print(json.dumps({"manifest": _display_path(args.manifest), "fit": manifest["roles"]["study_fit"]["count"], "holdout": manifest["roles"]["study_holdout"]["count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
