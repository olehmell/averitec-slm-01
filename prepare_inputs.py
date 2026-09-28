"""Rebuild the private AVeriTeC runtime from exact upstream train/dev files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "experiments/orchestration/averitec-controller-reliability-2026"
SUPPORT = STUDY / "support"
MANIFEST = SUPPORT / "manifests/study-split-80-20-20260917.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    inputs = {name: ROOT / manifest["inputs"][name]["path"] for name in ("train", "dev")}
    for name, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"missing {name} input: {path}")
        if sha256(path) != manifest["inputs"][name]["sha256"]:
            raise SystemExit(f"{name} input SHA-256 differs from the published study input")
    sys.path.insert(0, str(SUPPORT))
    from prepare_study_split import build_study_split

    output = ROOT / "datasets/prepared_averitec/study-80-20-20260917"
    generated = build_study_split(inputs["train"], inputs["dev"], output, MANIFEST)
    for item in generated["outputs"].values():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("prepared_output_hash:" + item["path"])
    print(json.dumps({"prepared": True, "fit": generated["roles"]["study_fit"]["count"],
                      "holdout": generated["roles"]["study_holdout"]["count"],
                      "gold_and_runtime_separated": True}, sort_keys=True))


if __name__ == "__main__":
    main()
