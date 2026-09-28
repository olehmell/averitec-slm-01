"""Offline structural check and synthetic workflow smoke for the public package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "experiments/orchestration/averitec-controller-reliability-2026"
SUPPORT = STUDY / "support"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> dict:
    required = [
        STUDY / "config.yaml",
        STUDY / "manifests/selection.json",
        STUDY / "results/final/workflow.csv",
        STUDY / "results/final/hero.csv",
        SUPPORT / "manifests/study-split-80-20-20260917.json",
        SUPPORT / "manifests/source-corpora-averitec-train-dev-20260917.json",
        SUPPORT / "manifests/leakage-exclusions.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise ValueError("missing_package_files:" + ",".join(missing))
    import yaml
    config = yaml.safe_load((STUDY / "config.yaml").read_text())
    manifest_path = ROOT / config["model_assets_manifest"]
    if digest(manifest_path) != config["model_assets_manifest_sha256"]:
        raise ValueError("model_assets_manifest_sha256")
    selection = json.loads((STUDY / "manifests/selection.json").read_text())
    if len(selection["cases"]) != 120:
        raise ValueError("selection_case_count")
    with (STUDY / "results/final/workflow.csv").open(newline="") as source:
        workflow = list(csv.DictReader(source))
    with (STUDY / "results/final/hero.csv").open(newline="") as source:
        hero = list(csv.DictReader(source))
    arms = {"jev", "qwen", "gemini", "lfm", "lfm26", "jeff", "laya_typed"}
    if not arms.issubset({r["controller"] for r in workflow}):
        raise ValueError("workflow_arms")
    if not arms.issubset({r["controller"] for r in hero}):
        raise ValueError("hero_arms")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".sqlite3"}:
            continue
        relative = path.relative_to(ROOT)
        if any(part in {".git", ".venv", ".private-artifacts", "datasets", "data_store", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.suffix not in {".py", ".yaml", ".json", ".md", ".txt", ".toml", ".csv", ""}:
            continue
        body = path.read_text(encoding="utf-8")
        if "/Users/" + "olehmell" in body or "/ce" + "ph" in body:
            raise ValueError("machine_specific_path:" + str(relative))
    runtime = ROOT / config["dataset_refs"][0]
    return {"package_valid": True, "selected_cases": 120, "workflow_arms": 7,
            "hero_arms": 7, "private_runtime_present": runtime.is_file(),
            "model_calls": 0}


def smoke() -> dict:
    sys.path.insert(0, str(STUDY))
    from engine import FaultTools, METRICS, OracleController, SCENARIOS, run_episode

    class SyntheticTools:
        def run(self, action):
            return {"status": "ok", "metrics": dict.fromkeys(METRICS, 0), "error_code": None}

    outcomes = {}
    for scenario in SCENARIOS:
        result = run_episode(OracleController(), FaultTools(SyntheticTools(), scenario))
        if result["outcome"] not in {"completed", "correct_abort"}:
            raise ValueError("synthetic_workflow_failed:" + scenario)
        if not all(event["compliant"] for event in result["events"]):
            raise ValueError("synthetic_action_noncompliance:" + scenario)
        outcomes[scenario] = result["outcome"]
    return {"synthetic_smoke_passed": True, "outcomes": outcomes, "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.check and not args.synthetic_smoke:
        parser.error("choose --check and/or --synthetic-smoke")
    if args.check:
        print(json.dumps(check(), sort_keys=True))
    if args.synthetic_smoke:
        print(json.dumps(smoke(), sort_keys=True))
