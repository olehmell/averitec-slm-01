"""Export a minimal committed, label-free source snapshot. No model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EXPERIMENT = "experiments/orchestration/averitec-controller-reliability-2026"
EXTENSION = EXPERIMENT + "/extensions/evidence_selection"
DEPENDENCIES = [EXPERIMENT + "/" + name for name in (
    "providers.py", "generation_profiles.py", "gemini_provider.py", "tracing.py", "config.yaml",
    "gpu/vega/check_controller_assets.py", "gpu/vega/check_controller_runtime.py",
    "manifests/model-assets-expansion-20260919.json",
)] + ["experiments/orchestration/averitec-controller-reliability-2026/support/gpu/vega/check_model_cache.py"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(destination: Path, candidates: Path, reference: Path, *, protocol: str = "v1", baseline_root: Path | None = None) -> dict:
    from run_selector import CONTROLLERS, CONTROLLERS_V2
    if protocol not in {"v1", "v2"}:
        raise ValueError("unknown_selector_protocol")
    controllers = CONTROLLERS_V2 if protocol == "v2" else CONTROLLERS
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", EXTENSION], cwd=ROOT, text=True).splitlines()
    if EXTENSION + "/run_selector.py" not in names:
        raise ValueError("selector_not_committed")
    names = sorted(set(names + DEPENDENCIES))
    blobs = {}
    for name in names:
        blob = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
        if (ROOT / name).read_bytes() != blob:
            raise ValueError("committed_source_worktree_drift:" + name)
        blobs[name] = blob
    candidate_bytes = candidates.read_bytes()
    reference_hash = digest(reference.read_bytes())
    if digest(candidate_bytes) != "219c745aac6f50501a55c4f2511d383505317fd75338b0efaed25f44bbe23b98":
        raise ValueError("candidate_binding_changed")
    if reference_hash != "1d4422de9dc1975683648a1b389d2bebf323a86f70d5e01fde9c4cccf9657604":
        raise ValueError("reference_binding_changed")
    manifest = {"schema": "averitec-selector-launch/" + protocol, "source_commit": commit,
        "candidates_sha256": digest(candidate_bytes), "reference_sha256": reference_hash,
        "instructions_sha256": digest(blobs[EXTENSION + "/selector_instructions.txt"]),
        "code_sha256": {name: digest(blob) for name, blob in blobs.items()},
        "controllers": {name: {"model": model, "profile": profile, "measured_calls": 1000,
            "warmup_calls": 1, "maximum_calls": 1001, "wall_seconds": wall}
            for name, (model, profile, wall) in controllers.items()},
        "max_gpu_seconds": 10800, "retries": 0, "approved_date": "2026-09-20"}
    comparison = None
    if protocol == "v2":
        if baseline_root is None:
            raise ValueError("v2_requires_completed_baseline_root")
        previous = json.loads((baseline_root / "inputs/launch-manifest.json").read_text())
        for key in ("candidates_sha256", "reference_sha256", "instructions_sha256"):
            if previous.get(key) != manifest[key]:
                raise ValueError("comparison_input_binding_mismatch:" + key)
        comparison = {"schema": "averitec-selector-comparison-selection/v1",
            "selection_rule": "fixed_whole_arms_before_v2_inference_no_best_of_or_row_merging",
            "baseline_root": str(baseline_root.resolve()), "replacement_controllers": ["gemini", "qwen", "lfm26"],
            "baseline_receipts": {}}
        for name in ("jev", "lfm", "laya_typed"):
            output = baseline_root / "outputs" / name
            receipt_bytes = (output / "receipt.json").read_bytes()
            receipt = json.loads(receipt_bytes)
            if receipt.get("complete") is not True or receipt.get("controller") != name:
                raise ValueError("comparison_baseline_incomplete")
            if set(receipt.get("journal_sha256", {})) != {"run-metadata.json", "intents.jsonl", "results.jsonl", "traces.jsonl"}:
                raise ValueError("comparison_baseline_journal_set")
            for file, expected in receipt["journal_sha256"].items():
                if file not in {"run-metadata.json", "intents.jsonl", "results.jsonl", "traces.jsonl"}:
                    raise ValueError("comparison_baseline_journal_path")
                if digest((output / file).read_bytes()) != expected:
                    raise ValueError("comparison_baseline_journal_drift")
            comparison["baseline_receipts"][name] = {"receipt_sha256": digest(receipt_bytes),
                "journal_sha256": receipt["journal_sha256"], "planned_ids_sha256": receipt["planned_ids_sha256"]}
    destination.mkdir(mode=0o700)  # Never merge or overwrite an older launch.
    source = destination / "source"
    for name, blob in blobs.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    (source / "SOURCE_COMMIT").write_text(commit + "\n")
    inputs = destination / "inputs"
    inputs.mkdir()
    (inputs / "candidates.json").write_bytes(candidate_bytes)
    (inputs / "launch-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (destination / "outputs").mkdir()
    (destination / "dispatch").mkdir()
    if comparison is not None:
        (destination / "comparison-selection.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    return {"source_commit": commit, "protocol": protocol, "code_files": len(blobs), "root": str(destination),
            "gold_staged": False, "model_calls": 0, "submitted_jobs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v1")
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination.resolve(), args.candidates, args.reference, protocol=args.protocol, baseline_root=args.baseline_root), sort_keys=True))
