"""Prepare a hash-bound, one-job GPU launch plan; no submission or model call."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
THREE = HERE.parent
ROOT = HERE.parents[5]
IMAGE_SHA256 = "88e35b0554e3d1da16dfe8b9944e866adde00566935b69b366acf83d32678c08"
SOURCE_FILES = (
    "experiments/orchestration/averitec-controller-reliability-2026/support/inference/contracts.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/downstream.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/verdict_runner.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/recovery_jev_selector/hero_scope_run.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/recovery_jev_selector/hero_gpu_plan.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/gpu/download_hero_assets.py",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/gpu/run_hero_jev_followup.sh",
    "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict/gpu/hero_jev_followup.slurm",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    inputs, source = bundle / "inputs", bundle / "source"
    scope_file, manifest_file = inputs / "scope.json", inputs / "verdict-manifest.json"
    original_file = inputs / "original-hero-gpu-launch.json"
    scope, manifest, original = (json.loads(p.read_text()) for p in (scope_file, manifest_file, original_file))
    if (scope.get("schema") != "averitec-jev-recovery-hero-scope/v1"
            or scope.get("planned_cases") != 100 or scope.get("planned_model_calls") != 101
            or scope.get("raw_slot") != {"arm": "jev", "pass": 4}
            or scope["new_packages"]["sha256"] != sha(inputs / "packages.json")
            or scope["new_verdict_manifest"]["sha256"] != sha(manifest_file)):
        raise ValueError("scope_binding")
    if (manifest.get("arms") != ["jev"] or manifest.get("passes") != [4]
            or manifest.get("maximum_calls") != 101 or manifest.get("retries") != 0
            or manifest.get("runner_sha256") != sha(THREE / "verdict_runner.py")):
        raise ValueError("verdict_policy")
    if original.get("schema") != "three-pass-hero-gpu-full/v1" or original.get("model") != manifest["model"] or original.get("revision") != manifest["revision"]:
        raise ValueError("original_launch_policy")
    if sha(args.image) != IMAGE_SHA256 or original.get("image_sha256") != IMAGE_SHA256:
        raise ValueError("image_hash")
    if any(sha(inputs / name) != digest for name, digest in original["small_files_sha256"].items()):
        raise ValueError("model_small_file_hash")
    staged = source / "experiments/orchestration/averitec-controller-reliability-2026/extensions/three_pass_verdict"
    commit_file = source / "SOURCE_COMMIT"
    source_commit = commit_file.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("staged_source_commit")
    source_files = {name: sha(source / name) for name in SOURCE_FILES}
    if any(sha(ROOT / name) != digest for name, digest in source_files.items()):
        raise ValueError("staged_source_hash")
    if not (staged / "verdict_runner.py").is_file():
        raise ValueError("staged_source_missing")
    if (sha(staged / "verdict_runner.py") != sha(THREE / "verdict_runner.py")
            or sha(staged / "gpu/run_hero_jev_followup.sh") != sha(THREE / "gpu/run_hero_jev_followup.sh")
            or sha(staged / "recovery_jev_selector/hero_scope_run.py") != sha(HERE / "hero_scope_run.py")):
        raise ValueError("staged_source_hash")
    plan = {"schema": "three-pass-hero-jev-followup-gpu/v1", "model": manifest["model"],
            "revision": manifest["revision"], "scope_sha256": sha(scope_file),
            "verdict_manifest_sha256": sha(manifest_file),
            "packages_sha256": sha(inputs / "packages.json"),
            "candidates_sha256": sha(inputs / "candidates.json"),
            "original_launch_sha256": sha(original_file), "image_sha256": IMAGE_SHA256,
            "source_commit": source_commit, "source_commit_file_sha256": sha(commit_file),
            "source_files_sha256": source_files,
            "runner_sha256": sha(staged / "verdict_runner.py"),
            "wrapper_sha256": sha(staged / "gpu/run_hero_jev_followup.sh"),
            "scope_runner_sha256": sha(staged / "recovery_jev_selector/hero_scope_run.py"),
            "gpu_allocation_seconds": 1800, "maximum_gpu_jobs": 1,
            "maximum_model_calls": 101, "retries": 0, "download_seconds": 870,
            "startup_seconds": 330, "runner_seconds": 500,
            "output": "hero-jev-recovery-pass-4"}
    target = inputs / "hero-jev-followup-launch.json"
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(plan, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"launch": str(target), "launch_sha256": sha(target), "model_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
