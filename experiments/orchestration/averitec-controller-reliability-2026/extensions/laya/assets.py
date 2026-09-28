#!/usr/bin/env python3
"""Download and verify the exact public assets for the Laya extension.

This module deliberately uses only the Python standard library. It never imports
Laya, Transformers, Torch, or the downloaded safetensors files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SCHEMA = "averitec-laya-assets/v1"
MANIFEST = Path(__file__).with_name("model-assets.json")
REQUIRED_MODEL_FILES = {
    "encoder/config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "typed-decisions/encoder/config.json",
    "typed-decisions/model.safetensors",
    "typed-decisions/rl_agent_config.json",
    "typed-decisions/tokenizer/tokenizer.json",
    "typed-decisions/tokenizer/tokenizer_config.json",
}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


class AssetError(RuntimeError):
    """A manifest, containment, download, or digest verification failure."""


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise AssetError("asset_path_not_string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise AssetError(f"unsafe_asset_path:{value!r}")
    return value


def _inside(root: Path, relative: str) -> Path:
    """Return a contained path, rejecting symlinks in every existing component."""
    relative = _safe_relative(relative)
    root = root.expanduser().resolve()
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise AssetError(f"asset_path_symlink:{relative}")
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise AssetError(f"asset_path_escape:{relative}") from error
    return candidate


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssetError("manifest_invalid_json") from error
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise AssetError("manifest_schema")

    package = value.get("package")
    model = value.get("model")
    runtime = value.get("runtime")
    if not all(isinstance(item, dict) for item in (package, model, runtime)):
        raise AssetError("manifest_sections")
    if (
        package.get("name") != "laya"
        or package.get("version") != "0.3.3"
        or not isinstance(package.get("size"), int)
        or package["size"] <= 0
        or not _HEX64.fullmatch(str(package.get("sha256", "")))
        or _safe_relative(package.get("filename")) != package.get("filename")
        or not str(package.get("url", "")).startswith("https://files.pythonhosted.org/")
    ):
        raise AssetError("manifest_package")
    dependencies = package.get("dependencies")
    if not isinstance(dependencies, dict) or set(dependencies) != {
        "torch", "transformers", "safetensors", "huggingface_hub", "numpy"
    } or not all(isinstance(item, str) and item.startswith(">=") for item in dependencies.values()):
        raise AssetError("manifest_dependencies")

    repository = model.get("repository")
    revision = model.get("revision")
    files = model.get("files")
    if (
        not isinstance(repository, str)
        or not _REPOSITORY.fullmatch(repository)
        or not isinstance(revision, str)
        or not _HEX40.fullmatch(revision)
        or not isinstance(files, list)
    ):
        raise AssetError("manifest_model")
    names: set[str] = set()
    for row in files:
        if not isinstance(row, dict):
            raise AssetError("manifest_model_file")
        name = _safe_relative(row.get("path"))
        algorithm = row.get("digest_algorithm")
        digest = row.get("digest")
        size = row.get("size")
        if name in names or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise AssetError(f"manifest_model_file:{name}")
        if algorithm == "git-sha1":
            valid_digest = isinstance(digest, str) and _HEX40.fullmatch(digest)
        elif algorithm == "sha256" and row.get("git_lfs") is True:
            valid_digest = isinstance(digest, str) and _HEX64.fullmatch(digest)
            valid_digest = bool(valid_digest and _HEX40.fullmatch(str(row.get("git_blob", ""))))
        else:
            valid_digest = False
        if not valid_digest:
            raise AssetError(f"manifest_model_digest:{name}")
        names.add(name)
    if names != REQUIRED_MODEL_FILES:
        missing = sorted(REQUIRED_MODEL_FILES - names)
        extra = sorted(names - REQUIRED_MODEL_FILES)
        raise AssetError(f"manifest_model_members:missing={missing}:extra={extra}")
    if not _HEX64.fullmatch(str(runtime.get("base_image_sha256", ""))):
        raise AssetError("manifest_runtime_image")
    return value


def asset_paths(root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    model = manifest["model"]
    repo_dir = model["repository"].replace("/", "--")
    snapshot_rel = f"checkpoints/{repo_dir}/{model['revision']}"
    return {
        "root": root.expanduser().resolve(),
        "wheel": _inside(root, f"wheels/{manifest['package']['filename']}"),
        "snapshot": _inside(root, snapshot_rel),
        "english": _inside(root, snapshot_rel),
        "typed_decisions": _inside(root, f"{snapshot_rel}/typed-decisions"),
        "python_target": _inside(root, "runtime/python"),
    }


def _hash_file(path: Path, algorithm: str, expected_size: int) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
    elif algorithm == "git-sha1":
        digest = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
    else:
        raise AssetError(f"unsupported_digest:{algorithm}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, *, size: int, algorithm: str, digest: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise AssetError(f"asset_missing_or_not_regular:{path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise AssetError(f"asset_size:{path}:{actual_size}!={size}")
    actual_digest = _hash_file(path, algorithm, size)
    if actual_digest != digest:
        raise AssetError(f"asset_digest:{path}:{actual_digest}!={digest}")


def checkpoint_path(
    root: Path | str,
    name: str,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Return the complete local checkpoint path for ``base`` or ``typed``."""
    manifest = manifest or load_manifest()
    paths = asset_paths(Path(root), manifest)
    if name == "base":
        return paths["english"]
    if name == "typed":
        return paths["typed_decisions"]
    raise AssetError(f"unknown_checkpoint:{name}")


def verify_assets(
    root: Path | str,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every declared wheel/checkpoint member without loading weights."""
    manifest = manifest or load_manifest()
    root = Path(root)
    paths = asset_paths(root, manifest)
    package = manifest["package"]
    wheel_dir = paths["wheel"].parent
    if not wheel_dir.is_dir() or any(item.is_symlink() for item in wheel_dir.iterdir()):
        raise AssetError("wheel_directory_invalid")
    wheel_members = {item.name for item in wheel_dir.iterdir() if item.is_file()}
    if wheel_members != {package["filename"]}:
        raise AssetError(f"unexpected_wheel_members:{sorted(wheel_members)}")
    _verify_file(
        paths["wheel"], size=package["size"], algorithm="sha256", digest=package["sha256"]
    )
    checked = 1
    total = package["size"]
    snapshot_rel = paths["snapshot"].relative_to(paths["root"]).as_posix()
    if not paths["snapshot"].is_dir():
        raise AssetError("snapshot_directory_missing")
    snapshot_entries = list(paths["snapshot"].rglob("*"))
    if any(item.is_symlink() for item in snapshot_entries):
        raise AssetError("snapshot_symlink")
    actual_members = {
        item.relative_to(paths["snapshot"]).as_posix()
        for item in snapshot_entries
        if item.is_file()
    }
    if actual_members != REQUIRED_MODEL_FILES:
        raise AssetError(
            f"unexpected_snapshot_members:missing={sorted(REQUIRED_MODEL_FILES - actual_members)}:"
            f"extra={sorted(actual_members - REQUIRED_MODEL_FILES)}"
        )
    for row in manifest["model"]["files"]:
        target = _inside(paths["root"], f"{snapshot_rel}/{row['path']}")
        _verify_file(
            target,
            size=row["size"],
            algorithm=row["digest_algorithm"],
            digest=row["digest"],
        )
        checked += 1
        total += row["size"]
    return {
        "schema": "averitec-laya-assets-receipt/v1",
        "repository": manifest["model"]["repository"],
        "revision": manifest["model"]["revision"],
        "files_verified": checked,
        "bytes_verified": total,
        "wheel": paths["wheel"].relative_to(paths["root"]).as_posix(),
        "english_checkpoint": paths["english"].relative_to(paths["root"]).as_posix(),
        "typed_decisions_checkpoint": paths["typed_decisions"].relative_to(paths["root"]).as_posix(),
    }


def _copy_stream(source: BinaryIO, target: BinaryIO) -> None:
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            return
        target.write(chunk)


def _download(
    url: str,
    target: Path,
    expected_size: int,
    timeout: float,
    algorithm: str,
    digest: str,
) -> None:
    """Download to a retained partial and atomically publish only a complete file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    restart = target.with_name(target.name + ".partial.restart")
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise AssetError(f"asset_target_not_regular:{target}")
        return
    if partial.is_symlink() or restart.is_symlink():
        raise AssetError(f"asset_partial_symlink:{target}")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > expected_size:
        raise AssetError(f"asset_partial_oversize:{partial}")
    if offset == expected_size:
        _verify_file(partial, size=expected_size, algorithm=algorithm, digest=digest)
        os.replace(partial, target)
        return
    headers = {"User-Agent": "averitec-laya-assets/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            if offset and status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise AssetError(f"asset_bad_content_range:{target}")
                output = partial
                mode = "ab"
            elif status == 200:
                output = restart if offset else partial
                mode = "wb"
            else:
                raise AssetError(f"asset_http_status:{target}:{status}")
            with output.open(mode) as handle:
                _copy_stream(response, handle)
                handle.flush()
                os.fsync(handle.fileno())
            if output == restart:
                os.replace(restart, partial)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise AssetError(f"asset_download_failed:{target}:{error}") from error
    if partial.stat().st_size != expected_size:
        raise AssetError(f"asset_download_size:{target}:{partial.stat().st_size}!={expected_size}")
    _verify_file(partial, size=expected_size, algorithm=algorithm, digest=digest)
    os.replace(partial, target)


def prepare_assets(root: Path, manifest: dict[str, Any], timeout: float) -> dict[str, Any]:
    paths = asset_paths(root, manifest)
    root = paths["root"]
    root.mkdir(parents=True, exist_ok=True)
    package = manifest["package"]
    _download(
        package["url"], paths["wheel"], package["size"], timeout, "sha256", package["sha256"]
    )
    try:
        _verify_file(
            paths["wheel"], size=package["size"], algorithm="sha256", digest=package["sha256"]
        )
    except AssetError:
        # Never silently replace a published but corrupt destination.
        raise

    model = manifest["model"]
    snapshot_rel = paths["snapshot"].relative_to(root).as_posix()
    for row in model["files"]:
        target = _inside(root, f"{snapshot_rel}/{row['path']}")
        url = (
            f"https://huggingface.co/{model['repository']}/resolve/"
            f"{model['revision']}/{quote(row['path'], safe='/')}"
        )
        _download(
            url, target, row["size"], timeout, row["digest_algorithm"], row["digest"]
        )
        _verify_file(
            target,
            size=row["size"],
            algorithm=row["digest_algorithm"],
            digest=row["digest"],
        )
    return verify_assets(root, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-manifest")
    verify = sub.add_parser("verify-assets")
    verify.add_argument("--assets-root", required=True, type=Path)
    prepare = sub.add_parser("prepare-assets")
    prepare.add_argument("--assets-root", required=True, type=Path)
    prepare.add_argument("--timeout-seconds", type=float, default=30.0)
    paths = sub.add_parser("paths")
    paths.add_argument("--assets-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate-manifest":
            result: dict[str, Any] = {
                "schema": manifest["schema"],
                "repository": manifest["model"]["repository"],
                "revision": manifest["model"]["revision"],
                "model_files": len(manifest["model"]["files"]),
            }
        elif args.command == "verify-assets":
            result = verify_assets(args.assets_root, manifest)
        elif args.command == "prepare-assets":
            if not 1.0 <= args.timeout_seconds <= 120.0:
                raise AssetError("timeout_seconds_out_of_range")
            result = prepare_assets(args.assets_root, manifest, args.timeout_seconds)
        else:
            result = {key: str(value) for key, value in asset_paths(args.assets_root, manifest).items()}
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except AssetError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
