"""Offline exact tokenizer preflight for the three local HTTP selectors.

Only pinned tokenizer assets are downloaded. Adapter transports are replaced
by selector_passes._capture_http_requests before any request can leave the host.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import selector_passes as selector

ARMS = {"qwen", "lfm", "lfm26"}
ASSET_MANIFEST = selector.EXPERIMENT / "manifests/model-assets-expansion-20260919.json"
FILES = {
    "qwen": ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "vocab.json", "merges.txt"),
    "lfm": ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "special_tokens_map.json"),
    "lfm26": ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def pinned_assets(arm: str, *, manifest: Path = ASSET_MANIFEST) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("local_preflight_arm")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("schema") != "averitec-model-assets/v1":
        raise ValueError("local_preflight_asset_manifest")
    model = selector.ARMS[arm][0]
    entries = [item for item in metadata["models"] if item.get("repository") == model]
    if len(entries) != 1 or len(entries[0].get("revision", "")) != 40:
        raise ValueError("local_preflight_model_revision")
    return entries[0]


def resolve_assets(arm: str, *, fetch: bool = False, manifest: Path = ASSET_MANIFEST) -> tuple[Path, dict[str, Any]]:
    """Resolve pinned HF snapshot, verifying every file before tokenization."""
    from huggingface_hub import hf_hub_download

    pin = pinned_assets(arm, manifest=manifest)
    listed = {entry["path"]: entry for entry in pin["files"]}
    required = FILES[arm]
    if any(name not in listed for name in required):
        raise ValueError("local_preflight_missing_pin")
    paths: list[Path] = []
    hashes: dict[str, str] = {}
    for name in required:
        try:
            path = Path(hf_hub_download(repo_id=pin["repository"], filename=name,
                                        revision=pin["revision"], local_files_only=not fetch))
        except Exception as exc:
            raise ValueError(f"local_preflight_asset_unavailable:{arm}:{name}") from exc
        raw = path.read_bytes()
        expected = listed[name]
        actual_blob = _git_blob_sha1(raw) if len(expected["blob"]) == 40 else _sha256(raw)
        if len(raw) != expected["size"] or actual_blob != expected["blob"]:
            raise ValueError(f"local_preflight_asset_drift:{arm}:{name}")
        paths.append(path)
        hashes[name] = _sha256(raw)
    roots = {path.parent.resolve() for path in paths}
    if len(roots) != 1:
        raise ValueError("local_preflight_split_snapshot")
    return roots.pop(), {"repository": pin["repository"], "revision": pin["revision"],
                         "asset_sha256": hashes, "asset_manifest_sha256": selector.digest(manifest)}


def count_request(tokenizer: Any, arm: str, request: dict[str, Any]) -> int:
    if request.get("model") != selector.ARMS[arm][0] or not isinstance(request.get("messages"), list):
        raise ValueError("local_preflight_request_shape")
    options = request.get("chat_template_kwargs", {})
    if arm == "qwen":
        if options != {"enable_thinking": False}:
            raise ValueError("local_preflight_qwen_template_options")
    elif options:
        raise ValueError("local_preflight_template_options")
    encoded = tokenizer.apply_chat_template(request["messages"], tokenize=True,
                                            add_generation_prompt=True, truncation=False, **options)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
        raise ValueError("local_preflight_tokenizer_result")
    return len(ids)


def build_report(arm: str, candidates: Path, *, fetch: bool = False,
                 manifest: Path = ASSET_MANIFEST) -> tuple[dict[str, Any], dict[str, Any]]:
    from transformers import AutoTokenizer, __version__ as transformers_version
    import tokenizers

    snapshot, identity = resolve_assets(arm, fetch=fetch, manifest=manifest)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
                                               use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("local_preflight_not_fast_tokenizer")
    planned = selector.previous.plan_observations(candidates)
    observations = [selector.WARMUP] + [item["observation"] for item in planned]
    instructions = (selector.EVIDENCE / "selector_instructions.txt").read_text(encoding="utf-8")
    adapter = selector._make_selector(arm, endpoint="http://127.0.0.1:1/v1", laya_checkpoint=None, keys={})
    requests = selector._capture_http_requests(arm, adapter, observations, instructions)
    if len(requests) != 1001 or len(observations) != 1001:
        raise ValueError("local_preflight_request_count")
    limits = {request.get("max_tokens") for request in requests}
    if len(limits) != 1 or next(iter(limits)) not in {160, 4096}:
        raise ValueError("local_preflight_output_limit")
    output_limit = next(iter(limits))
    identity.update({"transformers_version": transformers_version, "tokenizers_version": tokenizers.__version__,
                     "counter_code_sha256": selector.digest(Path(__file__)),
                     "adapter_code_sha256": selector.digest(selector.HERE / "selector_passes.py"),
                     "candidates_sha256": selector.digest(candidates),
                     "instructions_sha256": selector.digest(selector.EVIDENCE / "selector_instructions.txt"),
                     "count_method": "AutoTokenizer.apply_chat_template(tokenize=True,add_generation_prompt=True,truncation=False)"})
    method_hash = _sha256(selector.canonical(identity).encode())
    records = []
    for observation, request in zip(observations, requests):
        wire = selector.wire_bytes(request)
        records.append({"observation_sha256": _sha256(selector.canonical(observation).encode()),
                        "request_sha256": _sha256(wire), "serialized_bytes": len(wire),
                        "input_tokens": count_request(tokenizer, arm, request)})
    report = {"schema": "averitec-selector-token-preflight/v2", "arm": arm,
              "model": selector.ARMS[arm][0], "profile": selector.ARMS[arm][1],
              "method": "exact_pinned_tokenizer", "method_identity_sha256": method_hash,
              "context_limit_tokens": 8192, "output_request_limit_tokens": output_limit,
              "output_limit_enforced": True, "input_margin_tokens": 0, "records": records}
    if any(row["input_tokens"] + output_limit > 8192 for row in records):
        raise ValueError("local_preflight_context_exceeded")
    return report, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(ARMS))
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Download tokenizer files only at pinned revision")
    args = parser.parse_args()
    if args.report.exists() or args.identity.exists():
        parser.error("report and identity paths must be new")
    report, identity = build_report(args.arm, args.candidates, fetch=args.fetch)
    if not args.report.parent.is_dir() or not args.identity.parent.is_dir():
        parser.error("output parent directory must already exist")
    args.report.write_text(selector.canonical(report) + "\n", encoding="utf-8")
    args.identity.write_text(selector.canonical(identity) + "\n", encoding="utf-8")
    print(selector.canonical({"arm": args.arm, "records": len(report["records"]),
                             "min_input_tokens": min(r["input_tokens"] for r in report["records"]),
                             "max_input_tokens": max(r["input_tokens"] for r in report["records"]),
                             "sum_input_tokens": sum(r["input_tokens"] for r in report["records"]),
                             "report_sha256": selector.digest(args.report),
                             "identity_sha256": selector.digest(args.identity)}))


if __name__ == "__main__":
    main()
