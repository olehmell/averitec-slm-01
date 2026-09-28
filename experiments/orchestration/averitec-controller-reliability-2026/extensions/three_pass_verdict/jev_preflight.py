"""Offline Jev request inventory with explicitly estimated token reserves.

This captures the original adapter's 1,001 exact serialized request bodies
without network access. Byte lengths are observed; token reserves are an
assumption, since TypeSafe exposes no countTokens or output-limit endpoint.
The output is not a certificate of lossless provider context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import selector_passes as runner


def build(candidates: Path, openapi_snapshot: Path) -> dict:
    contract = json.loads(openapi_snapshot.read_text(encoding="utf-8"))
    paths = contract.get("paths", {})
    request_schema = contract.get("components", {}).get("schemas", {}).get("SystemOneRequest", {})
    properties = request_schema.get("properties", {})
    if ("/v1/systemone" not in paths or set(properties) != {"model", "state", "questions"}
            or any("counttokens" in path.lower() for path in paths)):
        raise ValueError("jev_provider_contract_changed_review_required")
    measured = runner.previous.plan_observations(candidates)
    observations = [runner.WARMUP, *(row["observation"] for row in measured)]
    selector = runner._make_selector("jev", endpoint=None, laya_checkpoint=None, keys={})
    instructions = (runner.EVIDENCE / "selector_instructions.txt").read_text(encoding="utf-8")
    requests = runner._capture_http_requests("jev", selector, observations, instructions)
    records = []
    for observation, request in zip(observations, requests):
        serialized = runner.wire_bytes(request)
        if len(serialized) > 2048:
            raise ValueError("jev_request_exceeds_frozen_byte_limit")
        records.append({"observation_sha256": hashlib.sha256(runner.canonical(observation).encode()).hexdigest(),
                        "request_sha256": hashlib.sha256(serialized).hexdigest(),
                        "serialized_bytes": len(serialized),
                        "input_tokens": len(serialized) + 1024})
    if len(records) != 1001:
        raise ValueError("jev_request_count")
    return {"schema": "averitec-selector-token-preflight/v2", "arm": "jev", "model": "jev-1.13.0",
            "profile": "baseline", "method": "estimated_wire_bytes_plus_1024",
            "method_identity_sha256": runner.digest(openapi_snapshot),
            "context_limit_tokens": 32768, "output_request_limit_tokens": 1024,
            "output_limit_enforced": False, "input_margin_tokens": 0, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--openapi-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.candidates, args.openapi_snapshot)
    body = (runner.canonical(value) + "\n").encode()
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"requests": len(value["records"]),
                      "maximum_serialized_bytes": max(row["serialized_bytes"] for row in value["records"]),
                      "total_serialized_bytes": sum(row["serialized_bytes"] for row in value["records"]),
                      "report_sha256": hashlib.sha256(body).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
