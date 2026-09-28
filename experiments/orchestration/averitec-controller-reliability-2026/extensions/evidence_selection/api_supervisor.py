"""One-shot local API supervision; credentials never leave the local process.

The exclusive supervisor directory is a durable dispatch intent, including
when process creation has an unknown outcome. Never resume or retry it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

KEYS = {"jev": "TYPESAFE_API_KEY", "gemini": "GEMINI_API_KEY"}
WALL_SECONDS = 14400


def credential(name: str, path: Path) -> str:
    value = os.environ.get(name)
    if value:
        return value
    for line in path.read_text().splitlines():
        left, sep, right = line.strip().partition("=")
        if sep and left.removeprefix("export ").strip() == name:
            value = right.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value and not any(char in value for char in "\r\n"):
                return value
    raise ValueError("required_local_credential_missing")


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def supervise(controller: str, source: Path, inputs: Path, root: Path, credentials: Path) -> int:
    manifest = json.loads((inputs / "launch-manifest.json").read_text())
    if manifest.get("schema") not in {"averitec-selector-launch/v1", "averitec-selector-launch/v2"}:
        raise ValueError("unsupported_selector_manifest")
    if controller not in manifest.get("controllers", {}):
        raise ValueError("controller_not_in_launch_manifest")
    if manifest["schema"] == "averitec-selector-launch/v2" and controller != "gemini":
        raise ValueError("v2_only_gemini_api_authorized")
    policy = manifest["controllers"][controller]
    if policy.get("wall_seconds") != WALL_SECONDS or policy.get("maximum_calls") != 1001:
        raise ValueError("api_launch_cap_mismatch")
    key_name = KEYS[controller]
    secret = credential(key_name, credentials)
    supervisor = root / (controller + ".supervisor")
    supervisor.mkdir(mode=0o700)  # Exclusive: no second dispatch under any outcome.
    runner = source / "experiments/orchestration/averitec-controller-reliability-2026/extensions/evidence_selection/run_selector.py"
    output = root / controller
    if output.exists():
        raise ValueError("selector_output_already_exists")
    command = [sys.executable, str(runner), "--execute", "--controller", controller,
               "--candidates", str(inputs / "candidates.json"), "--launch-manifest",
               str(inputs / "launch-manifest.json"), "--output", str(output)]
    started = time.time()
    write_new(supervisor / "dispatch-intent.json", {"controller": controller, "started_unix": started,
        "supervisor_pid": os.getpid(), "wall_seconds": WALL_SECONDS, "maximum_calls": 1001,
        "launch_schema": manifest["schema"], "profile": policy["profile"], "retries": 0})
    env = {name: value for name, value in os.environ.items() if name in {"PATH", "HOME", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"}}
    env.update({key_name: secret, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    with (supervisor / "process.log").open("xb") as log:
        child = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=log, start_new_session=True)
        write_new(supervisor / "process.json", {"pid": child.pid, "supervisor_pid": os.getpid()})
        timed_out = False
        try:
            code = child.wait(timeout=WALL_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    write_new(supervisor / "terminal.json", {"controller": controller, "exit_code": code,
        "timed_out": timed_out, "elapsed_seconds": time.time() - started,
        "receipt_verification_required": True, "retries": 0})
    return code if code >= 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=tuple(KEYS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        status = supervise(args.controller, args.source.resolve(), args.inputs.resolve(), args.root.resolve(), args.credentials_file.resolve())
    except Exception:
        print("api_supervision_failed_no_retry", file=sys.stderr)
        status = 2
    raise SystemExit(status)
