"""Hash-bound, no-retry runner for one shared HerO direct-snippet verdict matrix.

Plan-only is offline. Execution requires an explicit endpoint and writes each
intent durably before a request. A used output directory is never resumed.
Gold and reference Q/A have no input path in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


HERE = Path(__file__).resolve().parent
_SIBLING = HERE.parents[1] / "support"
if str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))
from inference.contracts import ContractError, parse_hero_terminal_output  # noqa: E402


SCHEMA = "three-pass-verdict-launch/v1"
PACKAGE_SCHEMA = "three-pass-selected-passages/v1"
MODEL = "humane-lab/Meta-Llama-3.1-8B-HerO"
REVISION = "42a6f7120f44eae5ce3ffde72cf74a8ea6620e74"
SELECTORS = ("jev", "lfm", "laya_typed", "gemini", "qwen", "lfm26", "jeff")
CONTROLS = ("include_all", "include_none")
PROMPT_PREFIX = (
    "Use only the claim and numbered source snippets. Give a non-empty brief justification, "
    "then terminate with exactly one line: Verdict: Supported | Refuted | Not Enough Evidence | "
    "Conflicting Evidence/Cherrypicking.\n"
)
WARMUP_CLAIM = "Synthetic plumbing check: the supplied snippet says the lamp is blue."
WARMUP_SNIPPET = "The lamp is blue."
PASSAGE_FIELDS = ("id", "passage_id", "text", "url", "source_start", "source_text_length", "source_text_sha256")
PACKAGE_FIELDS = {"schema", "case_id", "group_id", "claim", "mode", "status", "errors", "selected_passages", "arm", "pass"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_astra_gate(gate_file: Path | None, manifest_file: Path) -> None:
    if gate_file is None:
        raise ValueError("verdict_astra_gate_required")
    try:
        review = _json(gate_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("verdict_astra_gate") from None
    expected = {"schema": "averitec-astra-launch-gate/v1", "manifest_sha256": sha256(manifest_file),
                "decision": "approved", "reviewer": "gpt-6-astra"}
    if review != expected:
        raise ValueError("verdict_astra_gate")


def render_prompt(claim: str, passages: list[dict]) -> str:
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("verdict_claim")
    snippets = "\n".join(f"{index}. {json.dumps(row['text'], ensure_ascii=False)}" for index, row in enumerate(passages, 1))
    return PROMPT_PREFIX + "Claim: " + claim + "\nSource snippets:\n" + (snippets or "(none)")


def _validate_manifest(manifest: dict, candidates: Path, packages: Path, tokenizer_dir: Path,
                       *, arms: tuple[str, ...], passes: tuple[int, ...], expected_cases: int) -> None:
    keys = {"schema", "candidates_sha256", "packages_sha256", "runner_sha256", "downstream_sha256", "contracts_sha256",
            "tokenizer_files_sha256", "model", "revision", "returned_model", "prompt_sha256",
            "arms", "passes", "cases", "context_tokens", "max_output_tokens", "maximum_total_input_tokens",
            "maximum_total_output_tokens", "input_margin_tokens", "endpoint", "timeout_seconds", "warmup_calls", "maximum_calls", "temperature", "retries"}
    if not isinstance(manifest, dict) or set(manifest) != keys or manifest.get("schema") != SCHEMA:
        raise ValueError("verdict_manifest_schema")
    if (manifest["model"] != MODEL or manifest["revision"] != REVISION or manifest["returned_model"] != MODEL
            or manifest["temperature"] != 0 or manifest["retries"] != 0 or manifest["warmup_calls"] != 1
            or manifest["arms"] != list(arms) or manifest["passes"] != list(passes)
            or manifest["cases"] != expected_cases or manifest["maximum_calls"] != len(arms) * len(passes) * expected_cases + 1):
        raise ValueError("verdict_manifest_policy")
    numeric = ("context_tokens", "max_output_tokens", "maximum_total_input_tokens", "maximum_total_output_tokens", "timeout_seconds")
    if any(type(manifest[key]) is not int or manifest[key] < 1 for key in numeric):
        raise ValueError("verdict_manifest_caps")
    if type(manifest["input_margin_tokens"]) is not int or not 0 <= manifest["input_margin_tokens"] <= 4096:
        raise ValueError("verdict_manifest_margin")
    endpoint = urlsplit(manifest["endpoint"]) if isinstance(manifest["endpoint"], str) else None
    if (endpoint is None or endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1"
            or endpoint.port is None or endpoint.path != "/v1/chat/completions"
            or endpoint.username is not None or endpoint.password is not None
            or endpoint.query or endpoint.fragment):
        raise ValueError("verdict_manifest_endpoint")
    if manifest["max_output_tokens"] >= manifest["context_tokens"]:
        raise ValueError("verdict_manifest_context")
    bindings = {"candidates_sha256": sha256(candidates), "packages_sha256": sha256(packages),
                "runner_sha256": sha256(Path(__file__)), "downstream_sha256": sha256(HERE / "downstream.py"),
                "contracts_sha256": sha256(_SIBLING / "inference/contracts.py"),
                "prompt_sha256": hashlib.sha256(PROMPT_PREFIX.encode()).hexdigest()}
    if any(manifest[key] != value for key, value in bindings.items()):
        raise ValueError("verdict_manifest_hash_drift")
    files = manifest["tokenizer_files_sha256"]
    if not isinstance(files, dict) or set(files) != {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
        raise ValueError("verdict_tokenizer_binding")
    if any(sha256(tokenizer_dir / name) != digest for name, digest in files.items()):
        raise ValueError("verdict_tokenizer_hash_drift")


def _load_packages(candidates_file: Path, packages_file: Path, arms: tuple[str, ...],
                   passes: tuple[int, ...], expected_cases: int) -> list[dict]:
    candidate_data, package_data = _json(candidates_file), _json(packages_file)
    cases = candidate_data.get("cases") if isinstance(candidate_data, dict) and candidate_data.get("gold_included") is False else None
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError("verdict_candidate_shape_or_gold")
    by_case = {row["case_id"]: row for row in cases}
    if len(by_case) != expected_cases:
        raise ValueError("verdict_duplicate_candidate_case")
    packages = package_data.get("packages") if isinstance(package_data, dict) and package_data.get("schema") == "three-pass-verdict-packages/v1" else None
    if not isinstance(packages, list):
        raise ValueError("verdict_package_file_schema")
    expected = {(arm, number, case_id) for arm in arms for number in passes for case_id in by_case}
    seen: set[tuple[str, int, str]] = set()
    ordered = []
    for package in packages:
        if not isinstance(package, dict) or set(package) != PACKAGE_FIELDS or package.get("schema") != PACKAGE_SCHEMA:
            raise ValueError("verdict_package_schema_or_gold")
        arm, number, case_id = package["arm"], package["pass"], package["case_id"]
        if not isinstance(arm, str) or type(number) is not int or not isinstance(case_id, str):
            raise ValueError("verdict_package_key")
        key = (arm, number, case_id)
        if key not in expected or key in seen:
            raise ValueError("verdict_package_unknown_or_duplicate")
        seen.add(key)
        frozen = by_case[case_id]
        mode = "selector" if arm in SELECTORS else arm
        if (package["mode"] != mode or package["group_id"] != frozen["group_id"]
                or package["claim"] != frozen["claim"]):
            raise ValueError("verdict_package_identity")
        source = frozen["candidates"]
        if not isinstance(source, list) or len(source) != 10:
            raise ValueError("verdict_candidate_count")
        selected = package["selected_passages"]
        if package["status"] == "failed":
            if arm in CONTROLS or selected is not None or not isinstance(package["errors"], list) or not package["errors"]:
                raise ValueError("verdict_package_failure_shape")
        elif package["status"] == "ready":
            if package["errors"] != [] or not isinstance(selected, list):
                raise ValueError("verdict_package_ready_shape")
            expected_rows = source if arm == "include_all" else [] if arm == "include_none" else source
            positions = {row["id"]: index for index, row in enumerate(expected_rows)}
            if len(positions) != len(expected_rows):
                raise ValueError("verdict_candidate_ids")
            previous = -1
            for row in selected:
                if not isinstance(row, dict) or set(row) != set(PASSAGE_FIELDS) or row.get("id") not in positions:
                    raise ValueError("verdict_selected_identity")
                index = positions[row["id"]]
                if index <= previous or row != {field: expected_rows[index][field] for field in PASSAGE_FIELDS}:
                    raise ValueError("verdict_selected_order_or_source")
                previous = index
            if arm == "include_all" and len(selected) != len(source) or arm == "include_none" and selected:
                raise ValueError("verdict_control_package")
        else:
            raise ValueError("verdict_package_status")
        ordered.append(package)
    if seen != expected:
        raise ValueError(f"verdict_package_missing:{len(expected - seen)}")
    order = {case_id: index for index, case_id in enumerate(by_case)}
    arm_order = {arm: index for index, arm in enumerate(arms)}
    return sorted(ordered, key=lambda row: (row["pass"], arm_order[row["arm"]], order[row["case_id"]]))


def _local_token_counter(tokenizer_dir: Path) -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ValueError("transformers_required_for_exact_preflight") from error
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    def count(prompt: str) -> int:
        encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                add_generation_prompt=True, truncation=False)
        ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("verdict_tokenizer_count_shape")
        return len(ids)
    return count


def prepare(manifest_file: Path, candidates_file: Path, packages_file: Path, tokenizer_dir: Path,
            *, arms: tuple[str, ...] = (*SELECTORS, *CONTROLS), passes: tuple[int, ...] = (1, 2, 3),
            expected_cases: int = 100, count_tokens: Callable[[str], int] | None = None) -> tuple[dict, list[dict], dict]:
    """Hash check and preflight all planned prompts, with no model calls."""
    manifest = _json(manifest_file)
    _validate_manifest(manifest, candidates_file, packages_file, tokenizer_dir,
                       arms=arms, passes=passes, expected_cases=expected_cases)
    packages = _load_packages(candidates_file, packages_file, arms, passes, expected_cases)
    counter = count_tokens or _local_token_counter(tokenizer_dir)
    warmup_prompt = render_prompt(WARMUP_CLAIM, [{"text": WARMUP_SNIPPET}])
    prompts = [(None, warmup_prompt)] + [(package, render_prompt(package["claim"], package["selected_passages"]))
                                          for package in packages if package["status"] == "ready"]
    counts = []
    for _, prompt in prompts:
        amount = counter(prompt)
        if (type(amount) is not int or amount < 1
                or amount + manifest["input_margin_tokens"] + manifest["max_output_tokens"] > manifest["context_tokens"]):
            raise ValueError("verdict_context_preflight")
        counts.append(amount)
    reserved_input = sum(counts) + len(counts) * manifest["input_margin_tokens"]
    if (reserved_input > manifest["maximum_total_input_tokens"]
            or len(prompts) * manifest["max_output_tokens"] > manifest["maximum_total_output_tokens"]):
        raise ValueError("verdict_token_budget_preflight")
    summary = {"schema": "three-pass-verdict-plan/v1", "packages": len(packages),
               "ready_packages": len(prompts) - 1, "failed_packages": len(packages) - len(prompts) + 1,
               "warmup_calls": 1, "maximum_actual_calls": len(prompts), "maximum_prompt_tokens": max(counts),
               "planned_input_tokens": sum(counts), "reserved_input_tokens": reserved_input,
               "input_margin_tokens": manifest["input_margin_tokens"], "context_preflight": "passed",
               "manifest_sha256": sha256(manifest_file), "model_calls": 0}
    return manifest, packages, summary


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _http_request(endpoint: str, model: str, timeout: int) -> Callable[[str, int], dict]:
    opener = build_opener(_NoRedirect())
    def call(prompt: str, maximum: int) -> dict:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": maximum}
        request = Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with opener.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list) or len(payload["choices"]) != 1:
            raise ValueError("verdict_transport_schema")
        choice = payload["choices"][0]
        usage = payload.get("usage")
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict) or not isinstance(usage, dict):
            raise ValueError("verdict_transport_schema")
        return {"model": payload.get("model"), "content": choice["message"].get("content"),
                "finish_reason": choice.get("finish_reason"), "usage": usage}
    return call


def execute(manifest_file: Path, candidates_file: Path, packages_file: Path, tokenizer_dir: Path,
            output_dir: Path, *, astra_gate: Path | None = None, endpoint: str | None = None,
            request_fn: Callable[[str, int], dict] | None = None,
            arms: tuple[str, ...] = (*SELECTORS, *CONTROLS), passes: tuple[int, ...] = (1, 2, 3),
            expected_cases: int = 100, count_tokens: Callable[[str], int] | None = None) -> dict:
    """Attempt each ready package once; preserve failures and stop on unsafe drift."""
    manifest, packages, plan = prepare(manifest_file, candidates_file, packages_file, tokenizer_dir,
                                       arms=arms, passes=passes, expected_cases=expected_cases, count_tokens=count_tokens)
    _check_astra_gate(astra_gate, manifest_file)
    if request_fn is None:
        if endpoint != manifest["endpoint"]:
            raise ValueError("verdict_endpoint_required")
        request_fn = _http_request(endpoint, manifest["model"], manifest["timeout_seconds"])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "plan.json").write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    intents, results = output_dir / "intents.jsonl", output_dir / "results.jsonl"
    intents.touch(exist_ok=False)
    results.touch(exist_ok=False)
    total_input = total_output = attempts = 0
    counter = count_tokens or _local_token_counter(tokenizer_dir)

    def attempt(identity: dict, prompt: str) -> str:
        nonlocal total_input, total_output, attempts
        try:
            planned_input = counter(prompt)
            if type(planned_input) is not int or planned_input < 1:
                raise ValueError("verdict_runtime_token_count")
        except Exception:
            _append(results, {**identity, "outcome": "runtime_token_count_failed", "verdict": None})
            return "stop"
        reserved_input = planned_input + manifest["input_margin_tokens"]
        if (reserved_input + manifest["max_output_tokens"] > manifest["context_tokens"]
                or total_input + reserved_input > manifest["maximum_total_input_tokens"]
                or total_output + manifest["max_output_tokens"] > manifest["maximum_total_output_tokens"]):
            _append(results, {**identity, "outcome": "token_cap_before_call", "verdict": None})
            return "stop"
        attempts += 1
        _append(intents, {**identity, "ordinal": attempts, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "reserved_input_tokens": reserved_input, "reserved_output_tokens": manifest["max_output_tokens"]})
        try:
            response = request_fn(prompt, manifest["max_output_tokens"])
        except Exception:
            _append(results, {**identity, "outcome": "unknown_submission_outcome", "verdict": None})
            return "stop"
        if not isinstance(response, dict) or response.get("model") != manifest["returned_model"]:
            _append(results, {**identity, "outcome": "model_identity_mismatch", "verdict": None})
            return "stop"
        usage = response.get("usage")
        if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] < 0
                or type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 0):
            _append(results, {**identity, "outcome": "invalid_usage", "verdict": None})
            return "stop"
        total_input += usage["prompt_tokens"]
        total_output += usage["completion_tokens"]
        if (usage["prompt_tokens"] > reserved_input or usage["completion_tokens"] > manifest["max_output_tokens"]
                or total_input > manifest["maximum_total_input_tokens"]
                or total_output > manifest["maximum_total_output_tokens"]):
            _append(results, {**identity, "outcome": "token_cap_exhausted", "verdict": None, "usage": usage})
            return "stop"
        if response.get("finish_reason") != "stop" or not isinstance(response.get("content"), str):
            _append(results, {**identity, "outcome": "invalid_completion", "verdict": None, "usage": usage})
            return "measured_failure"
        try:
            verdict, _ = parse_hero_terminal_output(response["content"])
        except ContractError:
            _append(results, {**identity, "outcome": "invalid_verdict", "verdict": None, "usage": usage})
            return "measured_failure"
        except Exception:
            _append(results, {**identity, "outcome": "unexpected_exception", "verdict": None, "usage": usage})
            return "stop"
        _append(results, {**identity, "outcome": "ok", "verdict": verdict, "returned_model": response["model"], "usage": usage})
        return "valid"

    complete = attempt({"phase": "warmup", "arm": None, "pass": None, "case_id": None},
                       render_prompt(WARMUP_CLAIM, [{"text": WARMUP_SNIPPET}])) == "valid"
    for package in packages:
        identity = {"phase": "measured", "arm": package["arm"], "pass": package["pass"], "case_id": package["case_id"]}
        if package["status"] == "failed":
            _append(results, {**identity, "outcome": "upstream_package_failure", "verdict": None})
        elif not complete:
            _append(results, {**identity, "outcome": "not_attempted_after_stop", "verdict": None})
        elif attempt(identity, render_prompt(package["claim"], package["selected_passages"])) == "stop":
            complete = False
    receipt = {"schema": "three-pass-verdict-run/v1", "complete": complete,
               "manifest_sha256": plan["manifest_sha256"], "attempts": attempts,
               "measured_results": sum(1 for line in results.read_text().splitlines() if json.loads(line)["phase"] == "measured"),
               "input_tokens": total_input, "output_tokens": total_output,
               "intents_sha256": sha256(intents), "results_sha256": sha256(results),
               "no_retries": True}
    (output_dir / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--endpoint")
    parser.add_argument("--astra-gate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-one-case", action="store_true",
                        help="Use the separately reviewed one-case include-all development smoke scope")
    args = parser.parse_args()
    if args.plan_only == args.execute:
        raise ValueError("choose_plan_or_execute")
    scope = ({"arms": ("include_all",), "passes": (1,), "expected_cases": 1}
             if args.smoke_one_case else {})
    if args.plan_only:
        _, _, summary = prepare(args.manifest, args.candidates, args.packages, args.tokenizer_dir, **scope)
        print(json.dumps(summary, sort_keys=True))
    else:
        if args.output is None:
            raise ValueError("verdict_output_required")
        print(json.dumps(execute(args.manifest, args.candidates, args.packages,
                                 args.tokenizer_dir, args.output, astra_gate=args.astra_gate,
                                 endpoint=args.endpoint, **scope), sort_keys=True))


if __name__ == "__main__":
    main()
