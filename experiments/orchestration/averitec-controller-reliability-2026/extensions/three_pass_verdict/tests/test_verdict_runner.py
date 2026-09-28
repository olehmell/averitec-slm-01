"""Offline fixtures for the hash-bound, no-retry verdict runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "verdict_runner.py"
SPEC = importlib.util.spec_from_file_location("three_pass_verdict_runner", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_local_token_counter_uses_input_ids_not_batch_encoding_keys(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
    fake = ModuleType("transformers")
    fake.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert runner._local_token_counter(tmp_path)("claim") == 4


def files(tmp_path: Path, *, failed: bool = False) -> tuple[Path, Path, Path, Path]:
    cases = []
    for number in (0, 1):
        candidates = [{"id": f"C{i:02d}", "passage_id": f"passage-{number}-{i}",
                       "text": f"evidence {number}/{i}", "url": f"https://example.test/{number}/{i}",
                       "source_start": i, "source_text_length": 100,
                       "source_text_sha256": f"{number + 1:064x}"} for i in range(1, 11)]
        cases.append({"case_id": f"averitec-dev-{number:04d}", "group_id": f"g{number}",
                      "claim": f"Claim {number}", "candidates": candidates})
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(json.dumps({"gold_included": False, "cases": cases}))
    packages = []
    for arm in ("selector", "include_all", "include_none"):
        for case in cases:
            bad = failed and arm == "selector" and case is cases[1]
            selected = None if bad else case["candidates"] if arm != "include_none" else []
            packages.append({"schema": runner.PACKAGE_SCHEMA, "arm": arm, "pass": 1,
                             "case_id": case["case_id"], "group_id": case["group_id"], "claim": case["claim"],
                             "mode": "selector" if arm == "selector" else arm,
                             "status": "failed" if bad else "ready", "errors": ["missing_candidate_id:C01"] if bad else [],
                             "selected_passages": selected})
    packages_file = tmp_path / "packages.json"
    packages_file.write_text(json.dumps({"schema": "three-pass-verdict-packages/v1", "packages": packages}))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (tokenizer / name).write_text("{}")
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "schema": runner.SCHEMA, "candidates_sha256": runner.sha256(candidate_file),
        "packages_sha256": runner.sha256(packages_file), "runner_sha256": runner.sha256(MODULE_PATH),
        "downstream_sha256": runner.sha256(MODULE_PATH.parent / "downstream.py"),
        "contracts_sha256": runner.sha256(runner._SIBLING / "inference/contracts.py"),
        "tokenizer_files_sha256": {name: runner.sha256(tokenizer / name) for name in
                                   ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")},
        "model": runner.MODEL, "revision": runner.REVISION, "returned_model": runner.MODEL,
        "prompt_sha256": __import__("hashlib").sha256(runner.PROMPT_PREFIX.encode()).hexdigest(),
        "arms": ["selector", "include_all", "include_none"], "passes": [1], "cases": 2,
        "context_tokens": 1000, "max_output_tokens": 50,
        "maximum_total_input_tokens": 1000, "maximum_total_output_tokens": 350,
        "input_margin_tokens": 16, "endpoint": "http://127.0.0.1:8001/v1/chat/completions",
        "timeout_seconds": 10, "warmup_calls": 1, "maximum_calls": 7, "temperature": 0, "retries": 0,
    }))
    return manifest_file, candidate_file, packages_file, tokenizer


def prepare(paths, **kwargs):
    return runner.prepare(*paths, arms=("selector", "include_all", "include_none"), passes=(1,),
                          expected_cases=2, count_tokens=lambda prompt: len(prompt) // 20 + 1, **kwargs)


def execute(paths, output: Path, request_fn):
    gate = output.parent / "astra-gate.json"
    gate.write_text(json.dumps({"schema": "averitec-astra-launch-gate/v1",
                                "manifest_sha256": runner.sha256(paths[0]),
                                "decision": "approved", "reviewer": "gpt-6-astra"}))
    return runner.execute(*paths, output, astra_gate=gate, request_fn=request_fn,
                          arms=("selector", "include_all", "include_none"), passes=(1,),
                          expected_cases=2, count_tokens=lambda prompt: len(prompt) // 20 + 1)


def response(prompt: str, maximum: int) -> dict:
    assert maximum == 50
    return {"model": runner.MODEL, "content": "Synthetic explanation.\nVerdict: Supported",
            "finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_plan_preflights_every_ready_prompt_and_is_arm_blind(tmp_path: Path) -> None:
    paths = files(tmp_path)
    _, packages, summary = prepare(paths)
    assert summary["packages"] == 6 and summary["ready_packages"] == 6
    assert summary["maximum_actual_calls"] == 7 and summary["model_calls"] == 0
    first = runner.render_prompt(packages[0]["claim"], packages[0]["selected_passages"])
    control = runner.render_prompt(packages[2]["claim"], packages[2]["selected_passages"])
    assert first == control
    assert "selector" not in first and "gold" not in first
    assert "evidence 0/1" in first
    assert "(none)" in runner.render_prompt("Claim", [])


def test_execute_journals_warmup_and_each_measured_case_once(tmp_path: Path) -> None:
    paths = files(tmp_path, failed=True)
    calls = []
    def request(prompt: str, maximum: int) -> dict:
        calls.append(prompt)
        return response(prompt, maximum)
    output = tmp_path / "run"
    receipt = execute(paths, output, request)
    intents = [json.loads(line) for line in (output / "intents.jsonl").read_text().splitlines()]
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert receipt["complete"] is True and receipt["attempts"] == 6
    assert len(calls) == len(intents) == 6 and len(results) == 7
    assert intents[0]["phase"] == "warmup"
    assert sum(row["outcome"] == "upstream_package_failure" for row in results) == 1
    assert all(row.get("verdict") == "Supported" for row in results if row["outcome"] == "ok")
    with pytest.raises(FileExistsError):
        execute(paths, output, request)


def test_identity_mismatch_and_transport_failure_stop_without_retry(tmp_path: Path) -> None:
    paths = files(tmp_path)
    attempts = []
    def wrong(prompt: str, maximum: int) -> dict:
        attempts.append(prompt)
        return {**response(prompt, maximum), "model": "different-model"}
    output = tmp_path / "identity"
    receipt = execute(paths, output, wrong)
    assert receipt["complete"] is False and receipt["attempts"] == 1
    assert receipt["measured_results"] == 6
    assert len(attempts) == 1
    identity_results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert identity_results[0]["outcome"] == "model_identity_mismatch"
    assert all(row["outcome"] == "not_attempted_after_stop" for row in identity_results[1:])
    def failure(prompt: str, maximum: int) -> dict:
        raise TimeoutError("unknown submission")
    other = tmp_path / "unknown"
    receipt = execute(paths, other, failure)
    assert receipt["attempts"] == 1 and receipt["complete"] is False
    assert json.loads((other / "results.jsonl").read_text().splitlines()[0])["outcome"] == "unknown_submission_outcome"


def test_actual_input_above_preflight_margin_stops_before_next_call(tmp_path: Path) -> None:
    paths = files(tmp_path)
    calls = []
    def excessive(prompt: str, maximum: int) -> dict:
        calls.append(prompt)
        return {**response(prompt, maximum), "usage": {
            "prompt_tokens": len(prompt) // 20 + 18, "completion_tokens": 5}}
    output = tmp_path / "margin"
    receipt = execute(paths, output, excessive)
    assert receipt["complete"] is False and receipt["attempts"] == 1 and len(calls) == 1
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert rows[0]["outcome"] == "token_cap_exhausted"
    assert all(row["outcome"] == "not_attempted_after_stop" for row in rows[1:])


def test_manifest_rejects_nonlocal_endpoint_and_contract_drift(tmp_path: Path) -> None:
    paths = files(tmp_path)
    manifest = json.loads(paths[0].read_text())
    manifest["endpoint"] = "https://example.test/v1/chat/completions"
    paths[0].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verdict_manifest_endpoint"):
        prepare(paths)
    manifest["endpoint"] = "http://127.0.0.1:8001/v1/chat/completions"
    manifest["contracts_sha256"] = "0" * 64
    paths[0].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verdict_manifest_hash_drift"):
        prepare(paths)


def test_invalid_terminal_line_stops_without_repair(tmp_path: Path) -> None:
    paths = files(tmp_path)
    def invalid(prompt: str, maximum: int) -> dict:
        return {**response(prompt, maximum), "content": "Verdict: Supported"}
    receipt = execute(paths, tmp_path / "invalid", invalid)
    assert receipt["attempts"] == 1 and receipt["complete"] is False


def test_measured_invalid_verdict_and_completion_do_not_stop_later_cases(tmp_path: Path) -> None:
    paths = files(tmp_path)
    calls = 0
    def mixed(prompt: str, maximum: int) -> dict:
        nonlocal calls
        calls += 1
        base = response(prompt, maximum)
        if calls == 2:
            return {**base, "content": "Verdict: Supported"}  # No justification.
        if calls == 3:
            return {**base, "finish_reason": "length"}
        return base
    output = tmp_path / "mixed"
    receipt = execute(paths, output, mixed)
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert calls == 7 and receipt["attempts"] == 7 and receipt["complete"] is True
    assert [row["outcome"] for row in results[:4]] == ["ok", "invalid_verdict", "invalid_completion", "ok"]
    assert len(results) == 7 and all(row["outcome"] != "not_attempted_after_stop" for row in results)


def test_execution_requires_exact_astra_review_gate_before_call(tmp_path: Path) -> None:
    paths = files(tmp_path)
    calls = []
    def request(prompt: str, maximum: int) -> dict:
        calls.append(prompt)
        return response(prompt, maximum)
    kwargs = {"request_fn": request, "arms": ("selector", "include_all", "include_none"),
              "passes": (1,), "expected_cases": 2, "count_tokens": lambda _: 10}
    with pytest.raises(ValueError, match="verdict_astra_gate_required"):
        runner.execute(*paths, tmp_path / "no-gate", **kwargs)
    assert not calls and not (tmp_path / "no-gate").exists()
    gate = tmp_path / "bad-gate.json"
    gate.write_text(json.dumps({"schema": "averitec-astra-launch-gate/v1",
                                "manifest_sha256": "0" * 64, "decision": "approved", "reviewer": "gpt-6-astra"}))
    with pytest.raises(ValueError, match="verdict_astra_gate"):
        runner.execute(*paths, tmp_path / "bad-gate-output", astra_gate=gate, **kwargs)
    assert not calls and not (tmp_path / "bad-gate-output").exists()


def test_hash_order_and_context_fail_closed(tmp_path: Path) -> None:
    paths = files(tmp_path)
    manifest, candidates, packages, tokenizer = paths
    package_data = json.loads(packages.read_text())
    package_data["packages"][0]["selected_passages"].reverse()
    packages.write_text(json.dumps(package_data))
    with pytest.raises(ValueError, match="hash_drift"):
        prepare(paths)
    manifest_data = json.loads(manifest.read_text())
    manifest_data["packages_sha256"] = runner.sha256(packages)
    manifest.write_text(json.dumps(manifest_data))
    with pytest.raises(ValueError, match="selected_order_or_source"):
        prepare(paths)
    package_data["packages"][0]["selected_passages"].reverse()
    packages.write_text(json.dumps(package_data))
    manifest_data["packages_sha256"] = runner.sha256(packages)
    manifest.write_text(json.dumps(manifest_data))
    with pytest.raises(ValueError, match="context_preflight"):
        runner.prepare(*paths, arms=("selector", "include_all", "include_none"), passes=(1,),
                       expected_cases=2, count_tokens=lambda _: 1000)
