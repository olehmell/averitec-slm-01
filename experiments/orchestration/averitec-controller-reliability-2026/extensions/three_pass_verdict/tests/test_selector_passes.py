from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import selector_passes as runner  # noqa: E402
from providers import ControllerResult  # noqa: E402


def test_input_usage_policy_matches_provider_preflight_semantics() -> None:
    # The Jev smoke returned 555 tokens for a 2,459-token estimated reserve.
    assert runner._input_usage_allowed("jev", 555, 2459, 0)
    assert not runner._input_usage_allowed("jev", 2460, 2459, 0)
    assert runner._input_usage_allowed("gemini", 279, 279, 128)
    assert runner._input_usage_allowed("gemini", 407, 279, 128)
    assert not runner._input_usage_allowed("gemini", 408, 279, 128)
    assert runner._input_usage_allowed("qwen", 410, 410, 0)
    assert not runner._input_usage_allowed("qwen", 409, 410, 0)
    assert not runner._input_usage_allowed("jev", None, 2459, 0)


def test_jeff_nominal_output_token_semantics_are_distinct_from_missing_input() -> None:
    assert runner._warmup_usage_valid("jeff", {"input_tokens": 160, "output_tokens": None})
    assert not runner._warmup_usage_valid("jeff", {"input_tokens": None, "output_tokens": None})
    assert not runner._warmup_usage_valid("jeff", {"input_tokens": 160, "output_tokens": 0})
    assert not runner._warmup_usage_valid("laya_typed", {"input_tokens": 160, "output_tokens": None})
    assert runner._warmup_usage_valid("laya_typed", {"input_tokens": 160, "output_tokens": 0})


def test_http_guard_binds_one_wire_request_and_rejects_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"model": "jev-1.13.0", "input": "test"}
    expected = hashlib.sha256(runner.wire_bytes(body)).hexdigest()
    seen: list[object] = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self): return b'{"ok":true}'

    class Opener:
        def open(self, request, timeout):
            seen.append((request.full_url, timeout))
            return Response()

    def opener(handler):
        assert isinstance(handler, runner._NoRedirect)
        seen.append(handler.redirect_request(None) is None)
        return Opener()

    monkeypatch.setattr(runner, "build_opener", opener)
    original = runner.provider_module._http_post
    with runner._guard_http_request("jev", expected, None, 5) as calls:
        assert runner.provider_module._http_post(runner.provider_module.TYPESAFE_SYSTEM_ONE_URL,
                                                 body, {}, 2) == {"ok": True}
        assert calls() == 1
        with pytest.raises(ValueError, match="request_drift"):
            runner.provider_module._http_post(runner.provider_module.TYPESAFE_SYSTEM_ONE_URL,
                                              body, {}, 2)
    assert runner.provider_module._http_post is original
    assert seen == [True, (runner.provider_module.TYPESAFE_SYSTEM_ONE_URL, 2)]


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _candidates(path: Path) -> None:
    cases = []
    for number in range(100):
        passages = [{"id": f"C{n:02d}", "passage_id": f"p-{number}-{n}",
                     "text": f"Evidence {number}/{n}.", "url": f"https://example.invalid/{number}/{n}",
                     "source_start": 0, "source_text_length": 20, "source_text_sha256": "a" * 64}
                    for n in range(1, 11)]
        cases.append({"case_id": f"case-{number:03d}", "group_id": f"g-{number}",
                      "claim": f"Claim {number}.", "candidates": passages})
    _json(path, {"schema": "averitec-frozen-evidence-candidates/v1", "source_freeze_sha256": "b" * 64,
                 "source_traces_sha256": "c" * 64, "gold_included": False, "cases": cases})


def _launch(path: Path, candidates: Path) -> None:
    arms = {name: {"model": model, "profile": profile,
                   "returned_model": runner.previous_jeff.RETURNED_MODEL if name == "jeff" else model,
                   "wall_seconds": 600, "maximum_calls": 1001,
                   "gpu_allocation_seconds": 0 if name in {"jev", "gemini"} else 700,
                   "token_caps": {"input_per_call": 10000, "output_per_call": 5000,
                                  "input_total": 10_010_000, "output_total": 5_005_000,
                                  "context_limit": 20000} if name in runner.HTTP_ARMS else None,
                   "token_preflight_sha256": "f" * 64 if name in runner.HTTP_ARMS else None}
            for name, (model, profile) in runner.ARMS.items()}
    _json(path, {"schema": runner.SCHEMA,
                 "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runner.REPO, text=True).strip(),
                 "candidates_sha256": runner.digest(candidates),
                 "instructions_sha256": runner.digest(runner.EVIDENCE / "selector_instructions.txt"),
                 "code_sha256": {name: runner.digest(runner.REPO / name) for name in runner.REQUIRED_CODE},
                 "passes": [1, 2, 3], "arms": arms, "retries": 0, "total_measured_calls": 21000,
                 "resource_caps": {"maximum_gpu_seconds": 10500, "maximum_api_requests": 7000,
                                   "maximum_jobs": 21},
                 "jeff_asset_manifest_sha256": "d" * 64, "jeff_checkpoint_sha256": "e" * 64,
                 "hosted_budget_seed_sha256": "f" * 64})


def _gate(path: Path, launch: Path) -> None:
    _json(path, {"schema": "averitec-astra-launch-gate/v1", "manifest_sha256": runner.digest(launch),
                 "decision": "approved", "reviewer": "gpt-6-astra"})


def test_frozen_source_audit_accepts_historical_code_hashes_but_keeps_gate(tmp_path: Path) -> None:
    candidates, launch, gate = (tmp_path / name for name in ("candidates.json", "launch.json", "gate.json"))
    _candidates(candidates)
    _launch(launch, candidates)
    value = json.loads(launch.read_text(encoding="utf-8"))
    value["source_commit"] = "a" * 40
    value["code_sha256"][next(iter(value["code_sha256"]))] = "b" * 64
    _json(launch, value)
    _gate(gate, launch)
    runner.load_launch(launch, candidates, gate=gate, frozen_source=True)
    with pytest.raises(ValueError, match="manifest_policy"):
        runner.load_launch(launch, candidates, gate=gate)
    _json(gate, {"schema": "averitec-astra-launch-gate/v1", "manifest_sha256": "0" * 64,
                 "decision": "approved", "reviewer": "gpt-6-astra"})
    with pytest.raises(ValueError, match="astra_gate"):
        runner.load_launch(launch, candidates, gate=gate, frozen_source=True)


def test_scoped_gate_rejects_hosted_arm(tmp_path: Path) -> None:
    candidates, launch, gate = tmp_path / "candidates.json", tmp_path / "launch.json", tmp_path / "gate.json"
    _candidates(candidates)
    _launch(launch, candidates)
    _json(gate, {"schema": "averitec-astra-launch-gate/v2", "manifest_sha256": runner.digest(launch),
                 "decision": "approved", "reviewer": "gpt-6-astra",
                 "allowed_arms": ["jeff", "laya_typed", "lfm", "lfm26", "qwen"],
                 "allowed_passes": [1, 2, 3]})
    runner.load_launch(launch, candidates, gate=gate, arm="qwen", pass_number=1)
    with pytest.raises(ValueError, match="astra_gate_scope"):
        runner.load_launch(launch, candidates, gate=gate, arm="jev", pass_number=1)


def test_jeff_native_dictionary_identity_is_checked_before_forward(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_hash = "a" * 64
    asset_file = tmp_path / "assets.json"
    _json(asset_file, {"schema": runner.previous_jeff.ASSET_SCHEMA,
                       "checkpoint_sha256": checkpoint_hash,
                       "model": runner.previous_jeff.MODEL,
                       "model_revision": runner.previous_jeff.MODEL_REVISION})
    identity = {"upstream_commit": runner.previous_jeff.UPSTREAM_COMMIT,
                "model": runner.previous_jeff.MODEL,
                "returned_model": runner.previous_jeff.RETURNED_MODEL,
                "model_revision": runner.previous_jeff.MODEL_REVISION,
                "checkpoint_sha256": checkpoint_hash}
    class Runtime:
        def __init__(self) -> None:
            self.identity = identity
        def preflight(self, *_args: object) -> None: pass
        def execute(self, *_args: object) -> None: pass
    runtime = Runtime()
    monkeypatch.setattr(runner.previous_jeff, "_validate_checkpoint_tree", lambda *_args: None)
    monkeypatch.setattr(runner.previous_jeff, "_load_runtime", lambda *_args: runtime)
    result, observed = runner._make_jeff(checkpoint=tmp_path, assets=asset_file,
                                         expected_assets=runner.digest(asset_file),
                                         expected_checkpoint=checkpoint_hash)
    assert result is runtime and observed == identity


class FakeSelector:
    def __init__(self, arm: str, **_kwargs: object):
        self.model = runner.ARMS[arm][0]
        self.expected_returned_model = self.model
        self.resolved_profile = {"name": runner.ARMS[arm][1]}
        self.timeout_seconds = 30
        self.tracer = None
        self.calls = 0

    def choose(self, observation: dict, instructions: str, actions: list[str]) -> ControllerResult:
        self.calls += 1
        assert set(observation) == {"claim", "candidate"}
        assert set(observation["candidate"]) == {"id", "text", "url"}
        assert instructions and actions == ["include", "exclude"]
        with self.tracer.span("controller.request", kind="generation", input=observation, model=self.model) as span:
            span.update(output={"model": self.model, "action": "include"}, usage={"input": 4, "output": 1})
        return ControllerResult("include", "ok", 1.0, 4, 1, self.model, returned_model=self.model)


class FakeNominalJeff(FakeSelector):
    def __init__(self, arm: str, **kwargs: object):
        super().__init__(arm, **kwargs)
        self.expected_returned_model = runner.previous_jeff.RETURNED_MODEL

    def choose(self, observation: dict, instructions: str, actions: list[str]) -> ControllerResult:
        self.calls += 1
        assert actions == ["include", "exclude"]
        with self.tracer.span("controller.request", kind="generation", input=observation, model=self.model) as span:
            span.update(output={"model": self.expected_returned_model, "action": "exclude"},
                        metadata={"output_token_semantics": "nominal_not_generated"})
        return ControllerResult("exclude", "ok", 1.0, 160, None, self.model,
                                returned_model=self.expected_returned_model)


def test_plan_is_read_only_and_binds_exact_21000(tmp_path: Path) -> None:
    candidates, launch = tmp_path / "candidates.json", tmp_path / "launch.json"
    _candidates(candidates)
    _launch(launch, candidates)
    result = runner.plan(launch, candidates)
    assert result["total_measured_calls"] == 21000
    assert result["total_warmups"] == 21
    assert result["model_calls"] == 0
    assert len(list(tmp_path.iterdir())) == 2


def test_one_arm_pass_append_only_and_verifiable(tmp_path: Path) -> None:
    candidates, launch, gate, output = (tmp_path / name for name in ("candidates.json", "launch.json", "gate.json", "run"))
    _candidates(candidates)
    _launch(launch, candidates)
    _gate(gate, launch)
    receipt = runner.execute("jev", 2, candidates, launch, gate, output, selector_factory=FakeSelector)
    assert receipt["complete"] and receipt["all_valid"]
    assert receipt["qualification"] == "test_only"
    assert json.loads((output / "preflight.json").read_text())["complete"] is False
    assert len((output / "intents.jsonl").read_text().splitlines()) == 1001
    assert len((output / "results.jsonl").read_text().splitlines()) == 1001
    assert runner.verify(output, candidates, launch, gate) == receipt
    with pytest.raises(ValueError, match="new_directory"):
        runner.execute("jev", 2, candidates, launch, gate, output, selector_factory=FakeSelector)


def test_jeff_nominal_output_usage_survives_warmup_and_receipt(tmp_path: Path) -> None:
    candidates, launch, gate, output = (tmp_path / name for name in ("candidates.json", "launch.json", "gate.json", "run"))
    _candidates(candidates)
    _launch(launch, candidates)
    _gate(gate, launch)
    receipt = runner.execute("jeff", 1, candidates, launch, gate, output, selector_factory=FakeNominalJeff)
    assert receipt["complete"] and receipt["all_valid"]
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(results) == 1001
    assert all(row["usage"] == {"input_tokens": 160, "output_tokens": None} for row in results)
    assert receipt["token_usage"] == {"input_tokens": 160160, "output_tokens": None}
    assert runner.verify(output, candidates, launch, gate) == receipt
    tampered = {**receipt, "token_usage": {"input_tokens": 0, "output_tokens": 0}}
    _json(output / "receipt.json", tampered)
    with pytest.raises(ValueError, match="native_token_usage_receipt"):
        runner.verify(output, candidates, launch, gate)


def test_native_usage_unknown_is_not_a_zero() -> None:
    assert runner._native_token_usage("jeff", [
        {"usage": {"input_tokens": 160, "output_tokens": None}},
        {"usage": {"input_tokens": None, "output_tokens": None}},
    ]) == {"input_tokens": None, "output_tokens": None}
    assert runner._native_token_usage("laya_typed", [
        {"usage": {"input_tokens": 12, "output_tokens": 0}},
        {"usage": {"input_tokens": 13, "output_tokens": None}},
    ]) == {"input_tokens": 25, "output_tokens": None}


def test_frozen_v13_laya_zero_aggregate_is_reported_without_rewriting_receipt() -> None:
    rows = [{"usage": {"input_tokens": 154, "output_tokens": 0}},
            {"usage": {"input_tokens": 200, "output_tokens": 0}}]
    reported = {"input_tokens": 0, "output_tokens": 0}
    audit = runner._audit_native_usage("laya_typed", "f9f62f46b3dd566d9ef9e10f7eef40c2f9747602",
                                       rows, reported, frozen_source=True)
    assert audit == {"reason": "historical_v13_laya_zero_aggregate", "reported": reported,
                     "recomputed_from_results": {"input_tokens": 354, "output_tokens": 0}}
    for source, frozen in (("f9f62f46b3dd566d9ef9e10f7eef40c2f9747602", False),
                           ("0" * 40, True)):
        with pytest.raises(ValueError, match="native_token_usage_receipt"):
            runner._audit_native_usage("laya_typed", source, rows, reported, frozen_source=frozen)


def test_manifest_drift_and_false_astra_gate_fail_closed(tmp_path: Path) -> None:
    candidates, launch, gate = (tmp_path / name for name in ("candidates.json", "launch.json", "gate.json"))
    _candidates(candidates)
    _launch(launch, candidates)
    _gate(gate, launch)
    invalid = json.loads(gate.read_text())
    invalid["decision"] = "pending"
    _json(gate, invalid)
    with pytest.raises(ValueError, match="astra_gate"):
        runner.load_launch(launch, candidates, gate=gate)
    _gate(gate, launch)
    value = json.loads(launch.read_text())
    value["reference_path"] = "gold.json"
    _json(launch, value)
    with pytest.raises(ValueError, match="manifest_shape"):
        runner.plan(launch, candidates)


def test_intent_hash_detects_receipt_tampering(tmp_path: Path) -> None:
    candidates, launch, gate, output = (tmp_path / name for name in ("candidates.json", "launch.json", "gate.json", "run"))
    _candidates(candidates)
    _launch(launch, candidates)
    _gate(gate, launch)
    runner.execute("jev", 1, candidates, launch, gate, output, selector_factory=FakeSelector)
    path = output / "intents.jsonl"
    text = path.read_text().replace("case-000", "case-999", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="journal_drift"):
        runner.verify(output, candidates, launch, gate)


def test_measured_failure_is_kept_without_reissue(tmp_path: Path) -> None:
    candidates, launch, gate, output = (tmp_path / name for name in ("candidates.json", "launch.json", "gate.json", "run"))
    _candidates(candidates)
    _launch(launch, candidates)
    _gate(gate, launch)

    class OneFailure(FakeSelector):
        def choose(self, observation: dict, instructions: str, actions: list[str]) -> ControllerResult:
            result = super().choose(observation, instructions, actions)
            if self.calls == 2:
                return ControllerResult(None, "invalid_output", 1.0, 4, 1, self.model,
                                        error_code="bad_choice", returned_model=self.model)
            return result

    receipt = runner.execute("jev", 3, candidates, launch, gate, output, selector_factory=OneFailure)
    assert receipt["complete"] and not receipt["all_valid"]
    assert receipt["outcome_counts"] == {"invalid_output": 1, "ok": 1000}
    measured = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()[1:]]
    assert measured[0]["outcome"] == "invalid_output"
    assert measured[1]["id"] == "C02" and measured[1]["outcome"] == "ok"
    assert runner.verify(output, candidates, launch, gate) == receipt


def test_unknown_transport_stops_after_one_failed_intent(tmp_path: Path) -> None:
    candidates, launch, gate, output = (tmp_path / name for name in ("candidates.json", "launch.json", "gate.json", "run"))
    _candidates(candidates)
    _launch(launch, candidates)
    _gate(gate, launch)

    class UnknownTransport(FakeSelector):
        def choose(self, observation: dict, instructions: str, actions: list[str]) -> ControllerResult:
            result = super().choose(observation, instructions, actions)
            if self.calls == 2:
                return ControllerResult(None, "transport_error", 1.0, None, None, self.model,
                                        error_code="transport_failure")
            return result

    receipt = runner.execute("jev", 1, candidates, launch, gate, output, selector_factory=UnknownTransport)
    assert receipt["complete"] is False
    assert receipt["stop_reason"] == "unknown_transport_outcome_no_reissue"
    intents = (output / "intents.jsonl").read_text().splitlines()
    results = (output / "results.jsonl").read_text().splitlines()
    assert len(intents) == len(results) == 2
    assert json.loads(results[1])["outcome"] == "transport_error"


def test_real_http_arm_cannot_launch_without_exact_token_report(tmp_path: Path) -> None:
    candidates, launch, gate, output = (tmp_path / name for name in ("candidates.json", "launch.json", "gate.json", "run"))
    _candidates(candidates)
    _launch(launch, candidates)
    _gate(gate, launch)
    with pytest.raises(ValueError, match="dispatch_claim_required"):
        runner.execute("jev", 1, candidates, launch, gate, output)
    assert not output.exists()


def test_exact_token_report_is_bound_to_all_request_bodies(tmp_path: Path) -> None:
    report = tmp_path / "tokens.json"
    observations = [runner.WARMUP] * 1001
    requests = [{"model": "Qwen/Qwen3.5-4B", "max_tokens": 160, "input": n} for n in range(1001)]
    caps = {"input_per_call": 100, "output_per_call": 160, "input_total": 100100,
            "output_total": 160160, "context_limit": 260}
    rows = [{"observation_sha256": hashlib.sha256(runner.canonical(o).encode()).hexdigest(),
             "request_sha256": hashlib.sha256(runner.wire_bytes(r)).hexdigest(),
             "serialized_bytes": len(runner.wire_bytes(r)),
             "input_tokens": 100} for o, r in zip(observations, requests)]
    _json(report, {"schema": "averitec-selector-token-preflight/v2", "arm": "qwen",
                   "model": "Qwen/Qwen3.5-4B", "profile": "qwen_baseline_warmup180_v2",
                   "method": "exact_pinned_tokenizer", "method_identity_sha256": "a" * 64,
                   "context_limit_tokens": 260, "output_request_limit_tokens": 160,
                   "output_limit_enforced": True, "input_margin_tokens": 0, "records": rows})
    checked = runner._load_http_preflight(report, runner.digest(report), "qwen", "Qwen/Qwen3.5-4B",
                                           "qwen_baseline_warmup180_v2", caps, observations, requests)
    assert len(checked) == 1001
    requests[700] = {"model": "Qwen/Qwen3.5-4B", "max_tokens": 160, "input": "drift"}
    with pytest.raises(ValueError, match="preflight_record"):
        runner._load_http_preflight(report, runner.digest(report), "qwen", "Qwen/Qwen3.5-4B",
                                           "qwen_baseline_warmup180_v2", caps, observations, requests)


def test_http_request_capture_uses_adapter_without_network() -> None:
    selector = runner._make_selector("jev", endpoint=None, laya_checkpoint=None, keys={})
    requests = runner._capture_http_requests("jev", selector, [runner.WARMUP], "Choose one action.")
    assert len(requests) == 1
    assert requests[0]["model"] == "jev-1.13.0"
    assert requests[0]["state"] == runner.WARMUP
    assert requests[0]["questions"]["next_action"]["criteria"] == {"include": None, "exclude": None}
    changed = runner._capture_http_requests("jev", selector, [runner.WARMUP], "Different instructions.")
    assert runner.canonical(requests[0]) != runner.canonical(changed[0])


def test_jev_requires_explicit_estimated_reserve_report(tmp_path: Path) -> None:
    path = tmp_path / "claimed-report.json"
    _json(path, {"claimed": "exact"})
    with pytest.raises(ValueError, match="token_preflight_shape"):
        runner._load_http_preflight(path, runner.digest(path), "jev", "jev-1.13.0", "baseline",
                                           {}, [], [])


def test_jev_estimate_binds_wire_bytes_and_keeps_context_unverified(tmp_path: Path) -> None:
    path = tmp_path / "jev-estimated.json"
    observations = [runner.WARMUP] * 1001
    requests = [{"model": "jev-1.13.0", "state": runner.WARMUP,
                 "questions": {"next_action": {"type": "choice", "instructions": "choose",
                                                "criteria": {"include": None, "exclude": None}}}}] * 1001
    raw = runner.wire_bytes(requests[0])
    rows = [{"observation_sha256": hashlib.sha256(runner.canonical(runner.WARMUP).encode()).hexdigest(),
             "request_sha256": hashlib.sha256(raw).hexdigest(), "serialized_bytes": len(raw),
             "input_tokens": len(raw) + 1024}] * 1001
    report = {"schema": "averitec-selector-token-preflight/v2", "arm": "jev", "model": "jev-1.13.0",
              "profile": "baseline", "method": "estimated_wire_bytes_plus_1024",
              "method_identity_sha256": "a" * 64, "context_limit_tokens": 32768,
              "output_request_limit_tokens": 1024, "output_limit_enforced": False,
              "input_margin_tokens": 0, "records": rows}
    _json(path, report)
    caps = {"input_per_call": 4096, "output_per_call": 1024,
            "input_total": 10000, "output_total": 2000, "context_limit": 32768}
    assert len(runner._load_http_preflight(path, runner.digest(path), "jev", "jev-1.13.0",
                                           "baseline", caps, observations, requests)) == 1001
    report["records"][0] = {**rows[0], "serialized_bytes": len(raw) - 1}
    _json(path, report)
    with pytest.raises(ValueError, match="preflight_record"):
        runner._load_http_preflight(path, runner.digest(path), "jev", "jev-1.13.0",
                                    "baseline", caps, observations, requests)
