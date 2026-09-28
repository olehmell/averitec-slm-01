"""The approved comparison population and resources cannot drift silently."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).parents[1]
sys.path.insert(0, str(HERE))
import run_selector


def test_approved_plan_matches_runner_resource_policy():
    plan = json.loads((HERE / "recovery-v2-plan.json").read_text())
    assert set(plan["rerun_whole_arms"]) == set(run_selector.CONTROLLERS_V2)
    assert set(plan["reuse_completed_arms"]).isdisjoint(plan["rerun_whole_arms"])
    assert set(plan["reuse_completed_arms"] + plan["rerun_whole_arms"]) == set(run_selector.CONTROLLERS)
    assert sum(plan["gpu_allocation_caps_seconds"].values()) == plan["maximum_gpu_seconds"] == 10800
    for name, seconds in plan["gpu_allocation_caps_seconds"].items():
        assert run_selector.CONTROLLERS_V2[name][2] == seconds
    assert plan["maximum_gemini_calls"] == 1001
    assert plan["failure_policy"]["per_item_retries"] == 0
    assert plan["failure_policy"]["select_best_of_old_and_new"] is False


def test_rubric_and_input_identity_are_not_changed_by_recovery():
    import hashlib
    plan = json.loads((HERE / "recovery-v2-plan.json").read_text())
    assert hashlib.sha256((HERE / "selector_instructions.txt").read_bytes()).hexdigest() == plan["same_instructions_sha256"]
    assert plan["same_candidates_sha256"] == "219c745aac6f50501a55c4f2511d383505317fd75338b0efaed25f44bbe23b98"
    assert plan["same_reference_sha256"] == "1d4422de9dc1975683648a1b389d2bebf323a86f70d5e01fde9c4cccf9657604"
