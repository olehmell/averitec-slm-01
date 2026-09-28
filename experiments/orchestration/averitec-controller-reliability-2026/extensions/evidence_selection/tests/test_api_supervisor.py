import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("api_supervisor", Path(__file__).parents[1] / "api_supervisor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_credentials_are_literal_not_shell(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    path = tmp_path / "keys"
    path.write_text('UNRELATED=ignored\nexport TYPESAFE_API_KEY="literal$(not-executed)"\n')
    assert module.credential("TYPESAFE_API_KEY", path) == "literal$(not-executed)"


def test_exclusive_dispatch_intent(tmp_path):
    path = tmp_path / "intent.json"
    module.write_new(path, {"retries": 0})
    with pytest.raises(FileExistsError):
        module.write_new(path, {"retries": 1})


def test_environment_key_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "only-test-placeholder")
    assert module.credential("GEMINI_API_KEY", tmp_path / "absent") == "only-test-placeholder"


def test_v2_rejects_jev_before_credentials_or_dispatch(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "launch-manifest.json").write_text(json.dumps({"schema": "averitec-selector-launch/v2", "controllers": {"jev": {}}}))
    with pytest.raises(ValueError, match="v2_only_gemini_api_authorized"):
        module.supervise("jev", tmp_path, inputs, tmp_path, tmp_path / "absent-secret")
    assert not (tmp_path / "jev.supervisor").exists()


def test_api_cap_mismatch_rejected_before_credentials(tmp_path):
    (tmp_path / "launch-manifest.json").write_text(json.dumps({"schema": "averitec-selector-launch/v2", "controllers": {"gemini": {"wall_seconds": 14400, "maximum_calls": 2000}}}))
    with pytest.raises(ValueError, match="api_launch_cap_mismatch"):
        module.supervise("gemini", tmp_path, tmp_path, tmp_path, tmp_path / "absent-secret")
