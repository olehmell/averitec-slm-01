from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jeff_native as native


BODY = {
    "model": native.RETURNED_MODEL,
    "state": {"claim": "A claim.", "candidate": {"id": "C01", "text": "A snippet.", "url": "https://example.test"}},
    "questions": {"next_action": {"type": "choice", "instructions": "Fixed instructions", "criteria": {"include": None, "exclude": None}}},
}


class Request:
    def __init__(self, body):
        self.state = body["state"]
        self.questions = {
            key: type("Choice", (), {"type": value.get("type"), "criteria": value.get("criteria", {})})()
            for key, value in body["questions"].items()
        }


class RequestSchema:
    @staticmethod
    def model_validate(body):
        if body.get("questions") is None:
            raise ValueError("bad")
        return Request(body)


class Response:
    def model_dump(self):
        return {"model": native.RETURNED_MODEL, "answers": {"next_action": {"type": "choice", "choice": "include", "probabilities": {"include": 0.8, "exclude": 0.2}, "confidence": 0.6}}, "usage": {"input_tokens": 11, "output_tokens": 6}}


class Engine:
    opts = type("Opts", (), {"state_format": "kv"})()

    def __init__(self):
        self.calls = 0

    def run(self, request):
        self.calls += 1
        return Response()


def runtime(*, preflight=None):
    return native.JeffNativeRuntime(
        checkpoint="/tmp/checkpoint",
        device="cuda",
        engine=Engine(),
        backend=object(),
        modules={"SystemOneRequest": RequestSchema},
        identity={"test": True},
        preflight_override=preflight,
    )


def test_execute_uses_one_direct_engine_call_and_no_http_boundary():
    subject = runtime(preflight=lambda body: None)
    assert subject.execute(BODY)["model"] == native.RETURNED_MODEL
    assert subject._engine.calls == 1
    assert subject.identity["test"] is True


def test_preflight_failure_blocks_engine_execution():
    def blocked(_body):
        raise ValueError("jeff_preflight_state_instructions_or_options_truncated")

    subject = runtime(preflight=blocked)
    with pytest.raises(ValueError, match="truncated"):
        subject.preflight(BODY)
    assert subject._engine.calls == 0


@pytest.mark.parametrize("change", [
    {"model": "jev"},
    {"questions": {"other": {}}},
    {"questions": {"next_action": {"type": "choice", "criteria": {"include": None, "later": None}}}},
])
def test_request_schema_is_closed_before_forward(change):
    subject = runtime(preflight=lambda body: None)
    body = {**BODY, **change}
    with pytest.raises(ValueError):
        subject.execute(body)
    assert subject._engine.calls == 0


def test_runtime_requires_cuda_before_any_package_or_model_load(monkeypatch):
    monkeypatch.setattr(native, "_checkpoint_evidence", lambda *_args: pytest.fail("unexpected checkpoint access"))
    with pytest.raises(ValueError, match="cuda"):
        native.create_runtime("/absent", device="cpu")


class _Sum:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class _Tensor:
    def __init__(self, values): self.values = tuple(values)
    def equal(self, other): return isinstance(other, _Tensor) and self.values == other.values
    def sum(self): return _Sum(sum(self.values))


class _Collator:
    instances = 0
    def __init__(self, config, **_kwargs):
        self.config = config
        _Collator.instances += 1
    def __call__(self, items):
        tokens = items[0]["tokenized_text"][:self.config.max_len]
        return {"input_ids": _Tensor(tokens), "attention_mask": _Tensor([1] * len(tokens))}


class _Model:
    data_collator_class = _Collator
    def __init__(self, limit):
        self.config = SimpleNamespace(max_len=limit)
        self.data_processor = SimpleNamespace(words_splitter=lambda text: [(token, 0, 0) for token in text.split()])
    def prepare_inputs(self, texts):
        return ([text.split() for text in texts], [], [])


class _Backend:
    def __init__(self, limit):
        self.model = _Model(limit)
        self._collator = _Collator(self.model.config, data_processor=self.model.data_processor, return_tokens=True, prepare_labels=False)


class _Group:
    labels = ("include", "exclude")
    name = "Fixed instructions"
    description = None


def _native_preflight_runtime(limit):
    backend = _Backend(limit)
    modules = {
        "SystemOneRequest": RequestSchema,
        "serialize_state": lambda state, _format: state["text"],
        "build_groups": lambda _questions, _opts: [_Group()],
    }
    return native.JeffNativeRuntime(checkpoint="/tmp/checkpoint", device="cuda", engine=Engine(), backend=backend, modules=modules, identity={"test": True}), backend


def test_actual_collator_preflight_compares_unbounded_once_and_caches_it():
    _Collator.instances = 0
    subject, backend = _native_preflight_runtime(limit=8)
    body = {**BODY, "state": {"text": "all input survives"}}
    assert subject.preflight(body)["input_tokens"] == 3
    assert subject.preflight(body)["input_tokens"] == 3
    assert _Collator.instances == 2  # Jeff's actual collator plus one cached unlimited clone.
    assert subject._cached_unbounded_collator is not None


def test_actual_collator_preflight_fails_closed_on_truncation_before_forward():
    subject, _backend = _native_preflight_runtime(limit=1)
    body = {**BODY, "state": {"text": "all input survives"}}
    with pytest.raises(ValueError, match="truncated"):
        subject.preflight(body)
    assert subject._engine.calls == 0
