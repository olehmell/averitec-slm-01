"""Isolated, pinned Jeff/GLiFormer runtime for the evidence-selection arm.

This module is deliberately not an HTTP client and does not start Jeff's
server or batcher.  ``create_runtime`` loads the pinned Jeff source directly,
with its real eager ``TorchBackend`` and ``Engine``.  The returned object has
the small boundary consumed by :mod:`jeff_adapter`: ``preflight(body)``,
``execute(body)``, and immutable identity evidence.

Preflight performs no model forward.  It constructs the exact item used by
Jeff's backend, invokes the actual GLiFormer collator, and compares it to an
otherwise identical collator whose explicit sequence-length limits have been
lifted.  If that comparison cannot be made, it fails closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable


JEFF_COMMIT = "34b32f99a727c47b679adde33f4702a001e02979"
JEFF_TREE = "8127f6ea6caae9e19f4fd57154d379ee481c2d5d"
MODEL = "knowledgator/gliformer-large-v1"
MODEL_REVISION = "d0a4e53d09cebe6bc963dd9be319d4279084bb2d"
RETURNED_MODEL = "gliformer-large-v1"
GLIFORMER_VERSION = "0.1.2"
GLIFORMER_WHEEL_SHA256 = "b9a1194b56fc0ab193fef919ad68d0b3086ce2fe3f0b4392ce26873139ac8768"
GLINER_VERSION = "0.2.29"
GLINER_WHEEL_SHA256 = "0c8cfb9f5c2daf7aff329ebb5ab1609052d7ad0aca0ef26049b9476b018b3fde"
WEIGHTS_FILE = "pytorch_model.bin"
WEIGHTS_SHA256 = "f80b29199d66f878669f283703e4dba9fd726755dcc20aba1ed0d24fce4a23f1"
RUNTIME_SEED = 20260919

# These are the only upstream files this direct, serverless path imports.
_JEFF_SOURCE_HASHES = {
    "jeff.core.engine": "51a5a615c1f63181769185699c3c0b3fb26fd710e6e9a18180b7cd17f8a91f0d",
    "jeff.backends.torch_backend": "356e715a53742af2e630a6e3ed8977a7d944be671f637cb057cd113680c548f4",
    "jeff.core.groups": "453eac70757f6a4c1c7e551e0ea12d54e94e4377aab044bb0e2544b95b5bda29",
    "jeff.core.state": "1c2c2bebb5c5dd9538384a7448ad7a50fa4d569c43cb78d8ecbc9b94466070b7",
    "jeff.core.schemas": "8047003d5fe864f305bc95d9ce79cb38f60c36b646626c895474ac7aca815a9c",
}
_LENGTH_LIMIT_NAMES = {
    "max_len",
    "max_length",
    "max_seq_len",
    "max_seq_length",
    "max_tokens",
    "max_token_length",
}
_UNBOUNDED_LIMIT = 1_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_evidence(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise RuntimeError("jeff_checkpoint_missing_or_unsafe")
    weights = checkpoint / WEIGHTS_FILE
    if not weights.is_file() or weights.is_symlink():
        raise RuntimeError("jeff_pytorch_weights_missing")
    if (checkpoint / "model.safetensors").exists():
        raise RuntimeError("jeff_unpinned_safetensors_present")
    weight_digest = _sha256_file(weights)
    if weight_digest != WEIGHTS_SHA256:
        raise RuntimeError("jeff_pytorch_weights_digest_mismatch")
    files: dict[str, str] = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files[str(path.relative_to(checkpoint))] = _sha256_file(path)
        elif path.is_symlink():
            raise RuntimeError("jeff_checkpoint_symlink_forbidden")
    required = {"gliner_config.json", WEIGHTS_FILE, "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(files):
        raise RuntimeError("jeff_checkpoint_incomplete")
    prefixed = {"checkpoint/" + name: digest for name, digest in files.items()}
    return {
        "path": str(checkpoint),
        "revision": MODEL_REVISION,
        "files_sha256": files,
        "checkpoint_sha256": hashlib.sha256(
            json.dumps(prefixed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }


def _distribution_evidence(name: str, expected_version: str, expected_wheel_sha256: str | None = None) -> dict[str, Any]:
    try:
        installed = metadata.version(name)
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"jeff_dependency_missing:{name}") from None
    if installed != expected_version:
        raise RuntimeError(f"jeff_dependency_version_mismatch:{name}")
    record = Path(distribution._path) / "RECORD"  # importlib.metadata public data has no RECORD path API.
    if not record.is_file():
        raise RuntimeError(f"jeff_dependency_record_missing:{name}")
    return {
        "version": installed,
        "record_sha256": _sha256_file(record),
        "pinned_wheel_sha256": expected_wheel_sha256,
    }


def _module_source_evidence(module_name: str, expected_sha256: str) -> str:
    module = importlib.import_module(module_name)
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError(f"jeff_source_unavailable:{module_name}")
    actual = _sha256_file(Path(source))
    if actual != expected_sha256:
        raise RuntimeError(f"jeff_source_hash_mismatch:{module_name}")
    return actual


def _loaded_modules() -> dict[str, Any]:
    actual = {name: _module_source_evidence(name, digest) for name, digest in _JEFF_SOURCE_HASHES.items()}
    return {
        "Engine": importlib.import_module("jeff.core.engine").Engine,
        "PromptOptions": importlib.import_module("jeff.core.groups").PromptOptions,
        "build_groups": importlib.import_module("jeff.core.groups").build_groups,
        "SystemOneRequest": importlib.import_module("jeff.core.schemas").SystemOneRequest,
        "serialize_state": importlib.import_module("jeff.core.state").serialize_state,
        "TorchBackend": importlib.import_module("jeff.backends.torch_backend").TorchBackend,
        "source_hashes": actual,
    }


def _safe_torch_backend(torch_backend: Any, checkpoint: Path, device: str) -> Any:
    """Load locally with safe torch deserialization and without hub access."""
    torch = importlib.import_module("torch")
    real_load = torch.load

    def weights_only_load(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is False:
            raise RuntimeError("jeff_unsafe_torch_load_rejected")
        kwargs["weights_only"] = True
        return real_load(*args, **kwargs)

    old_env = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")}
    try:
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
        torch.load = weights_only_load
        # This is the exact Jeff TorchBackend path, constrained to deterministic
        # eager, batch-one, no-compile, no-startup-warmup operation.
        return torch_backend(
            str(checkpoint),
            device=device,
            attn_kernel="eager",
            compile_model=False,
            warmup=False,
            batch_size=1,
        )
    finally:
        torch.load = real_load
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _set_unbounded_limits(value: Any, seen: set[int] | None = None) -> int:
    """Lift only explicit sequence-length limits on a private collator copy."""
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen or isinstance(value, (str, bytes, bytearray, type(None), bool, int, float)):
        return 0
    seen.add(marker)
    changed = 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _LENGTH_LIMIT_NAMES and type(child) is int and child > 0:
                value[key] = _UNBOUNDED_LIMIT  # type: ignore[index]
                changed += 1
            else:
                changed += _set_unbounded_limits(child, seen)
        return changed
    if isinstance(value, list):
        for child in value:
            changed += _set_unbounded_limits(child, seen)
        return changed
    if isinstance(value, tuple):
        for child in value:
            changed += _set_unbounded_limits(child, seen)
        return changed
    try:
        values = vars(value)
    except TypeError:
        return changed
    for key, child in values.items():
        if key.lower() in _LENGTH_LIMIT_NAMES and type(child) is int and child > 0:
            try:
                setattr(value, key, _UNBOUNDED_LIMIT)
            except (AttributeError, TypeError):
                raise RuntimeError("jeff_preflight_limit_not_mutable") from None
            changed += 1
        else:
            changed += _set_unbounded_limits(child, seen)
    return changed


def _equal_batch(left: Any, right: Any) -> bool:
    """Structural equality for collator outputs, including torch tensors."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_equal_batch(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal_batch(a, b) for a, b in zip(left, right))
    equal = getattr(left, "equal", None)
    if callable(equal):
        try:
            return bool(equal(right))
        except (TypeError, RuntimeError):
            return False
    return left == right


class JeffNativeRuntime:
    """Direct, one-request native Jeff runtime; it contains no retry or fallback."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: str,
        engine: Any,
        backend: Any,
        modules: Mapping[str, Any],
        identity: Mapping[str, Any],
        preflight_override: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._checkpoint = Path(checkpoint)
        self._device = device
        self._engine, self._backend, self._modules = engine, backend, dict(modules)
        self._preflight_override = preflight_override
        # The five sealed identity fields stay flat for the bounded runner;
        # the evidence envelope is persisted alongside them in its receipt.
        self.identity = dict(identity)
        self.evidence = MappingProxyType(dict(identity.get("evidence", {})))
        self._cached_unbounded_collator: Any | None = None

    def _request(self, body: dict[str, Any]) -> Any:
        if not isinstance(body, dict) or body.get("model") != RETURNED_MODEL:
            raise ValueError("jeff_native_model_required")
        try:
            request = self._modules["SystemOneRequest"].model_validate(body)
        except Exception:
            raise ValueError("jeff_native_request_invalid") from None
        questions = request.questions
        if set(questions) != {"next_action"}:
            raise ValueError("jeff_native_single_binary_question_required")
        question = questions["next_action"]
        if getattr(question, "type", None) != "choice" or set(question.criteria) != {"include", "exclude"}:
            raise ValueError("jeff_native_binary_options_required")
        return request

    def _item(self, request: Any) -> tuple[dict[str, Any], int]:
        text = self._modules["serialize_state"](request.state, self._engine.opts.state_format)
        groups = self._modules["build_groups"](request.questions, self._engine.opts)
        if len(groups) != 1 or tuple(groups[0].labels) != ("include", "exclude"):
            raise RuntimeError("jeff_preflight_group_mismatch")
        prepared = self._backend.model.prepare_inputs([text])
        if not isinstance(prepared, tuple) or len(prepared) != 3 or len(prepared[0]) != 1:
            raise RuntimeError("jeff_preflight_prepare_inputs_invalid")
        tokenized_text = prepared[0][0]
        # GLiFormer 0.1.2 ``prepare_inputs`` is source-verified above.  It is
        # just the word splitter, but compare its output directly so an
        # unexpectedly different installed model path cannot hide a trim before
        # the collator sees the text.
        try:
            expected_tokens = [token for token, _start, _end in self._backend.model.data_processor.words_splitter(text)]
        except Exception:
            raise RuntimeError("jeff_preflight_wordsplitter_unavailable") from None
        if list(tokenized_text) != expected_tokens:
            raise RuntimeError("jeff_preflight_prepare_inputs_trimmed")
        item = {
            "tokenized_text": tokenized_text,
            "classification": [{
                "name": groups[0].name,
                "description": groups[0].description,
                "all_labels": list(groups[0].labels),
                "true_labels": [],
            }],
        }
        return item, len(text)

    def _unbounded_collator(self) -> Any:
        if self._cached_unbounded_collator is not None:
            return self._cached_unbounded_collator
        try:
            config = deepcopy(self._backend.model.config)
            processor = deepcopy(self._backend.model.data_processor)
        except Exception:
            raise RuntimeError("jeff_preflight_copy_unavailable") from None
        changed = _set_unbounded_limits(config) + _set_unbounded_limits(processor)
        if changed == 0:
            raise RuntimeError("jeff_preflight_unbounded_limit_unknown")
        collator_class = self._backend.model.data_collator_class
        if collator_class is None:
            try:
                collator_class = importlib.import_module("gliformer.gliformer").resolve_gliformer_collator_class(config)
            except Exception:
                raise RuntimeError("jeff_preflight_collator_unavailable") from None
        try:
            self._cached_unbounded_collator = collator_class(
                config, data_processor=processor, return_tokens=True, prepare_labels=False
            )
            return self._cached_unbounded_collator
        except Exception:
            raise RuntimeError("jeff_preflight_unbounded_collator_unavailable") from None

    def preflight(self, body: dict[str, Any]) -> dict[str, int]:
        """Prove that the exact state, instructions, and options survive collation."""
        if self._preflight_override is not None:
            self._preflight_override(body)
            return {"input_tokens": 0, "state_characters": 0}
        request = self._request(body)
        item, state_characters = self._item(request)
        try:
            actual = self._backend._collator([item])
            unlimited = self._unbounded_collator()([item])
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError("jeff_preflight_collation_failed") from None
        if not _equal_batch(actual, unlimited):
            raise ValueError("jeff_preflight_state_instructions_or_options_truncated")
        attention = actual.get("attention_mask") if isinstance(actual, Mapping) else None
        if attention is None or not hasattr(attention, "sum"):
            raise RuntimeError("jeff_preflight_attention_mask_missing")
        try:
            tokens = int(attention.sum().item())
        except Exception:
            raise RuntimeError("jeff_preflight_attention_mask_invalid") from None
        if tokens < 1:
            raise RuntimeError("jeff_preflight_empty_sequence")
        return {"input_tokens": tokens, "state_characters": state_characters}

    def execute(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute exactly one eager Engine request after the caller's preflight."""
        request = self._request(body)
        try:
            torch = self._modules.get("torch")
            if torch is not None:
                torch.cuda.synchronize(self._device)
            response = self._engine.run(request)
            if torch is not None:
                torch.cuda.synchronize(self._device)
            payload = response.model_dump()
        except Exception:
            raise RuntimeError("jeff_native_forward_failed") from None
        if not isinstance(payload, dict) or payload.get("model") != RETURNED_MODEL:
            raise RuntimeError("jeff_native_response_identity_mismatch")
        return payload


def create_runtime(checkpoint: str | Path, device: str = "cuda") -> JeffNativeRuntime:
    """Load the local pinned checkpoint without downloads, warmups, compile, or fallbacks."""
    if device != "cuda":
        raise ValueError("jeff_cuda_required")
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_evidence = _checkpoint_evidence(checkpoint_path)
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("jeff_exactly_one_cuda_device_required")
    torch.manual_seed(RUNTIME_SEED)
    torch.cuda.manual_seed_all(RUNTIME_SEED)
    dependencies = {
        "gliformer": _distribution_evidence("gliformer", GLIFORMER_VERSION, GLIFORMER_WHEEL_SHA256),
        "gliner": _distribution_evidence("gliner", GLINER_VERSION, GLINER_WHEEL_SHA256),
    }
    for optional in ("torch", "transformers", "scipy", "accelerate", "numpy", "pydantic"):
        try:
            dependencies[optional] = {"version": metadata.version(optional)}
        except metadata.PackageNotFoundError:
            dependencies[optional] = {"version": None}
    modules = _loaded_modules()
    modules["torch"] = torch
    backend = _safe_torch_backend(modules["TorchBackend"], checkpoint_path, device)
    if getattr(backend, "device", None) != "cuda" or getattr(backend, "compiled", None) is not False:
        raise RuntimeError("jeff_runtime_mode_mismatch")
    if getattr(backend, "batch_size", None) != 1 or getattr(backend, "attn_kernel", None) != "eager":
        raise RuntimeError("jeff_runtime_mode_mismatch")
    try:
        parameter_count = sum(parameter.numel() for parameter in backend.model.parameters())
    except Exception:
        raise RuntimeError("jeff_parameter_count_unavailable") from None
    load_keys = {
        "missing": getattr(backend.model, "missing_keys", None),
        "unexpected": getattr(backend.model, "unexpected_keys", None),
    }
    options = modules["PromptOptions"](state_format="kv", isolate="nouls")
    engine = modules["Engine"](backend, RETURNED_MODEL, options)
    evidence = {
        "adapter": "jeff_native_runtime/v1",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "returned_model": RETURNED_MODEL,
        "upstream": {"repository": "https://github.com/logan-markewich/jeff", "commit": JEFF_COMMIT, "tree": JEFF_TREE, "source_sha256": modules["source_hashes"]},
        "checkpoint": checkpoint_evidence,
        "dependencies": dependencies,
        "execution": {"backend": "torch", "requested_device": "cuda", "actual_device": backend.device, "attention": "eager", "compiled": False, "batch_size": 1, "startup_warmup": False, "http_server": False, "fallback": "forbidden", "seed": RUNTIME_SEED, "parameter_count": parameter_count, "checkpoint_load_keys": load_keys},
        "loading": {"offline": True, "trust_remote_code": False, "torch_load_weights_only": True},
        "qualification": "runtime_loaded_but_native_forward_not_yet_qualified",
    }
    identity = {
        "upstream_commit": JEFF_COMMIT,
        "model": MODEL,
        "returned_model": RETURNED_MODEL,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": checkpoint_evidence["checkpoint_sha256"],
        "evidence": evidence,
    }
    return JeffNativeRuntime(checkpoint=checkpoint_path, device=device, engine=engine, backend=backend, modules=modules, identity=identity)
