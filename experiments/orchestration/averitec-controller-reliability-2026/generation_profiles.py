"""Pinned, JSON-safe generation profiles for OpenAI-compatible controllers.

Profiles are deployment identities, not adaptive fallbacks.  Resolving one is
therefore deliberately strict: a profile for one model family cannot silently
be applied to another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ModelFamily = Literal["any", "lfm", "qwen"]


@dataclass(frozen=True)
class GenerationProfile:
    """A complete, named request policy with only JSON-serializable fields."""

    name: str
    model_family: ModelFamily
    temperature: float
    max_tokens: int
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    enable_thinking: bool | None = None
    timeout_seconds: float | None = None
    exact_model: str | None = None
    # LFM2.5-2.6B reasons unconditionally in its deployment.  This is a
    # receipt-level capability declaration, not Qwen's chat-template toggle.
    reasoning_mode: Literal["always"] | None = None

    def compatible_with(self, model: str) -> bool:
        normalized = model.lower()
        if self.exact_model is not None:
            return normalized == self.exact_model.lower()
        # The 2.6B deployment has a separate, long-budget, always-reasoning
        # profile.  Never let an omitted/legacy profile silently downgrade it
        # to the 160-token policy intended for the 1.2B deployment.
        if normalized == LFM26_MODEL.lower():
            return False
        if self.model_family == "any":
            return True
        return self.model_family in normalized

    def request_parameters(self, model: str) -> dict[str, Any]:
        """Return the extra OpenAI-compatible request fields for ``model``.

        ``enable_thinking`` is a Qwen chat-template extension, rather than an
        OpenAI field.  It is intentionally never sent to LFM endpoints.
        """
        if not self.compatible_with(model):
            raise ValueError("generation_profile_model_incompatible")
        result: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        for key, value in (
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("presence_penalty", self.presence_penalty),
            ("repetition_penalty", self.repetition_penalty),
        ):
            if value is not None:
                result[key] = value
        if "qwen" in model.lower() and self.enable_thinking is not None:
            result["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return result

    def identity(self, model: str) -> dict[str, Any]:
        """Return a receipt-safe identity; never include endpoints or secrets."""
        if not self.compatible_with(model):
            raise ValueError("generation_profile_model_incompatible")
        result = {
            "name": self.name,
            "model_family": self.model_family,
            "model": model,
            "request_parameters": self.request_parameters(model),
            "timeout_seconds": self.timeout_seconds,
        }
        if self.reasoning_mode is not None:
            result["reasoning_mode"] = self.reasoning_mode
        return result


LFM26_MODEL = "LiquidAI/LFM2.5-2.6B"


_PROFILES = {
    # Exact pre-profile behaviour: temperature zero, 160 output tokens, and
    # Qwen thinking disabled.  The global nine-action vocabulary and JSON
    # schema remain the responsibility of OpenAIController/engine.
    "baseline": GenerationProfile(
        name="baseline", model_family="any", temperature=0, max_tokens=160,
        enable_thinking=False,
    ),
    "lfm_native": GenerationProfile(
        name="lfm_native", model_family="lfm", temperature=0.1, max_tokens=160,
        top_k=50, repetition_penalty=1.05, enable_thinking=False,
    ),
    "lfm26_native": GenerationProfile(
        name="lfm26_native", model_family="lfm", temperature=0.1, max_tokens=2048,
        top_k=50, repetition_penalty=1.1, timeout_seconds=60,
        exact_model=LFM26_MODEL, reasoning_mode="always",
    ),
    "qwen_native": GenerationProfile(
        name="qwen_native", model_family="qwen", temperature=0.7, max_tokens=160,
        top_p=0.8, top_k=20, presence_penalty=1.5, repetition_penalty=1,
        enable_thinking=False,
    ),
    "qwen_thinking": GenerationProfile(
        name="qwen_thinking", model_family="qwen", temperature=1, max_tokens=2048,
        top_p=0.95, top_k=20, presence_penalty=1.5, repetition_penalty=1,
        enable_thinking=True, timeout_seconds=60,
    ),
}


def resolve_generation_profile(name: str, model: str) -> GenerationProfile:
    """Resolve a known profile and reject an incompatible model before I/O."""
    if not isinstance(name, str) or name not in _PROFILES:
        raise ValueError("unknown_generation_profile")
    if not isinstance(model, str) or not model:
        raise ValueError("invalid_model")
    profile = _PROFILES[name]
    if not profile.compatible_with(model):
        raise ValueError("generation_profile_model_incompatible")
    return profile


def generation_profile_identity(name: str, model: str) -> dict[str, Any]:
    """Resolve and serialize a profile for a run/deployment receipt."""
    return resolve_generation_profile(name, model).identity(model)
