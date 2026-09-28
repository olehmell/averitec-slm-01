"""One pre-commit resource ledger shared by every condition."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable


class BudgetExceeded(RuntimeError):
    """Raised when either a reservation or observed provider use breaches a cap."""


_PRIVATE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PrivateTrace:
    """Append-only, local-only raw-completion trace for one case/condition phase.

    The caller opts in with an absolute root.  A trace is always isolated at
    ``root / condition / case_id / phase.jsonl`` and is created only once;
    an existing or malformed trace fails closed rather than risking a second
    run appending unrelated completions.  Its narrowly typed writers never
    accept prompts, environment values, or credentials.
    """

    schema = "averitec-private-trace/v2"

    def __init__(self, root: Path, *, condition: str, case_id: str, phase: str) -> None:
        if not root.is_absolute() or any(not _PRIVATE_COMPONENT.fullmatch(value) for value in (condition, case_id, phase)):
            raise ValueError("private_trace_path_invalid")
        if root.exists() and root.is_symlink():
            raise ValueError("private_trace_root_unsafe")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("private_trace_root_unsafe")
        directory = root / condition / case_id
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("private_trace_path_unsafe")
        self.path = directory / f"{phase}.jsonl"
        if self.path.exists() or self.path.is_symlink():
            raise ValueError("private_trace_exists")
        self.condition, self.case_id, self.phase, self.sequence = condition, case_id, phase, 0

    def _write(self, kind: str, **payload: Any) -> None:
        self.sequence += 1
        row = {"schema": self.schema, "case_id": self.case_id, "condition": self.condition,
               "phase": self.phase, "sequence": self.sequence, "kind": kind, **payload}
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        try:
            with self.path.open("x" if self.sequence == 1 else "a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as error:
            raise ValueError("private_trace_exists") from error

    def completion(self, content: str, *, role: str, repair: bool) -> None:
        self._write("completion", role=role, repair=repair,
                    content=content, content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest())

    def qa_input(self, passages: list[dict[str, Any]], *, contract_version: str) -> None:
        self._write("qa_input", contract_version=contract_version, passages=passages)

    def qa_decision(self, *, returned_index: int, accepted: bool, reason: str,
                    contract_version: str, passage_id: str | None = None,
                    answer_start: int | None = None, answer_end: int | None = None,
                    absolute_start: int | None = None, absolute_end: int | None = None,
                    offset_types: dict[str, str] | None = None, qa_id: str | None = None) -> None:
        payload: dict[str, Any] = {"returned_index": returned_index, "accepted": accepted,
                                   "reason": reason, "contract_version": contract_version}
        if passage_id is not None:
            payload["passage_id"] = passage_id
        if answer_start is not None:
            payload["answer_start"] = answer_start
        if answer_end is not None:
            payload["answer_end"] = answer_end
        if absolute_start is not None:
            payload["absolute_start"] = absolute_start
        if absolute_end is not None:
            payload["absolute_end"] = absolute_end
        if offset_types is not None:
            payload["offset_types"] = offset_types
        if qa_id is not None:
            payload["qa_id"] = qa_id
        self._write("qa_decision", **payload)


def token_count(value: str) -> int:
    """Conservative fallback: each UTF-8 byte can be a token plus chat overhead."""
    return len(value.encode("utf-8")) + 256


@dataclass
class CaseLedger:
    caps: dict[str, int | float]
    token_counter: Callable[[str], int] = token_count
    private_trace: PrivateTrace | None = None
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    controller_calls: int = 0
    worker_calls: int = 0
    verdict_calls: int = 0
    repair_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    measured_input_tokens: int = 0
    measured_output_tokens: int = 0
    measured_calls: int = 0
    search_calls: int = 0
    delegation_calls: int = 0
    repeat_calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    process_checkpoints: list[dict[str, Any]] = field(default_factory=list)

    def _cap(self, key: str, fallback: int | float) -> int | float:
        return self.caps.get(key, fallback)

    def remaining_seconds(self) -> float:
        return max(0.0, float(self._cap("wall_clock_seconds_per_case", 900)) - (time.monotonic() - self.started))

    def _wall_ok(self) -> bool:
        return self.remaining_seconds() > 0

    def _model_reservation_failure(self, prompt: str, maximum_output: int, *, role: str, repair: bool = False) -> str | None:
        """Return the first cap that would reject a call without mutating usage."""
        input_count = self.token_counter(prompt)
        if not self._wall_ok():
            return "wall_clock_seconds_per_case"
        if self.model_calls + 1 > int(self._cap("max_model_calls_per_case", 12)):
            return "max_model_calls_per_case"
        if self.input_tokens + input_count > int(self._cap("max_input_tokens_per_case", 24576)):
            return "max_input_tokens_per_case"
        if self.output_tokens + maximum_output > int(self._cap("max_output_tokens_per_case", 9728)):
            return "max_output_tokens_per_case"
        # Every non-terminal call must leave room for one HerO verdict. This
        # is a reservation, not observed use, and prevents an optional
        # controller/repair/repeat from silently spending the final call.
        if role != "verdict":
            final_input = int(self._cap("finalization_input_token_reserve", 4096))
            final_output = int(self._cap("finalization_output_token_reserve", 500))
            if self.model_calls + 2 > int(self._cap("max_model_calls_per_case", 12)):
                return "finalization_model_call_reserve"
            if self.input_tokens + input_count + final_input > int(self._cap("max_input_tokens_per_case", 24576)):
                return "finalization_input_token_reserve"
            if self.output_tokens + maximum_output + final_output > int(self._cap("max_output_tokens_per_case", 9728)):
                return "finalization_output_token_reserve"
        if repair and self.repair_calls + 1 > int(self._cap("max_repairs_per_case", 1)):
            return "max_repairs_per_case"
        return None

    def can_reserve_model(self, prompt: str, maximum_output: int, *, role: str, repair: bool = False) -> bool:
        """Check an optional call and its reserved terminal verdict without spending."""
        return self._model_reservation_failure(prompt, maximum_output, role=role, repair=repair) is None

    def reserve_model(self, prompt: str, maximum_output: int, *, role: str, repair: bool = False) -> int:
        input_count = self.token_counter(prompt)
        failure = self._model_reservation_failure(prompt, maximum_output, role=role, repair=repair)
        if failure is not None:
            raise BudgetExceeded(failure)
        self.model_calls += 1
        self.input_tokens += input_count
        self.output_tokens += maximum_output
        self.reserved_input_tokens += input_count
        self.reserved_output_tokens += maximum_output
        if role == "controller":
            self.controller_calls += 1
        elif role == "verdict":
            self.verdict_calls += 1
        elif role != "generation":
            self.worker_calls += 1
        if repair:
            self.repair_calls += 1
        return input_count

    def settle_model(self, maximum_output: int, actual_output: str, *, reserved_input: int, reported_input: int | None = None, reported_output: int | None = None) -> None:
        """Reconcile a successful call and fail if measured use crosses a cap."""
        if reported_input is not None:
            self.input_tokens += reported_input - reserved_input
            self.measured_input_tokens += reported_input
        if reported_output is not None:
            self.output_tokens += reported_output - maximum_output
            self.measured_output_tokens += reported_output
        # Without a provider usage receipt, retain the complete output
        # reservation.  A regex estimate could undercount tokenizer output.
        if reported_input is not None and reported_output is not None:
            self.measured_calls += 1
        if self.input_tokens > int(self._cap("max_input_tokens_per_case", 24576)):
            raise BudgetExceeded("observed_max_input_tokens_per_case")
        if self.output_tokens > int(self._cap("max_output_tokens_per_case", 9728)):
            raise BudgetExceeded("observed_max_output_tokens_per_case")
        if not self._wall_ok():
            raise BudgetExceeded("wall_clock_seconds_per_case")

    def reserve_search(self, tool: str, arguments: dict[str, Any]) -> None:
        if not self._wall_ok():
            raise BudgetExceeded("wall_clock_seconds_per_case")
        if self.search_calls + 1 > int(self._cap("max_search_calls_per_case", 4)):
            raise BudgetExceeded("max_search_calls_per_case")
        self.search_calls += 1
        self.events.append({"kind": "tool", "tool": tool, "arguments": arguments})

    def reserve_finalization(self, *, maximum_output: int = 500) -> bool:
        """Check that one terminal verdict remains affordable before a repeat."""
        available = self.finalization_available(maximum_output=maximum_output)
        self.event("finalization_reservation", available=available, maximum_output=maximum_output)
        return available

    def finalization_available(self, *, maximum_output: int = 500) -> bool:
        """Return terminal HerO affordability without mutating the public trace."""
        return (self._wall_ok()
                and self.model_calls + 1 <= int(self._cap("max_model_calls_per_case", 12))
                and self.input_tokens + int(self._cap("finalization_input_token_reserve", 4096)) <= int(self._cap("max_input_tokens_per_case", 24576))
                and self.output_tokens + maximum_output <= int(self._cap("max_output_tokens_per_case", 9728)))

    def delegation(self, *, role: str, repeat: bool = False) -> None:
        self.delegation_calls += 1
        if repeat:
            self.repeat_calls += 1
        self.event("delegation", role=role, repeat=repeat, sequential=True)

    def repeat(self, *, action: str, number: int) -> None:
        """Account for one executor-scheduled repeat without implying delegation."""
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError("repeat_number")
        self.repeat_calls += 1
        self.event("repeat", action=action, number=number, sequential=True)

    def tool_result(self, result: Any) -> None:
        self.events[-1]["result"] = result

    def restore_usage(self, usage: dict[str, Any]) -> None:
        """Continue a stage's single case budget in a later process."""
        if self.model_calls or self.search_calls or self.events:
            raise ValueError("ledger_restore_requires_fresh_ledger")
        fields = (
            "model_calls", "controller_calls", "worker_calls", "verdict_calls", "repair_calls",
            "input_tokens", "output_tokens", "reserved_input_tokens", "reserved_output_tokens",
            "measured_input_tokens", "measured_output_tokens", "measured_calls", "search_calls", "delegation_calls", "repeat_calls",
        )
        values = {field: usage.get(field, 0) if field in {"delegation_calls", "repeat_calls"} else usage.get(field) for field in fields}
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("ledger_restore_usage_schema")
        elapsed = usage.get("wall_clock_seconds")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
            raise ValueError("ledger_restore_usage_schema")
        for field, value in values.items():
            setattr(self, field, value)
        if self.model_calls > int(self._cap("max_model_calls_per_case", 12)):
            raise ValueError("ledger_restore_model_cap")
        if self.search_calls > int(self._cap("max_search_calls_per_case", 4)):
            raise ValueError("ledger_restore_search_cap")
        if self.repair_calls > int(self._cap("max_repairs_per_case", 1)):
            raise ValueError("ledger_restore_repair_cap")
        if self.input_tokens > int(self._cap("max_input_tokens_per_case", 24576)) or self.output_tokens > int(self._cap("max_output_tokens_per_case", 9728)):
            raise ValueError("ledger_restore_token_cap")
        self.started = time.monotonic() - float(elapsed)

    def event(self, kind: str, **summary: Any) -> None:
        """Append an auditable, bounded semantic summary; never prompts/output text."""
        self.events.append({"kind": kind, **summary})

    def process_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Retain one gold-free evidence snapshot for evaluator-only trajectory scoring."""
        if not isinstance(checkpoint, dict):
            raise ValueError("process_checkpoint_schema")
        # Round-trip copying prevents later state mutation from rewriting a
        # checkpoint that has already been observed.
        copied = json.loads(json.dumps(checkpoint, sort_keys=True, allow_nan=False))
        self.process_checkpoints.append(copied)

    def private_completion(self, content: str, *, role: str, repair: bool) -> None:
        """Persist a raw completion only when the caller opted into a private trace."""
        if self.private_trace is not None:
            self.private_trace.completion(content, role=role, repair=repair)

    def private_qa_input(self, passages: list[dict[str, Any]], *, contract_version: str) -> None:
        """Bind QA to ranked source-window identifiers/hashes, never its prompt."""
        if self.private_trace is not None:
            self.private_trace.qa_input(passages, contract_version=contract_version)

    def private_qa_decision(self, *, returned_index: int, accepted: bool, reason: str,
                            contract_version: str, passage_id: str | None = None,
                            answer_start: int | None = None, answer_end: int | None = None,
                            absolute_start: int | None = None, absolute_end: int | None = None,
                            offset_types: dict[str, str] | None = None,
                            qa_id: str | None = None) -> None:
        if self.private_trace is not None:
            self.private_trace.qa_decision(
                returned_index=returned_index, accepted=accepted, reason=reason,
                contract_version=contract_version, passage_id=passage_id,
                answer_start=answer_start, answer_end=answer_end,
                absolute_start=absolute_start, absolute_end=absolute_end,
                offset_types=offset_types, qa_id=qa_id,
            )

    def usage(self) -> dict[str, Any]:
        accounting = "measured" if self.measured_calls == self.model_calls else "reserved_or_estimated"
        return {
            "model_calls": self.model_calls, "controller_calls": self.controller_calls,
            "worker_calls": self.worker_calls, "verdict_calls": self.verdict_calls,
            "repair_calls": self.repair_calls,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "reserved_input_tokens": self.reserved_input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "measured_input_tokens": self.measured_input_tokens,
            "measured_output_tokens": self.measured_output_tokens,
            "measured_calls": self.measured_calls,
            "token_accounting": accounting, "search_calls": self.search_calls,
            "delegation_calls": self.delegation_calls, "repeat_calls": self.repeat_calls,
            "wall_clock_seconds": round(time.monotonic() - self.started, 6),
            "gpu_seconds": None, "estimated_cost_usd": None,
        }
