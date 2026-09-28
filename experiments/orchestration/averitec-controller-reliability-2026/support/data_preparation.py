"""Validate label-free runtime rows prepared from the frozen study split."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterator


RUNTIME_FIELDS = frozenset({"case_id", "claim", "split"})
CASE_ID_RE = re.compile(r"averitec-(dev|train)-(\d{4})$")


def load_runtime_claims(path: Path) -> Iterator[dict[str, str]]:
    """Read the v5 runtime view without opening labels or reference evidence."""
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != RUNTIME_FIELDS:
                raise ValueError(f"runtime_fields:{line_number}")
            identifier, claim, split = record.get("case_id"), record.get("claim"), record.get("split")
            match = CASE_ID_RE.fullmatch(identifier) if isinstance(identifier, str) else None
            if not match or not isinstance(claim, str) or not claim.strip() or split != match.group(1):
                raise ValueError(f"runtime_record:{line_number}")
            if identifier in seen:
                raise ValueError(f"runtime_case_id_duplicate:{identifier}")
            seen.add(identifier)
            yield {"case_id": identifier, "claim": claim, "split": split}
