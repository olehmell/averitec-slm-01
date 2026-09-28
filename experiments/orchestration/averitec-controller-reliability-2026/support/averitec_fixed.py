"""Closed-world, HerO-shaped search surface for the fixed diagnostic runner.

The model-facing tool is called ``search_web`` so its interaction pattern is
that of ordinary web search.  The implementation deliberately never performs a
network request: it reads only the immutable per-claim corpus supplied to a
run.  Gold fields are rejected before a claim reaches this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable
from urllib.parse import unquote, urlsplit, urlunsplit


FORBIDDEN_RUNTIME_FIELDS = frozenset({"label", "justification", "questions", "answers", "source_url", "cached_source_url"})


@dataclass(frozen=True)
class Passage:
    """A locally scraped passage with stable provenance identifiers.

    ``text, url`` remains the positional constructor used by the fixed
    diagnostic.  The identifiers are derived when callers do not provide one,
    so a passage can safely cross process and archive boundaries.
    """

    text: str
    url: str
    passage_id: str | None = None
    # Retrieval may use a bounded window while evidence remains anchored to
    # the unmodified archived source and its original character offsets.
    source_text: str | None = None
    source_start: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("passage_text")
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("passage_url")
        expected = stable_passage_id(self.url, self.text)
        if self.passage_id is None:
            object.__setattr__(self, "passage_id", expected)
        elif self.passage_id != expected:
            raise ValueError("passage_id")
        raw = self.text if self.source_text is None else self.source_text
        if not isinstance(raw, str) or not raw.strip() or not isinstance(self.source_start, int) or isinstance(self.source_start, bool):
            raise ValueError("passage_source")
        if self.source_start < 0 or self.source_start + len(self.text) > len(raw) or raw[self.source_start:self.source_start + len(self.text)] != self.text:
            raise ValueError("passage_source_span")
        object.__setattr__(self, "source_text", raw)

    @property
    def url_id(self) -> str:
        return "url-" + _sha256(url_family(self.url))

    @property
    def content_id(self) -> str:
        return "content-" + _sha256(_normal_text(self.text))


def validate_runtime_claim(record: dict[str, object]) -> None:
    forbidden = FORBIDDEN_RUNTIME_FIELDS.intersection(record)
    if forbidden:
        raise ValueError("runtime_gold_field:" + ",".join(sorted(forbidden)))
    if not isinstance(record.get("claim"), str) or not record["claim"].strip():
        raise ValueError("runtime_claim")
    case_id = record.get("case_id")
    if case_id is not None and (not isinstance(case_id, str) or not re.fullmatch(r"averitec-(?:dev|train|test)-\d{4}", case_id)):
        raise ValueError("runtime_case_id")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normal_text(value: str) -> str:
    return " ".join(value.split())


def url_family(value: str) -> str:
    """Return the conservative origin key used for leakage exclusion.

    It intentionally merges only http/https, leading ``www.`` and Wayback
    captures of the *same* URL.  Path and query are retained, so unrelated
    pages on one host never collapse into one deny-list item.
    """
    candidate = value.strip()
    parsed = urlsplit(candidate)
    if parsed.hostname and parsed.hostname.casefold() == "web.archive.org":
        # A Wayback capture embeds the original URL after /web/<capture>/.
        match = re.match(r"^/web/[^/]+/(https?://.+)$", unquote(parsed.path), flags=re.IGNORECASE)
        if match:
            candidate = match.group(1)
            if parsed.query:
                candidate += "?" + parsed.query
            parsed = urlsplit(candidate)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url_family")
    host = parsed.hostname.casefold()
    if host.startswith("www."):
        host = host[4:]
    port = parsed.port
    netloc = host if port in {None, 80, 443} else f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(("https", netloc, path, parsed.query, ""))


def stable_passage_id(url: str, text: str) -> str:
    return "passage-" + _sha256(url_family(url) + "\n" + _normal_text(text))


def deduplicate_passages(passages: Iterable[Passage]) -> list[Passage]:
    """Keep the first deterministic occurrence of each content/URL pair."""
    kept: list[Passage] = []
    seen: set[str] = set()
    for passage in passages:
        if passage.passage_id not in seen:
            seen.add(str(passage.passage_id))
            kept.append(passage)
    return kept


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w]+", value.lower()))


def search_web(query: str, corpus: Iterable[Passage], *, blocked_urls: set[str], limit: int) -> list[Passage]:
    """Return deterministic offline search results from one filtered corpus."""
    query_tokens = _tokens(query)
    blocked = {url_family(url) for url in blocked_urls}
    ranked = []
    for position, passage in enumerate(corpus):
        if url_family(passage.url) in blocked:
            continue
        overlap = len(query_tokens.intersection(_tokens(passage.text)))
        ranked.append((-overlap, position, passage))
    return [passage for _, _, passage in sorted(ranked)[:limit]]
