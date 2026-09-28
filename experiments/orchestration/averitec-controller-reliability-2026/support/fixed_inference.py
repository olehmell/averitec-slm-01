#!/usr/bin/env python3
"""Read exact, provenance-preserving evidence passages from study corpora."""

from __future__ import annotations

import json
import re
from zipfile import ZipFile

from averitec_fixed import Passage, deduplicate_passages, url_family
from source_corpora import SourceCorpora

WINDOW_CHARS = 1200
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.DOTALL)


def blocked_families(values: list[str]) -> set[str]:
    return {url_family(value) for value in values}


def archive_passages(archive: SourceCorpora, *, split: str, index: int, blocked: set[str], window_chars: int = WINDOW_CHARS) -> list[Passage]:
    archive_path, member = archive.locate(split, index)
    with ZipFile(archive_path) as source:
        # JSON strings may contain U+2028/U+2029.  Only physical LF delimits
        # archive JSONL records; str.splitlines() corrupts those valid strings.
        rows = [json.loads(line.decode("utf-8")) for line in source.read(member).split(b"\n") if line.strip()]
    passages = []
    for row in rows:
        url, texts = row.get("url"), row.get("url2text")
        if not isinstance(url, str) or not isinstance(texts, list):
            continue
        # A malformed source row must not make a usable closed-world corpus
        # unavailable.  Valid URLs still pass through url_family, which is the
        # same strict leakage normalization used by Passage and search tools.
        try:
            family = url_family(url)
        except ValueError:
            continue
        if family in blocked:
            continue
        # The span contract is over the untouched archived string.  Retrieval
        # receives bounded sentence windows, each carrying its origin offsets;
        # a long source is never silently truncated into one prompt passage.
        for value in texts:
            if isinstance(value, str) and value.strip():
                passages.extend(_source_windows(value, url, maximum_chars=window_chars))
    return deduplicate_passages(passages)


def _source_windows(source: str, url: str, *, maximum_chars: int = WINDOW_CHARS) -> list[Passage]:
    """Partition an exact archived source into bounded, offset-preserving windows."""
    if maximum_chars < 64:
        raise ValueError("window_chars")
    spans = [(match.start(), match.end()) for match in _SENTENCE.finditer(source) if source[match.start():match.end()].strip()]
    if not spans:
        return []
    result: list[Passage] = []
    start, end = spans[0]
    for sentence_start, sentence_end in spans[1:]:
        if sentence_end - start > maximum_chars and end > start:
            result.append(Passage(source[start:end], url, source_text=source, source_start=start))
            start, end = sentence_start, sentence_end
        else:
            end = sentence_end
    if end > start:
        result.append(Passage(source[start:end], url, source_text=source, source_start=start))
    expanded: list[Passage] = []
    for passage in result:
        if len(passage.text) <= maximum_chars:
            expanded.append(passage); continue
        # A sentence longer than the window remains exact, but is split at a
        # deterministic character boundary rather than altered or discarded.
        for offset in range(0, len(passage.text), maximum_chars):
            text = passage.text[offset:offset + maximum_chars]
            if text.strip():
                expanded.append(Passage(text, passage.url, source_text=source, source_start=passage.source_start + offset))
    return expanded
