"""Shared closed-world tool surface.  No tool performs network I/O."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import hashlib
from typing import Any, Callable, Iterable

from averitec_fixed import Passage, deduplicate_passages, url_family
from inference.ledger import CaseLedger


TOKEN_RE = re.compile(r"[\w]+")
ALLOWED_TOOLS = frozenset({
    "decompose_claim", "formulate_queries", "search_sparse", "search_dense",
    "generate_qa", "assess_coverage", "select_evidence", "predict_verdict",
})


def words(value: str) -> list[str]:
    return TOKEN_RE.findall(value.casefold())


def _score_bm25(query: str, corpus: list[Passage]) -> list[tuple[float, int, Passage]]:
    terms = words(query)
    if not corpus:
        return []
    document_terms = [words(item.text) for item in corpus]
    frequencies = Counter(term for terms_ in document_terms for term in set(terms_))
    average = sum(len(terms_) for terms_ in document_terms) / len(document_terms)
    scored: list[tuple[float, int, Passage]] = []
    for position, (passage, values) in enumerate(zip(corpus, document_terms)):
        counts = Counter(values)
        score = 0.0
        for term in terms:
            if term not in counts:
                continue
            idf = math.log(1 + (len(corpus) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            score += idf * counts[term] * 2.2 / (counts[term] + 1.2 * (1 - 0.75 + 0.75 * len(values) / max(average, 1)))
        scored.append((-score, position, passage))
    return sorted(scored)


class LazyDenseEmbedder:
    """Pinned Qwen3 embedding model with an isolated vector-cache namespace."""

    _model: Any = None
    _passage_vectors: dict[str, Any] = {}
    model_id = "Qwen/Qwen3-Embedding-4B"
    revision = "5cf2132abc99cad020ac570b19d031efec650f2b"
    _cache_namespace = "Qwen/Qwen3-Embedding-4B:5cf2132abc99cad020ac570b19d031efec650f2b:maxseq4096:bf16"
    query_instruction = "Given a fact-checking claim, retrieve source passages that can verify or refute it."

    def __init__(self, embed: Callable[[list[str]], Any] | None = None) -> None:
        self._embed = embed

    @classmethod
    def clear_case_cache(cls) -> None:
        """Keep one model resident but release all per-case passage vectors."""
        cls._passage_vectors.clear()

    def _model_instance(self) -> Any:
        if LazyDenseEmbedder._model is None:
            import torch
            from sentence_transformers import SentenceTransformer  # lazy: never runs in CPU smoke/tests
            LazyDenseEmbedder._model = SentenceTransformer(
                self.model_id, revision=self.revision,
                device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
            )
            # The protocol bounds retrieval passages below the model's 32k
            # context window.  This cap is an experiment choice, not a claim
            # about the checkpoint's native limit.
            LazyDenseEmbedder._model.max_seq_length = 4096
        return LazyDenseEmbedder._model

    def encode_queries(self, queries: list[str]) -> Any:
        if not queries or any(not isinstance(value, str) or not value.strip() for value in queries):
            raise ValueError("dense_query")
        formatted = ["Instruct: " + self.query_instruction + "\nQuery: " + value for value in queries]
        if self._embed is not None:
            return self._embed(formatted)
        return self._model_instance().encode(formatted, normalize_embeddings=True, batch_size=2, show_progress_bar=False)

    def encode_documents(self, texts: list[str]) -> Any:
        if not texts:
            return []
        if self._embed is not None:
            return self._embed(texts)
        model = self._model_instance()
        document_keys = [hashlib.sha256((self._cache_namespace + "\n" + value).encode("utf-8")).hexdigest() for value in texts]
        # Keep every document aligned with its cache key.  Slicing ``texts``
        # here would silently omit the first passage and shift every later
        # vector to the preceding passage.
        missing = [(key, value) for key, value in zip(document_keys, texts) if key not in self._passage_vectors]
        if missing:
            vectors = model.encode([value for _, value in missing], normalize_embeddings=True, batch_size=2, show_progress_bar=False)
            self._passage_vectors.update(dict(zip((key for key, _ in missing), vectors)))
        documents = [self._passage_vectors[key] for key in document_keys]
        return documents


def _dot(left: Any, right: Any) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


@dataclass
class ClosedWorldTools:
    corpus: list[Passage]
    blocked_url_families: set[str]
    ledger: CaseLedger
    embedder: LazyDenseEmbedder
    sparse_limit: int = 10_000
    dense_limit: int = 20

    def __post_init__(self) -> None:
        self.corpus = deduplicate_passages(
            passage for passage in self.corpus if url_family(passage.url) not in self.blocked_url_families
        )

    def sparse(self, query: str) -> list[Passage]:
        self.ledger.reserve_search("search_sparse", {"query": query, "scope": "full_retained_per_claim_corpus"})
        result = [item for _, _, item in _score_bm25(query, self.corpus)]
        ordered_ids = "\n".join(str(item.passage_id) for item in result)
        observed_ids = "\n".join(str(item.passage_id) for item in result[: self.sparse_limit])
        self.ledger.tool_result({
            "retained_count": len(result),
            "candidate_ids_sha256": hashlib.sha256(ordered_ids.encode("utf-8")).hexdigest(),
            "observed_top_k_count": min(len(result), self.sparse_limit),
            "observed_top_k_ids_sha256": hashlib.sha256(observed_ids.encode("utf-8")).hexdigest(),
        })
        return result  # Always the whole retained corpus.

    def dense(self, query: str, candidates: list[Passage]) -> list[Passage]:
        bounded = candidates[:self.sparse_limit]
        self.ledger.reserve_search("search_dense", {"query": query, "candidate_count": len(candidates), "sparse_limit": self.sparse_limit, "dense_limit": self.dense_limit})
        if not bounded:
            self.ledger.tool_result({"passage_ids": [], "count": 0})
            return []
        query_vector = self.embedder.encode_queries([query])[0]
        document_vectors = self.embedder.encode_documents([item.text for item in bounded])
        result = [item for _, _, item in sorted((-_dot(query_vector, vector), position, passage) for position, (passage, vector) in enumerate(zip(bounded, document_vectors)))][:self.dense_limit]
        self.ledger.tool_result({"passage_ids": [item.passage_id for item in result], "count": len(result)})
        return result

    def hybrid(self, queries: list[str]) -> list[Passage]:
        """Run the fixed BM25 → Qwen dense route with deterministic ties.

        Each query is independently encoded; their L2-normalized vectors are
        averaged and normalized before comparison.  A caller supplies only the
        claims/HyDE query plan, never a prior condition's semantic output.
        """
        if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("query_plan")
        sparse = self.sparse(" ".join(queries))
        bounded = sparse[:self.sparse_limit]
        self.ledger.reserve_search("search_dense", {"query_count": len(queries), "candidate_count": len(sparse), "sparse_limit": self.sparse_limit, "dense_limit": self.dense_limit})
        if not bounded:
            self.ledger.tool_result({"passage_ids": [], "count": 0})
            return []
        query_vectors = self.embedder.encode_queries(queries)
        document_vectors = self.embedder.encode_documents([item.text for item in bounded])
        dimension = len(query_vectors[0])
        if not dimension or any(len(vector) != dimension for vector in [*query_vectors, *document_vectors]):
            raise ValueError("dense_embedding_dimension")
        mean = [sum(float(vector[position]) for vector in query_vectors) / len(query_vectors) for position in range(dimension)]
        norm = math.sqrt(sum(value * value for value in mean))
        if not norm:
            raise ValueError("dense_query_zero_norm")
        mean = [value / norm for value in mean]
        result = [item for _, _, item in sorted((-_dot(mean, vector), position, passage) for position, (passage, vector) in enumerate(zip(bounded, document_vectors)))][:self.dense_limit]
        self.ledger.tool_result({"passage_ids": [item.passage_id for item in result], "count": len(result), "aggregation": "normalized_mean"})
        return result

    @staticmethod
    def evidence(item: Passage, question: str, answer: str) -> dict[str, str]:
        return {"question": question, "answer": answer, "passage_id": str(item.passage_id), "url": item.url, "scraped_text": item.text}
