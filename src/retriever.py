from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, cast

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.exceptions import RetrievalError

if TYPE_CHECKING:
    from src.config import Settings


def _tokenize(text: str) -> list[str]:
    """Strip punctuation, lowercase, and split — called identically at index and query time."""
    return re.sub(r"[^\w\s]", "", text).lower().split()


class HybridRetriever:
    def __init__(
        self,
        vectorstore: FAISS,
        chunks: list[Document],
        settings: Settings,
    ) -> None:
        if not chunks:
            raise RetrievalError("Cannot initialize HybridRetriever with empty chunks")
        self._vectorstore = vectorstore
        self._chunks = chunks
        self._settings = settings
        tokenized = [_tokenize(chunk.page_content) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built from {n} chunks", n=len(chunks))

    def retrieve_dense(
        self,
        query: str,
        k: int,
        fetch_k: int,
    ) -> list[tuple[Document, float]]:
        """MMR retrieval over the FAISS vectorstore."""
        try:
            # FAISS types embedding_function as Union[Callable, Embeddings]; at runtime
            # from_documents always stores embed_query (a Callable).
            embed_fn = cast(
                Callable[[str], list[float]], self._vectorstore.embedding_function
            )
            query_vector = embed_fn(query)
            return self._vectorstore.max_marginal_relevance_search_with_score_by_vector(
                query_vector,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=self._settings.mmr_lambda,
            )
        except Exception as exc:
            raise RetrievalError("Dense retrieval failed", original_error=exc) from exc

    def retrieve_bm25(self, query: str, k: int) -> list[tuple[Document, float]]:
        """BM25 retrieval over all indexed chunks."""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in ranked[:k]]

    def retrieve_hybrid(self, query: str, k: int) -> list[Document]:
        """BM25 + dense MMR fused via Reciprocal Rank Fusion (RRF k=60)."""
        fetch_k = k * self._settings.fetch_k_multiplier
        dense_results = self.retrieve_dense(query, k=k, fetch_k=fetch_k)
        bm25_results = self.retrieve_bm25(query, k=k)

        # RRF: score = Σ 1/(rank + 60) per doc; key on page_content for deterministic dedup
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, (doc, _) in enumerate(dense_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 60)
            doc_map[key] = doc

        for rank, (doc, _) in enumerate(bm25_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 60)
            doc_map[key] = doc

        sorted_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
        return [doc_map[key] for key in sorted_keys[:k]]

    def expand_query_hyde(self, query: str, llm: Any) -> str:
        """Generate a hypothetical answer to close the query-document distribution gap (HyDE)."""
        prompt = (
            "Write a concise 2-3 sentence passage that would directly answer "
            f"the following question: {query}"
        )
        try:
            response = llm.invoke(prompt)
            hypothetical = (
                response.content if hasattr(response, "content") else str(response)
            )
            logger.debug("HyDE expanded query to {n} chars", n=len(hypothetical))
            return hypothetical
        except Exception as exc:
            logger.warning(
                "HyDE expansion failed, using original query: {err}", err=exc
            )
            return query
