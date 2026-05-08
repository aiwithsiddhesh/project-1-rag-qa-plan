from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import TOP_K_MAX, TOP_K_MIN, Settings
from src.embedder import get_embedding_model, load_vectorstore
from src.generator import build_prompt, call_llm_with_retry, extract_citations
from src.reranker import CrossEncoderReranker
from src.retriever import HybridRetriever
from src.utils import timer_context


def _extract_chunks(vectorstore: FAISS) -> list[Document]:
    """Extract all Documents from a FAISS vectorstore via its public docstore API."""
    chunks: list[Document] = []
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)
        if isinstance(doc, Document):
            chunks.append(doc)
    return chunks


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ready = False
        logger.info("Initializing RAGPipeline")

        embedding_model = get_embedding_model(settings.embedding_model)
        vectorstore = load_vectorstore(settings.vector_store_path, embedding_model)
        chunks = _extract_chunks(vectorstore)

        self._reranker = CrossEncoderReranker(settings.reranker_model)
        self._retriever = HybridRetriever(vectorstore, chunks, settings)
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[arg-type]  # str not assignable to pydantic.v1.SecretStr
        )

        self._ready = True
        logger.info("RAGPipeline ready")

    def query(
        self, question: str, use_hyde: bool | None = None, top_k: int | None = None
    ) -> dict:
        """Run the full RAG pipeline: validate → retrieve → rerank → generate → cite.

        Returns dict with keys: answer, sources, num_chunks_retrieved,
        retrieval_scores, contexts.
        Raises ValueError if question length is outside [3, 2000].
        use_hyde overrides settings.use_hyde when provided.
        top_k overrides settings.top_k_results when provided.
        """
        if len(question) < 3:
            raise ValueError("Question must be at least 3 characters.")
        if len(question) > 2000:
            raise ValueError("Question must be at most 2000 characters.")
        if top_k is not None and not TOP_K_MIN <= top_k <= TOP_K_MAX:
            raise ValueError(
                f"top_k must be between {TOP_K_MIN} and {TOP_K_MAX}, got {top_k}."
            )

        effective_use_hyde = (
            use_hyde if use_hyde is not None else self._settings.use_hyde
        )
        effective_top_k = top_k if top_k is not None else self._settings.top_k_results

        with timer_context("pipeline.query"):
            retrieval_query = question
            if effective_use_hyde:
                retrieval_query = self._retriever.expand_query_hyde(question, self._llm)

            candidates = self._retriever.retrieve_hybrid(
                retrieval_query,
                k=effective_top_k * self._settings.fetch_k_multiplier,
            )
            reranked = self._reranker.rerank(
                question, candidates, top_n=effective_top_k
            )

            prompt = build_prompt(question, reranked)
            answer = call_llm_with_retry(prompt, self._llm)
            sources = extract_citations(answer, reranked)

        return {
            "answer": answer,
            "sources": sources,
            "num_chunks_retrieved": len(candidates),
            "retrieval_scores": [],
            "contexts": [doc.page_content for doc in reranked],
        }

    def is_ready(self) -> bool:
        return self._ready
