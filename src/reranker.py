from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        logger.info("Loading CrossEncoder: {model}", model=model_name)
        self._model = CrossEncoder(model_name, device="cpu")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        """Score (query, doc) pairs in batch, return top_n by descending score."""
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        try:
            scores = self._model.predict(pairs)
            ranked = sorted(
                zip(documents, scores),
                key=lambda pair: pair[1],
                reverse=True,
            )
            return [doc for doc, _ in ranked[:top_n]]
        except Exception as exc:
            logger.warning(
                "CrossEncoder reranking failed, returning original order: {err}",
                err=exc,
            )
            return documents[:top_n]
