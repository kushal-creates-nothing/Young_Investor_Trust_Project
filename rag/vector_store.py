import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ArticleVectorStore:
    """
    ChromaDB-backed persistent vector store for financial news articles.
    Articles are embedded and stored so we can do semantic similarity search
    at query time — this powers the RAG pipeline.

    Uses LangChain's Chroma wrapper so the store is a drop-in LangChain retriever.
    """

    COLLECTION = "financial_articles"

    def __init__(self, persist_dir: str = "data/chroma", embedder=None):
        from langchain_community.vectorstores import Chroma
        from rag.embedder import FinancialNewsEmbedder

        os.makedirs(persist_dir, exist_ok=True)
        self._embedder = embedder or FinancialNewsEmbedder()
        self._store = Chroma(
            collection_name=self.COLLECTION,
            embedding_function=self._embedder,
            persist_directory=persist_dir,
        )
        logger.info(
            "Vector store ready at '%s' (backend: %s)",
            persist_dir,
            getattr(self._embedder, "backend", "unknown"),
        )

    # ── write ──────────────────────────────────────────────────────────────────

    def add_articles(self, articles, sentiment_results=None) -> int:
        """
        Embed and store articles. Returns count of newly added documents.
        sentiment_results list (parallel to articles) adds metadata like
        vader_compound and keywords to each document.
        """
        if not articles:
            return 0

        texts = [a.raw_text or f"{a.title} {a.description}" for a in articles]
        # stable deterministic ID based on URL so we don't duplicate on re-runs
        ids = [f"art_{hash(a.url) % (10**12)}" for a in articles]

        metadatas = []
        for i, a in enumerate(articles):
            meta = {
                "title": a.title[:200],
                "source": a.source,
                "url": a.url[:500],
                "published_at": a.published_at or "",
            }
            if sentiment_results and i < len(sentiment_results):
                r = sentiment_results[i]
                meta["vader_compound"] = round(r.vader_compound, 4)
                meta["vader_label"] = r.vader_label
                meta["is_safe_haven"] = str(r.is_safe_haven_signal)
                meta["is_risk_on"] = str(r.is_risk_on_signal)
            metadatas.append(meta)

        try:
            self._store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info("Added/updated %d articles in vector store", len(texts))
            return len(texts)
        except Exception as e:
            logger.error("Vector store write failed: %s", e)
            return 0

    # ── read ───────────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple]:
        """Return top-k (document, score) tuples for a query string."""
        try:
            return self._store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            logger.error("Similarity search failed: %s", e)
            return []

    def as_retriever(self, k: int = 5):
        """Return a LangChain BaseRetriever for use in chains."""
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def count(self) -> int:
        try:
            return self._store._collection.count()
        except Exception:
            return 0
