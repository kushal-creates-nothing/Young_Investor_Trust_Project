import logging
from typing import List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# try the good stuff first, fall back to TF-IDF so the code runs without torch
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_SBERT = True
except ImportError:
    _HAVE_SBERT = False
    logger.warning("sentence-transformers not installed — using TF-IDF fallback embeddings")


class FinancialNewsEmbedder(Embeddings):
    """
    Wraps sentence-transformers (or TF-IDF fallback) as a LangChain-compatible
    Embeddings object so it plugs straight into ChromaDB via LangChain Community.

    Default model: all-MiniLM-L6-v2 — 80MB, runs fast on CPU, good for short texts.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        if _HAVE_SBERT:
            self._model = SentenceTransformer(model_name)
            self._backend = "sbert"
        else:
            # lazy-init TF-IDF on first use since we need a corpus to fit on
            self._model = None
            self._backend = "tfidf"
            self._fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._backend == "sbert":
            return self._model.encode(
                texts, show_progress_bar=False, batch_size=32, normalize_embeddings=True
            ).tolist()
        return self._tfidf_embed(texts)

    def embed_query(self, text: str) -> List[float]:
        if self._backend == "sbert":
            return self._model.encode(
                [text], show_progress_bar=False, normalize_embeddings=True
            )[0].tolist()
        vecs = self._tfidf_embed([text])
        return vecs[0] if vecs else []

    # ── TF-IDF fallback ────────────────────────────────────────────────────────

    def _tfidf_embed(self, texts: List[str]) -> List[List[float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        if not texts:
            return []

        if self._model is None:
            self._model = TfidfVectorizer(max_features=384, sublinear_tf=True)

        if not self._fitted:
            self._model.fit(texts)
            self._fitted = True

        mat = self._model.transform(texts).toarray().astype(float)
        # L2-normalise so cosine similarity works the same as with SBERT
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).tolist()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_name(self) -> str:
        return self._model_name
