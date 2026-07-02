"""Semantic embedding utilities (CPU-only, local models)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridEmbedder:
    """Fast TF-IDF pre-filter plus optional sentence-transformer reranking."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", use_transformer: bool = True):
        self.model_name = model_name
        self.use_transformer = use_transformer
        self._model = None
        self._tfidf: TfidfVectorizer | None = None

    def _load_model(self):
        if self._model is None and self.use_transformer:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def tfidf_similarity(self, jd_text: str, documents: list[str]) -> np.ndarray:
        corpus = [jd_text, *documents]
        self._tfidf = TfidfVectorizer(
            max_features=25000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = self._tfidf.fit_transform(corpus)
        return cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    def transformer_similarity(self, jd_text: str, documents: list[str], batch_size: int = 128) -> np.ndarray:
        self._load_model()
        assert self._model is not None
        jd_embedding = self._model.encode([jd_text], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
        doc_embeddings = self._model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return (doc_embeddings @ jd_embedding.T).ravel()

    def hybrid_similarity(
        self,
        jd_text: str,
        documents: list[str],
        tfidf_weight: float = 0.35,
        transformer_weight: float = 0.65,
        batch_size: int = 128,
    ) -> np.ndarray:
        tfidf_scores = self.tfidf_similarity(jd_text, documents)
        if not self.use_transformer:
            return tfidf_scores
        transformer_scores = self.transformer_similarity(jd_text, documents, batch_size=batch_size)
        combined = tfidf_weight * tfidf_scores + transformer_weight * transformer_scores
        return np.clip(combined, 0.0, 1.0)
