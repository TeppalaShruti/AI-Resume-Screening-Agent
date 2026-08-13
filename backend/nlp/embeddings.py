"""Embedding backends.

Primary: Sentence Transformers (all-MiniLM-L6-v2) -> dense semantic vectors.
Fallback: deterministic TF-IDF vectorizer, used when the model cannot be
downloaded (offline CI, air-gapped demo). Both expose ``encode(texts)``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
STOPWORDS = {
    "the", "and", "for", "with", "you", "our", "are", "will", "have", "has", "this", "that",
    "from", "your", "who", "was", "were", "not", "but", "can", "all", "any", "out", "use",
    "using", "into", "about", "their", "them", "they", "his", "her", "she", "him", "its",
    "job", "role", "work", "team", "years", "year", "new", "per", "via", "etc",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in STOPWORDS]


class EmbeddingBackend(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class TfidfBackend:
    """Deterministic TF-IDF backend over the batch being compared."""

    name = "tfidf-fallback"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        docs = [tokenize(t) for t in texts]
        vocab = sorted({tok for doc in docs for tok in doc})
        index = {tok: i for i, tok in enumerate(vocab)}
        n_docs = max(1, len(docs))
        df = Counter(tok for doc in docs for tok in set(doc))
        vectors: list[list[float]] = []
        for doc in docs:
            counts = Counter(doc)
            total = max(1, len(doc))
            vec = [0.0] * len(vocab)
            for tok, count in counts.items():
                idf = math.log((1 + n_docs) / (1 + df[tok])) + 1.0
                vec[index[tok]] = (count / total) * idf
            vectors.append(vec)
        return vectors


class SentenceTransformerBackend:
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


@lru_cache(maxsize=1)
def get_backend() -> EmbeddingBackend:
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if os.getenv("DISABLE_TRANSFORMERS") == "1":
        return TfidfBackend()
    try:
        return SentenceTransformerBackend(model_name)
    except Exception as exc:  # pragma: no cover - offline environments
        logger.warning("Sentence Transformers unavailable (%s); using TF-IDF fallback.", exc)
        return TfidfBackend()
