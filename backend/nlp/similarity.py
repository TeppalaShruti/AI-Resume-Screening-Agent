"""Cosine similarity helpers used for the semantic relevance component.

Primary implementation: scikit-learn's ``cosine_similarity`` (vectorised,
industry standard). If scikit-learn is unavailable the identical pure-Python
formula is used so the pipeline never breaks.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence


@lru_cache(maxsize=1)
def similarity_backend() -> str:
    try:
        import sklearn  # noqa: F401

        return "sklearn.metrics.pairwise.cosine_similarity"
    except Exception:  # pragma: no cover - only on minimal installs
        return "pure-python-cosine"


def _cosine_python(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    if similarity_backend().startswith("sklearn"):
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

            return float(sk_cosine(np.asarray([a], dtype=float), np.asarray([b], dtype=float))[0][0])
        except Exception:  # pragma: no cover - defensive
            pass
    return _cosine_python(a, b)


def cosine_similarity_matrix(query: Sequence[float], docs: Sequence[Sequence[float]]) -> list[float]:
    """Cosine similarity of one vector against many (batched when sklearn exists)."""
    if not docs:
        return []
    if similarity_backend().startswith("sklearn"):
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

            sims = sk_cosine(np.asarray([query], dtype=float), np.asarray(docs, dtype=float))[0]
            return [float(s) for s in sims]
        except Exception:  # pragma: no cover - defensive
            pass
    return [_cosine_python(query, doc) for doc in docs]


def semantic_similarity(jd_text: str, resume_text: str, backend=None) -> float:
    """Return cosine similarity in [0, 1] between JD and resume embeddings."""
    from .embeddings import get_backend

    backend = backend or get_backend()
    jd_vec, resume_vec = backend.encode([jd_text, resume_text])
    raw = cosine_similarity(jd_vec, resume_vec)
    # cosine can be negative; clamp to [0, 1] because negative relevance == no relevance
    return max(0.0, min(1.0, raw))
