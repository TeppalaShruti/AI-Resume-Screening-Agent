"""Cosine similarity helpers used for the semantic relevance component."""

from __future__ import annotations

import math
from typing import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_similarity(jd_text: str, resume_text: str, backend=None) -> float:
    """Return cosine similarity in [0, 1] between JD and resume embeddings."""
    from .embeddings import get_backend

    backend = backend or get_backend()
    jd_vec, resume_vec = backend.encode([jd_text, resume_text])
    raw = cosine_similarity(jd_vec, resume_vec)
    # cosine can be negative; clamp to [0, 1] because negative relevance == no relevance
    return max(0.0, min(1.0, raw))
