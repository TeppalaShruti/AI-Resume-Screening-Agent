from .embeddings import EmbeddingBackend, get_backend
from .similarity import cosine_similarity, semantic_similarity

__all__ = ["EmbeddingBackend", "get_backend", "cosine_similarity", "semantic_similarity"]
