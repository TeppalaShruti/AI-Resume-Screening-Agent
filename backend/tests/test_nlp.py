from nlp.embeddings import TfidfBackend, get_backend
from nlp.similarity import cosine_similarity, semantic_similarity


def test_cosine_identical_vectors():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == 1.0


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_handles_bad_input():
    assert cosine_similarity([], [1, 2]) == 0.0
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_tfidf_backend_shapes():
    vectors = TfidfBackend().encode(["python nlp engineer", "python nlp researcher"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert cosine_similarity(*vectors) > 0.3


def test_relevant_resume_scores_higher_than_irrelevant():
    jd = "We need an NLP engineer skilled in python, transformers, embeddings and fastapi."
    good = "NLP engineer using python, transformers and fastapi to build embeddings services."
    bad = "Business analyst preparing power bi dashboards and stakeholder reports."
    backend = get_backend()
    assert semantic_similarity(jd, good, backend) > semantic_similarity(jd, bad, backend)
