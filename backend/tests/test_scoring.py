from llm.reasoning import generate_reasoning, llm_status
from parsers.extractor import extract_profile
from scoring.scorer import WEIGHTS, match_level, parse_job_description, rank_candidates, score_candidate

JD = """Junior AI Research Associate

Required qualifications
- Bachelor's degree in Computer Science
- 1+ years of experience with Python for machine learning
- Strong knowledge of NLP, Transformers, embeddings and cosine similarity
- Experience with PyTorch and scikit-learn
- Experience building REST APIs with FastAPI
- Comfortable with Git, Docker and Linux
"""

STRONG = """Fatima Khan
fatima.khan@example.com

Summary
AI research associate with 3 years of experience.

Skills
Python, PyTorch, Transformers, embeddings, cosine similarity, NLP, scikit-learn,
FastAPI, REST APIs, Docker, Git, Linux

Education
M.Tech in Artificial Intelligence, IISc, 2024

Projects
- Semantic search engine with embeddings and a FastAPI model service.
"""

WEAK = """Neha Gupta
neha.gupta@example.com

Summary
Business analyst with 4 years of experience in reporting.

Skills
Excel, Power BI, Tableau, documentation, stakeholder management, Agile

Education
MBA in Business Analytics, 2021
"""


def _score(text, name):
    jd = parse_job_description(JD)
    profile = extract_profile(text)
    return jd, score_candidate(jd, profile, text, name, f"{name}.txt")


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_job_description_parsing():
    jd = parse_job_description(JD)
    assert jd.title.startswith("Junior AI Research Associate")
    assert jd.min_years == 1.0
    assert jd.education_level == 2
    assert {"Python", "PyTorch", "NLP", "FastAPI"} <= set(jd.skills)


def test_component_caps_respected():
    _, candidate = _score(STRONG, "strong")
    b = candidate.breakdown
    assert b.skills <= 30 and b.semantic <= 25 and b.experience <= 20
    assert b.education <= 10 and b.required_tech <= 10 and b.projects <= 5
    assert 0 <= candidate.overall_score <= 100
    assert round(sum(b.to_dict().values()), 2) == candidate.overall_score


def test_strong_candidate_outranks_weak():
    _, strong = _score(STRONG, "strong")
    _, weak = _score(WEAK, "weak")
    assert strong.overall_score > weak.overall_score
    ranked = rank_candidates([weak, strong])
    assert ranked[0].candidate_id == "strong"
    assert ranked[0].rank == 1 and ranked[1].rank == 2


def test_matched_and_missing_skills_are_disjoint():
    jd, candidate = _score(STRONG, "strong")
    assert set(candidate.matched_skills) & set(candidate.missing_skills) == set()
    assert set(candidate.matched_skills) | set(candidate.missing_skills) == set(jd.skills)


def test_match_level_bands():
    assert match_level(92) == "Strong"
    assert match_level(70) == "Good"
    assert match_level(50) == "Moderate"
    assert match_level(20) == "Weak"


def test_scoring_is_deterministic():
    _, first = _score(STRONG, "strong")
    _, second = _score(STRONG, "strong")
    assert first.overall_score == second.overall_score


def test_fallback_reasoning_without_api_key():
    assert llm_status()["enabled"] is False
    _, candidate = _score(STRONG, "strong")
    text, source = generate_reasoning(candidate, "Junior AI Research Associate")
    assert source == "fallback"
    assert str(candidate.overall_score) in text
    assert candidate.recommendation in text
