import pytest

from parsers.document_parser import ParseError, parse_bytes
from parsers.extractor import extract_profile, extract_skills, extract_years

RESUME = """Aarav Sharma
Bengaluru | aarav.sharma@example.com | +91 98450 11223
github.com/aaravsharma

Summary
AI engineer with 3 years of professional experience in NLP.

Technical Skills
Python, PyTorch, Transformers, FastAPI, Docker, cosine similarity, embeddings

Experience
Senior AI Engineer, Quantly Labs, 2023 - Present

Education
M.Tech in Computer Science, IIT Hyderabad, 2021

Projects
- Semantic resume matcher using embeddings and cosine similarity.
"""


def test_parse_txt_bytes():
    doc = parse_bytes("resume.txt", RESUME.encode())
    assert "Aarav Sharma" in doc.text
    assert doc.pages == 1


def test_unsupported_extension_raises():
    with pytest.raises(ParseError):
        parse_bytes("resume.xyz", b"hello world " * 20)


def test_empty_file_raises():
    with pytest.raises(ParseError):
        parse_bytes("resume.txt", b"")


def test_too_short_document_raises():
    with pytest.raises(ParseError):
        parse_bytes("resume.txt", b"hi")


def test_extract_profile_fields():
    profile = extract_profile(RESUME)
    assert profile.name == "Aarav Sharma"
    assert profile.email == "aarav.sharma@example.com"
    assert profile.phone is not None
    assert profile.education_level == 3  # Master's
    assert profile.years_experience >= 3
    assert {"Python", "PyTorch", "Transformers", "FastAPI", "Docker"} <= set(profile.skills)


def test_extract_skills_aliases():
    skills = extract_skills("Experienced with sklearn, hugging face and node.js")
    assert "scikit-learn" in skills
    assert "Transformers" in skills
    assert "Node.js" in skills


def test_extract_years_from_date_ranges():
    assert extract_years("Engineer, Acme, 2019 - 2023") == 4.0
