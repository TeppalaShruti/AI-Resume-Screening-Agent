"""LLM explanation layer.

The LLM explains scores; it never produces them. Scores computed in
scoring/scorer.py are passed in and the model is instructed to reuse them
verbatim. If OPENAI_API_KEY is absent (or the call fails), a deterministic
template writes the same shape of explanation so the pipeline stays testable.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert technical recruiter assisting with resume screening. "
    "You will receive pre-computed, deterministic scores. "
    "NEVER invent, recompute or change any number: cite only the numbers given. "
    "Write 2-4 concise sentences (max 90 words) explaining why this candidate ranks "
    "where they do for this specific role, naming concrete matched skills and concrete "
    "gaps, and end with a clear shortlist recommendation."
)


def llm_status() -> dict:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return {
        "enabled": bool(key),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "mode": "openai" if key else "template-fallback",
    }


def _facts(candidate, jd_title: str) -> str:
    p = candidate.profile
    b = candidate.breakdown
    return (
        f"Role: {jd_title}\n"
        f"Candidate: {p.name} | {p.years_experience:g} yrs experience | {p.education_label}\n"
        f"Overall score: {candidate.overall_score}/100 ({candidate.match_level} match)\n"
        f"Breakdown -> skills {b.skills}/30, semantic similarity {b.semantic}/25, "
        f"experience {b.experience}/20, education {b.education}/10, "
        f"required tech {b.required_tech}/10, projects {b.projects}/5\n"
        f"Cosine similarity to JD: {candidate.semantic_similarity}\n"
        f"Matched skills: {', '.join(candidate.matched_skills) or 'none'}\n"
        f"Missing skills: {', '.join(candidate.missing_skills) or 'none'}\n"
        f"Missing required tech: {', '.join(candidate.missing_required_tech) or 'none'}\n"
        f"Experience: {candidate.experience_match}\n"
        f"Education: {candidate.education_match}\n"
        f"System recommendation: {candidate.recommendation}"
    )


def template_reasoning(candidate, jd_title: str) -> str:
    p = candidate.profile
    matched = ", ".join(candidate.matched_skills[:5]) or "no JD-listed skills"
    missing = ", ".join((candidate.missing_required_tech or candidate.missing_skills)[:4])
    parts = [
        f"{p.name} scores {candidate.overall_score}/100 ({candidate.match_level} match) for {jd_title}, "
        f"driven by {candidate.breakdown.skills}/30 on skills and {candidate.breakdown.semantic}/25 on "
        f"semantic similarity ({candidate.semantic_similarity * 100:.0f}% cosine match to the JD).",
        f"Strengths: {matched}; {candidate.experience_match.lower()}; {candidate.education_match.lower()}.",
    ]
    parts.append(
        f"Gaps: {missing}." if missing else "No material skill gaps against the JD were detected."
    )
    parts.append(f"Recommendation: {candidate.recommendation}.")
    return " ".join(parts)


def generate_reasoning(candidate, jd_title: str) -> tuple[str, str]:
    """Return (reasoning_text, source) where source is 'openai' or 'fallback'."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return template_reasoning(candidate, jd_title), "fallback"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _facts(candidate, jd_title)},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        return (text, "openai") if text else (template_reasoning(candidate, jd_title), "fallback")
    except Exception as exc:
        logger.warning("LLM reasoning failed (%s); using template fallback.", exc)
        return template_reasoning(candidate, jd_title), "fallback"


def generate_batch_reasoning(candidates, jd_title: str) -> None:
    for candidate in candidates:
        text, source = generate_reasoning(candidate, jd_title)
        candidate.reasoning = text
        candidate.reasoning_source = source
