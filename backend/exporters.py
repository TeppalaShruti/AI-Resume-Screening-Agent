"""CSV / JSON export helpers for a completed screening run."""

from __future__ import annotations

import csv
import io
import json

CSV_COLUMNS = [
    "rank", "name", "overall_score", "match_level", "shortlist", "recommendation",
    "email", "phone", "years_experience", "education", "filename",
    "skills_30", "semantic_25", "experience_20", "education_10", "required_tech_10", "projects_5",
    "cosine_similarity", "matched_skills", "missing_skills", "missing_required_tech",
    "strengths", "gaps", "reasoning",
]


def results_to_rows(candidates) -> list[dict]:
    rows = []
    for c in candidates:
        rows.append({
            "rank": c.rank,
            "name": c.profile.name,
            "overall_score": c.overall_score,
            "match_level": c.match_level,
            "shortlist": "yes" if c.shortlist else "no",
            "recommendation": c.recommendation,
            "email": c.profile.email or "",
            "phone": c.profile.phone or "",
            "years_experience": c.profile.years_experience,
            "education": c.profile.education_label,
            "filename": c.filename,
            "skills_30": c.breakdown.skills,
            "semantic_25": c.breakdown.semantic,
            "experience_20": c.breakdown.experience,
            "education_10": c.breakdown.education,
            "required_tech_10": c.breakdown.required_tech,
            "projects_5": c.breakdown.projects,
            "cosine_similarity": c.semantic_similarity,
            "matched_skills": "; ".join(c.matched_skills),
            "missing_skills": "; ".join(c.missing_skills),
            "missing_required_tech": "; ".join(c.missing_required_tech),
            "strengths": " | ".join(c.strengths),
            "gaps": " | ".join(c.gaps),
            "reasoning": c.reasoning,
        })
    return rows


def to_csv(candidates) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(results_to_rows(candidates))
    return buffer.getvalue()


def to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str)
