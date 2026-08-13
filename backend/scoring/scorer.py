"""Deterministic scoring engine.

Every number in the final report is produced here in Python.
The LLM only receives these numbers and writes prose about them.

Final score out of 100:
    Skills Match ................ 30
    Semantic JD/Resume similarity 25
    Relevant Experience ......... 20
    Education ................... 10
    Required Technologies ....... 10
    Project / Domain relevance ..  5
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from parsers.extractor import CandidateProfile, extract_skills
from nlp.similarity import semantic_similarity
from scoring.skills import DOMAIN_KEYWORDS, SOFT_SKILLS

WEIGHTS = {
    "skills": 30.0,
    "semantic": 25.0,
    "experience": 20.0,
    "education": 10.0,
    "required_tech": 10.0,
    "projects": 5.0,
}

EDU_WORDS = {
    "phd": 4, "doctorate": 4,
    "master": 3, "m.tech": 3, "msc": 3, "mba": 3,
    "bachelor": 2, "b.tech": 2, "bsc": 2, "be/btech": 2, "degree": 2,
    "diploma": 1,
}
EDU_LABELS = {0: "Not specified", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}

MIN_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)", re.I)
REQUIRED_BLOCK_RE = re.compile(
    r"(required|must[- ]have|mandatory|essential|requirements|qualifications)\s*:?\s*(.{0,1200})",
    re.I | re.S,
)


@dataclass
class JobRequirements:
    title: str
    text: str
    skills: list[str] = field(default_factory=list)
    required_tech: list[str] = field(default_factory=list)
    min_years: float = 0.0
    education_level: int = 0
    education_label: str = "Not specified"
    domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    skills: float
    semantic: float
    experience: float
    education: float
    required_tech: float
    projects: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredCandidate:
    candidate_id: str
    filename: str
    profile: CandidateProfile
    overall_score: float
    match_level: str
    breakdown: ScoreBreakdown
    component_percentages: dict
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    matched_required_tech: list[str]
    missing_required_tech: list[str]
    experience_match: str
    education_match: str
    semantic_similarity: float
    strengths: list[str]
    gaps: list[str]
    shortlist: bool
    recommendation: str
    reasoning: str = ""
    reasoning_source: str = "pending"
    rank: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["profile"] = self.profile.to_dict()
        data["breakdown"] = self.breakdown.to_dict()
        return data


def _job_title(text: str) -> str:
    for raw in text.split("\n")[:6]:
        line = raw.strip(" #*-•")
        if 4 <= len(line) <= 80 and not line.lower().startswith(("about", "we are", "company")):
            return re.sub(r"^(job title|role|position)\s*:\s*", "", line, flags=re.I)
    return "Untitled Role"


def parse_job_description(text: str) -> JobRequirements:
    skills = extract_skills(text)

    block = REQUIRED_BLOCK_RE.search(text)
    required_tech = extract_skills(block.group(2)) if block else []
    required_tech = [s for s in required_tech if s not in SOFT_SKILLS] or [
        s for s in skills if s not in SOFT_SKILLS
    ][:8]

    years = [float(m) for m in MIN_YEARS_RE.findall(text)]
    min_years = min(years) if years else 0.0

    lowered = text.lower()
    edu_level = max((v for k, v in EDU_WORDS.items() if k in lowered), default=0)

    domains = [
        name
        for name, words in DOMAIN_KEYWORDS.items()
        if sum(1 for w in words if w in lowered) >= 2
    ]

    return JobRequirements(
        title=_job_title(text),
        text=text,
        skills=skills,
        required_tech=sorted(set(required_tech)),
        min_years=min_years,
        education_level=edu_level,
        education_label=EDU_LABELS[edu_level],
        domains=domains or ["general"],
    )


def _skills_component(jd: JobRequirements, profile: CandidateProfile) -> tuple[float, list, list, list]:
    candidate = set(profile.skills)
    wanted = jd.skills or []
    if not wanted:
        return 0.0, [], [], sorted(candidate)[:20]
    total_weight = 0.0
    earned = 0.0
    matched, missing = [], []
    for skill in wanted:
        weight = 0.5 if skill in SOFT_SKILLS else 1.0
        total_weight += weight
        if skill in candidate:
            earned += weight
            matched.append(skill)
        else:
            missing.append(skill)
    pct = earned / total_weight if total_weight else 0.0
    extra = sorted(candidate - set(wanted))[:15]
    return pct, sorted(matched), sorted(missing), extra


def _experience_component(jd: JobRequirements, profile: CandidateProfile) -> tuple[float, str]:
    years = profile.years_experience
    required = jd.min_years
    title_bonus = 0.0
    jd_title_words = set(re.findall(r"[a-z]+", jd.title.lower()))
    for title in profile.titles:
        overlap = jd_title_words & set(re.findall(r"[a-z]+", title.lower()))
        if len(overlap) >= 2:
            title_bonus = 0.15
            break

    if required <= 0:
        base = min(1.0, years / 3.0) if years else 0.25
        label = f"{years:g} yrs experience (JD does not state a minimum)"
    elif years >= required:
        base = min(1.0, 0.85 + 0.15 * min(1.0, (years - required) / max(required, 1)))
        label = f"Meets requirement: {years:g} yrs vs {required:g}+ yrs required"
    else:
        base = max(0.0, years / required) * 0.8
        label = f"Below requirement: {years:g} yrs vs {required:g}+ yrs required"
    return min(1.0, base + title_bonus), label


def _education_component(jd: JobRequirements, profile: CandidateProfile) -> tuple[float, str]:
    have, need = profile.education_level, jd.education_level
    if need == 0:
        pct = 1.0 if have >= 2 else (0.6 if have == 1 else 0.3)
        return pct, f"{profile.education_label} (no explicit JD requirement)"
    if have >= need:
        return 1.0, f"{profile.education_label} meets required {EDU_LABELS[need]}"
    if have == need - 1:
        return 0.6, f"{profile.education_label} is one level below required {EDU_LABELS[need]}"
    if have == 0:
        return 0.0, f"No degree detected; JD requires {EDU_LABELS[need]}"
    return 0.3, f"{profile.education_label} is below required {EDU_LABELS[need]}"


def _projects_component(jd: JobRequirements, profile: CandidateProfile, resume_text: str) -> float:
    corpus = " ".join(profile.projects).lower() or resume_text.lower()
    hits, total = 0, 0
    for domain in jd.domains:
        words = DOMAIN_KEYWORDS.get(domain, [])
        total += len(words)
        hits += sum(1 for w in words if w in corpus)
    if total == 0:
        return 0.5
    base = min(1.0, hits / (total * 0.5))
    if profile.projects:
        base = min(1.0, base + 0.1)
    return base


def match_level(score: float) -> str:
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Good"
    if score >= 45:
        return "Moderate"
    return "Weak"


def score_candidate(
    jd: JobRequirements,
    profile: CandidateProfile,
    resume_text: str,
    candidate_id: str,
    filename: str,
    similarity: float | None = None,
    warnings: list[str] | None = None,
) -> ScoredCandidate:
    skills_pct, matched, missing, extra = _skills_component(jd, profile)
    sem = semantic_similarity(jd.text, resume_text) if similarity is None else similarity
    sem = max(0.0, min(1.0, sem))
    exp_pct, exp_label = _experience_component(jd, profile)
    edu_pct, edu_label = _education_component(jd, profile)

    candidate_skills = set(profile.skills)
    req_matched = [s for s in jd.required_tech if s in candidate_skills]
    req_missing = [s for s in jd.required_tech if s not in candidate_skills]
    req_pct = len(req_matched) / len(jd.required_tech) if jd.required_tech else 0.5

    proj_pct = _projects_component(jd, profile, resume_text)

    breakdown = ScoreBreakdown(
        skills=round(skills_pct * WEIGHTS["skills"], 2),
        semantic=round(sem * WEIGHTS["semantic"], 2),
        experience=round(exp_pct * WEIGHTS["experience"], 2),
        education=round(edu_pct * WEIGHTS["education"], 2),
        required_tech=round(req_pct * WEIGHTS["required_tech"], 2),
        projects=round(proj_pct * WEIGHTS["projects"], 2),
    )
    overall = round(sum(breakdown.to_dict().values()), 2)
    level = match_level(overall)

    strengths: list[str] = []
    if matched:
        strengths.append(f"Matches {len(matched)}/{len(jd.skills)} JD skills including {', '.join(matched[:5])}")
    if sem >= 0.5:
        strengths.append(f"High semantic alignment with the JD ({sem * 100:.0f}% cosine similarity)")
    if exp_pct >= 0.85:
        strengths.append(exp_label)
    if edu_pct >= 1.0:
        strengths.append(edu_label)
    if profile.projects:
        strengths.append(f"{len(profile.projects)} relevant project entries detected")
    if profile.certifications:
        strengths.append(f"Holds certifications: {profile.certifications[0][:80]}")

    gaps: list[str] = []
    if req_missing:
        gaps.append(f"Missing required technologies: {', '.join(req_missing)}")
    if missing:
        gaps.append(f"Missing {len(missing)} JD skills: {', '.join(missing[:6])}")
    if exp_pct < 0.7:
        gaps.append(exp_label)
    if edu_pct < 1.0:
        gaps.append(edu_label)
    if sem < 0.35:
        gaps.append(f"Low semantic overlap with the JD ({sem * 100:.0f}%)")

    shortlist = overall >= 65 and not (jd.required_tech and len(req_missing) == len(jd.required_tech))
    recommendation = (
        "Shortlist for interview" if overall >= 80
        else "Shortlist - worth a screening call" if shortlist
        else "Hold as backup" if overall >= 45
        else "Do not proceed"
    )

    return ScoredCandidate(
        candidate_id=candidate_id,
        filename=filename,
        profile=profile,
        overall_score=overall,
        match_level=level,
        breakdown=breakdown,
        component_percentages={
            "skills": round(skills_pct * 100, 1),
            "semantic": round(sem * 100, 1),
            "experience": round(exp_pct * 100, 1),
            "education": round(edu_pct * 100, 1),
            "required_tech": round(req_pct * 100, 1),
            "projects": round(proj_pct * 100, 1),
        },
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        matched_required_tech=req_matched,
        missing_required_tech=req_missing,
        experience_match=exp_label,
        education_match=edu_label,
        semantic_similarity=round(sem, 4),
        strengths=strengths[:6],
        gaps=gaps[:6],
        shortlist=shortlist,
        recommendation=recommendation,
        warnings=warnings or [],
    )


def rank_candidates(candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
    ordered = sorted(
        candidates,
        key=lambda c: (-c.overall_score, -c.semantic_similarity, c.profile.name.lower()),
    )
    for index, candidate in enumerate(ordered, start=1):
        candidate.rank = index
    return ordered
