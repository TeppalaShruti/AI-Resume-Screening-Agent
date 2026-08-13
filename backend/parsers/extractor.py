"""Rule-based information extraction from resume text.

Extracts: name, email, phone, links, skills, years of experience,
education level, job titles, projects and certifications.
Deterministic on purpose - the LLM never produces these facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from scoring.skills import SKILL_ALIASES, canonical_skill

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?\d[\d\s().-]{7,15}\d(?![\w.])")


def find_phone(text: str) -> str | None:
    """Return the first token that looks like a 10-13 digit phone number."""
    for match in PHONE_RE.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 13:
            return candidate
    return None
LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/[\w\-./]+", re.I)

DEGREE_LEVELS = [
    (4, r"\b(ph\.?d|doctorate|doctoral)\b"),
    (3, r"\b(m\.?tech|m\.?s\.?c|m\.?s\b|master'?s?|mba|m\.?e\b|m\.?c\.?a)\b"),
    (2, r"\b(b\.?tech|b\.?e\b|b\.?s\.?c|b\.?s\b|bachelor'?s?|b\.?c\.?a|b\.?com)\b"),
    (1, r"\b(diploma|associate degree|higher secondary|12th)\b"),
]
DEGREE_NAMES = {0: "None detected", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}

YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:professional|industry|relevant|work)?\s*experience",
    re.I,
)
DATE_RANGE_RE = re.compile(
    r"(20\d{2}|19\d{2})\s*[-–—to]{1,3}\s*(present|current|now|20\d{2}|19\d{2})", re.I
)

SECTION_HEADS = {
    "experience": r"(work experience|professional experience|employment|experience)",
    "education": r"(education|academic)",
    "skills": r"(technical skills|skills|technologies|tech stack)",
    "projects": r"(projects|personal projects|academic projects)",
    "certifications": r"(certifications?|licenses?|courses?)",
}

TITLE_RE = re.compile(
    r"\b((?:senior |junior |lead |principal |associate )?"
    r"(?:ai|ml|machine learning|deep learning|data|software|backend|full[- ]stack|frontend|nlp|research|devops|mlops|cloud|python)"
    r"[ -]?(?:engineer|scientist|developer|analyst|associate|researcher|intern|architect))\b",
    re.I,
)

STOP_NAME_WORDS = {"resume", "curriculum", "vitae", "cv", "profile", "summary"}


@dataclass
class CandidateProfile:
    name: str
    email: str | None
    phone: str | None
    links: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    years_experience: float = 0.0
    education_level: int = 0
    education_label: str = "None detected"
    education_text: str = ""
    titles: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _guess_name(text: str, email: str | None) -> str:
    for raw in text.split("\n")[:8]:
        line = raw.strip().strip("|-•*_ ")
        if not (3 <= len(line) <= 45):
            continue
        if any(ch.isdigit() for ch in line) or "@" in line or "http" in line.lower():
            continue
        words = line.split()
        if not (1 < len(words) <= 4):
            continue
        if any(w.lower().strip(",.") in STOP_NAME_WORDS for w in words):
            continue
        if sum(1 for w in words if w[:1].isupper()) >= max(2, len(words) - 1):
            return " ".join(w.strip(",.") for w in words)
    if email:
        local = re.split(r"[.@_\-0-9]+", email)[0]
        if local:
            return local.capitalize()
    return "Unknown Candidate"


def _section(text: str, key: str) -> str:
    pattern = SECTION_HEADS[key]
    heads = "|".join(SECTION_HEADS[k] for k in SECTION_HEADS)
    match = re.search(
        rf"^[^\S\n]*{pattern}[^\S\n]*:?[^\S\n]*$(?P<body>.*?)(?=^[^\S\n]*(?:{heads})[^\S\n]*:?[^\S\n]*$|\Z)",
        text,
        re.I | re.S | re.M,
    )
    if match:
        return match.group("body").strip()
    inline = re.search(rf"{pattern}\s*[:\-]\s*(?P<body>.+?)(?:\n\n|\Z)", text, re.I | re.S)
    return inline.group("body").strip() if inline else ""


def extract_skills(text: str) -> list[str]:
    lowered = " " + re.sub(r"[^a-z0-9+#./ \n-]", " ", text.lower()) + " "
    found: set[str] = set()
    for alias, canonical in SKILL_ALIASES.items():
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.add(canonical)
    return sorted(found)


def extract_years(text: str) -> float:
    explicit = [float(m) for m in YEARS_RE.findall(text)]
    best = max(explicit) if explicit else 0.0

    spans: list[tuple[int, int]] = []
    current_year = 2026
    for start, end in DATE_RANGE_RE.findall(text):
        s = int(start)
        e = current_year if end.lower() in {"present", "current", "now"} else int(end)
        if 1980 <= s <= e <= current_year:
            spans.append((s, e))
    merged_years = 0.0
    if spans:
        spans.sort()
        cur_s, cur_e = spans[0]
        for s, e in spans[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged_years += cur_e - cur_s
                cur_s, cur_e = s, e
        merged_years += cur_e - cur_s
    return round(max(best, float(merged_years)), 1)


def extract_education(text: str) -> tuple[int, str, str]:
    section = _section(text, "education") or text
    level = 0
    for value, pattern in DEGREE_LEVELS:
        if re.search(pattern, section, re.I) or re.search(pattern, text, re.I):
            level = max(level, value)
    snippet = " ".join(line.strip() for line in section.split("\n") if line.strip())[:240]
    return level, DEGREE_NAMES[level], snippet


def _bullets(section: str, limit: int = 6) -> list[str]:
    items: list[str] = []
    for raw in section.split("\n"):
        line = raw.strip(" •-*\t")
        if len(line) > 12:
            items.append(line[:220])
        if len(items) >= limit:
            break
    return items


def extract_profile(text: str) -> CandidateProfile:
    email_match = EMAIL_RE.search(text)
    phone = find_phone(text)
    email = email_match.group(0) if email_match else None
    level, label, edu_text = extract_education(text)
    titles = sorted({t.strip().title() for t in TITLE_RE.findall(text)})[:6]

    return CandidateProfile(
        name=_guess_name(text, email),
        email=email,
        phone=phone,
        links=sorted({link.rstrip("/.") for link in LINK_RE.findall(text)})[:4],
        skills=extract_skills(text),
        years_experience=extract_years(text),
        education_level=level,
        education_label=label,
        education_text=edu_text,
        titles=titles,
        projects=_bullets(_section(text, "projects")),
        certifications=_bullets(_section(text, "certifications"), limit=5),
    )


def canonical_skills(values: list[str]) -> list[str]:
    return sorted({c for c in (canonical_skill(v) for v in values) if c})
