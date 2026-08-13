/** Deterministic scoring engine (mirror of backend/scoring/scorer.py). */

import { extractSkills } from "./extract";
import { DOMAIN_KEYWORDS, SOFT_SKILLS } from "./skills";
import { WEIGHTS } from "./types";
import type {
  CandidateProfile,
  JobRequirements,
  MatchLevel,
  ScoreBreakdown,
  ScoredCandidate,
} from "./types";

const EDU_WORDS: Record<string, number> = {
  phd: 4, doctorate: 4,
  master: 3, "m.tech": 3, msc: 3, mba: 3,
  bachelor: 2, "b.tech": 2, bsc: 2, degree: 2,
  diploma: 1,
};
const EDU_LABELS: Record<number, string> = {
  0: "Not specified",
  1: "Diploma",
  2: "Bachelor's",
  3: "Master's",
  4: "PhD",
};

const MIN_YEARS_RE = /(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)/gi;
const REQUIRED_BLOCK_RE =
  /(?:required|must[- ]have|mandatory|essential|requirements|qualifications)\s*:?\s*([\s\S]{0,1200})/i;

function jobTitle(text: string): string {
  for (const raw of text.split("\n").slice(0, 6)) {
    const line = raw.replace(/^[\s#*\-•]+|[\s#*\-•]+$/g, "");
    if (line.length >= 4 && line.length <= 80 && !/^(about|we are|company)/i.test(line)) {
      return line.replace(/^(job title|role|position)\s*:\s*/i, "");
    }
  }
  return "Untitled Role";
}

export function parseJobDescription(text: string): JobRequirements {
  const skills = extractSkills(text);
  const block = REQUIRED_BLOCK_RE.exec(text);
  let requiredTech = block?.[1] ? extractSkills(block[1]).filter((s) => !SOFT_SKILLS.has(s)) : [];
  if (!requiredTech.length) requiredTech = skills.filter((s) => !SOFT_SKILLS.has(s)).slice(0, 8);

  const years = Array.from(text.matchAll(MIN_YEARS_RE), (m) => parseFloat(m[1] ?? "0"));
  const minYears = years.length ? Math.min(...years) : 0;

  const lowered = text.toLowerCase();
  let eduLevel = 0;
  for (const [word, level] of Object.entries(EDU_WORDS)) {
    if (lowered.includes(word)) eduLevel = Math.max(eduLevel, level);
  }

  const domains = Object.entries(DOMAIN_KEYWORDS)
    .filter(([, words]) => words.filter((w) => lowered.includes(w)).length >= 2)
    .map(([name]) => name);

  return {
    title: jobTitle(text),
    text,
    skills,
    required_tech: Array.from(new Set(requiredTech)).sort(),
    min_years: minYears,
    education_level: eduLevel,
    education_label: EDU_LABELS[eduLevel] ?? "Not specified",
    domains: domains.length ? domains : ["general"],
  };
}

export function matchLevel(score: number): MatchLevel {
  if (score >= 80) return "Strong";
  if (score >= 65) return "Good";
  if (score >= 45) return "Moderate";
  return "Weak";
}

const round2 = (n: number) => Math.round(n * 100) / 100;

export function scoreCandidate(args: {
  jd: JobRequirements;
  profile: CandidateProfile;
  resumeText: string;
  candidateId: string;
  filename: string;
  similarity: number;
  warnings?: string[];
}): ScoredCandidate {
  const { jd, profile, resumeText, candidateId, filename, warnings = [] } = args;
  const candidateSkills = new Set(profile.skills);

  // --- Skills (30) ---
  let totalWeight = 0;
  let earned = 0;
  const matched: string[] = [];
  const missing: string[] = [];
  for (const skill of jd.skills) {
    const weight = SOFT_SKILLS.has(skill) ? 0.5 : 1;
    totalWeight += weight;
    if (candidateSkills.has(skill)) {
      earned += weight;
      matched.push(skill);
    } else missing.push(skill);
  }
  const skillsPct = totalWeight ? earned / totalWeight : 0;
  const extra = profile.skills.filter((s) => !jd.skills.includes(s)).slice(0, 15);

  // --- Semantic (25) ---
  const sem = Math.max(0, Math.min(1, args.similarity));

  // --- Experience (20) ---
  const years = profile.years_experience;
  const required = jd.min_years;
  let titleBonus = 0;
  const jdWords = new Set(jd.title.toLowerCase().match(/[a-z]+/g) ?? []);
  for (const title of profile.titles) {
    const overlap = (title.toLowerCase().match(/[a-z]+/g) ?? []).filter((w) => jdWords.has(w));
    if (overlap.length >= 2) {
      titleBonus = 0.15;
      break;
    }
  }
  let expBase: number;
  let expLabel: string;
  if (required <= 0) {
    expBase = years ? Math.min(1, years / 3) : 0.25;
    expLabel = `${years} yrs experience (JD does not state a minimum)`;
  } else if (years >= required) {
    expBase = Math.min(1, 0.85 + 0.15 * Math.min(1, (years - required) / Math.max(required, 1)));
    expLabel = `Meets requirement: ${years} yrs vs ${required}+ yrs required`;
  } else {
    expBase = Math.max(0, years / required) * 0.8;
    expLabel = `Below requirement: ${years} yrs vs ${required}+ yrs required`;
  }
  const expPct = Math.min(1, expBase + titleBonus);

  // --- Education (10) ---
  const have = profile.education_level;
  const need = jd.education_level;
  let eduPct: number;
  let eduLabel: string;
  if (need === 0) {
    eduPct = have >= 2 ? 1 : have === 1 ? 0.6 : 0.3;
    eduLabel = `${profile.education_label} (no explicit JD requirement)`;
  } else if (have >= need) {
    eduPct = 1;
    eduLabel = `${profile.education_label} meets required ${EDU_LABELS[need]}`;
  } else if (have === need - 1) {
    eduPct = 0.6;
    eduLabel = `${profile.education_label} is one level below required ${EDU_LABELS[need]}`;
  } else if (have === 0) {
    eduPct = 0;
    eduLabel = `No degree detected; JD requires ${EDU_LABELS[need]}`;
  } else {
    eduPct = 0.3;
    eduLabel = `${profile.education_label} is below required ${EDU_LABELS[need]}`;
  }

  // --- Required technologies (10) ---
  const reqMatched = jd.required_tech.filter((s) => candidateSkills.has(s));
  const reqMissing = jd.required_tech.filter((s) => !candidateSkills.has(s));
  const reqPct = jd.required_tech.length ? reqMatched.length / jd.required_tech.length : 0.5;

  // --- Projects / domain relevance (5) ---
  const corpus = (profile.projects.join(" ") || resumeText).toLowerCase();
  let hits = 0;
  let totalWords = 0;
  for (const domain of jd.domains) {
    const words = DOMAIN_KEYWORDS[domain] ?? [];
    totalWords += words.length;
    hits += words.filter((w) => corpus.includes(w)).length;
  }
  let projPct = totalWords === 0 ? 0.5 : Math.min(1, hits / (totalWords * 0.5));
  if (profile.projects.length) projPct = Math.min(1, projPct + 0.1);

  const breakdown: ScoreBreakdown = {
    skills: round2(skillsPct * WEIGHTS.skills),
    semantic: round2(sem * WEIGHTS.semantic),
    experience: round2(expPct * WEIGHTS.experience),
    education: round2(eduPct * WEIGHTS.education),
    required_tech: round2(reqPct * WEIGHTS.required_tech),
    projects: round2(projPct * WEIGHTS.projects),
  };
  const overall = round2(Object.values(breakdown).reduce((a, b) => a + b, 0));

  const strengths: string[] = [];
  if (matched.length)
    strengths.push(
      `Matches ${matched.length}/${jd.skills.length} JD skills including ${matched.slice(0, 5).join(", ")}`,
    );
  if (sem >= 0.5) strengths.push(`High semantic alignment with the JD (${Math.round(sem * 100)}% cosine similarity)`);
  if (expPct >= 0.85) strengths.push(expLabel);
  if (eduPct >= 1) strengths.push(eduLabel);
  if (profile.projects.length) strengths.push(`${profile.projects.length} relevant project entries detected`);
  if (profile.certifications.length)
    strengths.push(`Holds certifications: ${profile.certifications[0]!.slice(0, 80)}`);

  const gaps: string[] = [];
  if (reqMissing.length) gaps.push(`Missing required technologies: ${reqMissing.join(", ")}`);
  if (missing.length) gaps.push(`Missing ${missing.length} JD skills: ${missing.slice(0, 6).join(", ")}`);
  if (expPct < 0.7) gaps.push(expLabel);
  if (eduPct < 1) gaps.push(eduLabel);
  if (sem < 0.35) gaps.push(`Low semantic overlap with the JD (${Math.round(sem * 100)}%)`);

  const shortlist =
    overall >= 65 && !(jd.required_tech.length > 0 && reqMissing.length === jd.required_tech.length);
  const recommendation =
    overall >= 80
      ? "Shortlist for interview"
      : shortlist
        ? "Shortlist - worth a screening call"
        : overall >= 45
          ? "Hold as backup"
          : "Do not proceed";

  return {
    candidate_id: candidateId,
    filename,
    profile,
    overall_score: overall,
    match_level: matchLevel(overall),
    breakdown,
    component_percentages: {
      skills: round2(skillsPct * 100),
      semantic: round2(sem * 100),
      experience: round2(expPct * 100),
      education: round2(eduPct * 100),
      required_tech: round2(reqPct * 100),
      projects: round2(projPct * 100),
    },
    matched_skills: matched.sort(),
    missing_skills: missing.sort(),
    extra_skills: extra,
    matched_required_tech: reqMatched,
    missing_required_tech: reqMissing,
    experience_match: expLabel,
    education_match: eduLabel,
    semantic_similarity: Math.round(sem * 10000) / 10000,
    strengths: strengths.slice(0, 6),
    gaps: gaps.slice(0, 6),
    shortlist,
    recommendation,
    reasoning: "",
    reasoning_source: "pending",
    rank: 0,
    warnings,
  };
}

export function rankCandidates(candidates: ScoredCandidate[]): ScoredCandidate[] {
  const ordered = [...candidates].sort(
    (a, b) =>
      b.overall_score - a.overall_score ||
      b.semantic_similarity - a.semantic_similarity ||
      a.profile.name.localeCompare(b.profile.name),
  );
  ordered.forEach((c, i) => {
    c.rank = i + 1;
  });
  return ordered;
}
