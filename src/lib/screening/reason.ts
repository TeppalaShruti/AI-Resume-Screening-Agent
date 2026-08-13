/**
 * Browser fallback reasoning (mirror of the template branch in
 * backend/llm/reasoning.py). When the Python API is reachable and an
 * OPENAI_API_KEY is configured, real LLM reasoning comes from the backend.
 * The explanation never introduces new numbers - it only narrates the
 * deterministic scores.
 */

import type { ScoredCandidate } from "./types";

export function templateReasoning(candidate: ScoredCandidate, jdTitle: string): string {
  const p = candidate.profile;
  const matched = candidate.matched_skills.slice(0, 5).join(", ") || "no JD-listed skills";
  const missing = (
    candidate.missing_required_tech.length ? candidate.missing_required_tech : candidate.missing_skills
  )
    .slice(0, 4)
    .join(", ");

  return [
    `${p.name} scores ${candidate.overall_score}/100 (${candidate.match_level} match) for ${jdTitle}, driven by ${candidate.breakdown.skills}/30 on skills and ${candidate.breakdown.semantic}/25 on semantic similarity (${Math.round(candidate.semantic_similarity * 100)}% cosine match to the JD).`,
    `Strengths: ${matched}; ${candidate.experience_match.toLowerCase()}; ${candidate.education_match.toLowerCase()}.`,
    missing ? `Gaps: ${missing}.` : "No material skill gaps against the JD were detected.",
    `Recommendation: ${candidate.recommendation}.`,
  ].join(" ");
}
