import type { ScreeningResult, ScoredCandidate } from "./types";

const CSV_COLUMNS = [
  "rank", "name", "overall_score", "match_level", "shortlist", "recommendation",
  "email", "phone", "years_experience", "education", "filename",
  "skills_30", "semantic_25", "experience_20", "education_10", "required_tech_10", "projects_5",
  "cosine_similarity", "matched_skills", "missing_skills", "missing_required_tech",
  "strengths", "gaps", "reasoning",
];

function row(c: ScoredCandidate): (string | number)[] {
  return [
    c.rank, c.profile.name, c.overall_score, c.match_level, c.shortlist ? "yes" : "no", c.recommendation,
    c.profile.email ?? "", c.profile.phone ?? "", c.profile.years_experience, c.profile.education_label, c.filename,
    c.breakdown.skills, c.breakdown.semantic, c.breakdown.experience, c.breakdown.education,
    c.breakdown.required_tech, c.breakdown.projects, c.semantic_similarity,
    c.matched_skills.join("; "), c.missing_skills.join("; "), c.missing_required_tech.join("; "),
    c.strengths.join(" | "), c.gaps.join(" | "), c.reasoning,
  ];
}

const escape = (value: string | number) => `"${String(value).replace(/"/g, '""')}"`;

export function toCsv(candidates: ScoredCandidate[]): string {
  return [CSV_COLUMNS.join(","), ...candidates.map((c) => row(c).map(escape).join(","))].join("\n");
}

export function download(filename: string, content: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function downloadCsv(result: ScreeningResult) {
  download(`screening-${result.run_id.slice(0, 8)}.csv`, toCsv(result.candidates), "text/csv");
}

export function downloadJson(result: ScreeningResult) {
  download(
    `screening-${result.run_id.slice(0, 8)}.json`,
    JSON.stringify(result, null, 2),
    "application/json",
  );
}
