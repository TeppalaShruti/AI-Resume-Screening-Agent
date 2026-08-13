export interface CandidateProfile {
  name: string;
  email: string | null;
  phone: string | null;
  links: string[];
  skills: string[];
  years_experience: number;
  education_level: number;
  education_label: string;
  education_text: string;
  titles: string[];
  projects: string[];
  certifications: string[];
}

export interface ScoreBreakdown {
  skills: number;
  semantic: number;
  experience: number;
  education: number;
  required_tech: number;
  projects: number;
}

export interface JobRequirements {
  title: string;
  text: string;
  skills: string[];
  required_tech: string[];
  min_years: number;
  education_level: number;
  education_label: string;
  domains: string[];
}

export type MatchLevel = "Strong" | "Good" | "Moderate" | "Weak";

export interface ScoredCandidate {
  candidate_id: string;
  filename: string;
  profile: CandidateProfile;
  overall_score: number;
  match_level: MatchLevel;
  breakdown: ScoreBreakdown;
  component_percentages: Record<keyof ScoreBreakdown, number>;
  matched_skills: string[];
  missing_skills: string[];
  extra_skills: string[];
  matched_required_tech: string[];
  missing_required_tech: string[];
  experience_match: string;
  education_match: string;
  semantic_similarity: number;
  strengths: string[];
  gaps: string[];
  shortlist: boolean;
  recommendation: string;
  reasoning: string;
  reasoning_source: string;
  rank: number;
  warnings: string[];
}

export interface ScreeningResult {
  run_id: string;
  job: JobRequirements;
  weights: Record<string, number>;
  embedding_backend: string;
  llm: { enabled: boolean; model: string; mode: string };
  processed: number;
  failed: { filename: string; error: string }[];
  duration_seconds: number;
  candidates: ScoredCandidate[];
  summary: {
    shortlisted: number;
    strong: number;
    good: number;
    moderate: number;
    weak: number;
    average_score: number;
  };
  engine: "python-api" | "browser-fallback";
}

export const WEIGHTS = {
  skills: 30,
  semantic: 25,
  experience: 20,
  education: 10,
  required_tech: 10,
  projects: 5,
} as const;

export const COMPONENT_LABELS: Record<keyof ScoreBreakdown, string> = {
  skills: "Skills Match",
  semantic: "Semantic Similarity",
  experience: "Relevant Experience",
  education: "Education",
  required_tech: "Required Technologies",
  projects: "Project / Domain Relevance",
};
