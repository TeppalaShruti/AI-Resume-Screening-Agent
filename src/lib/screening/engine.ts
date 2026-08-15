/**
 * Screening orchestrator.
 *
 * Prefers the Python FastAPI backend (Sentence Transformers + OpenAI reasoning)
 * at VITE_API_URL. If it is not reachable, it runs the identical deterministic
 * pipeline in the browser (TF-IDF cosine similarity + template reasoning) so the
 * demo always works end to end.
 */

import { parseFile, ParseError } from "./parse";
import { extractProfile } from "./extract";
import { encodeTfidf, cosineSimilarity } from "./similarity";
import { parseJobDescription, rankCandidates, scoreCandidate } from "./score";
import { templateReasoning } from "./reason";
import { WEIGHTS } from "./types";
import type { ScreeningResult, ScoredCandidate } from "./types";

export const API_URL = (import.meta.env["VITE_API_URL"] as string | undefined) ?? "http://localhost:8000";

export interface Progress {
  current: number;
  total: number;
  label: string;
}

export async function backendAvailable(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500);
    const response = await fetch(`${API_URL}/health`, { signal: controller.signal });
    clearTimeout(timer);
    return response.ok;
  } catch {
    return false;
  }
}

async function screenViaApi(jdText: string, files: File[]): Promise<ScreeningResult> {
  const form = new FormData();
  form.append("jd_text", jdText);
  form.append("use_llm", "true");
  files.forEach((file) => form.append("resumes", file));
  const response = await fetch(`${API_URL}/screen`, { method: "POST", body: form });
  if (!response.ok) throw new Error(`Backend error ${response.status}`);
  const data = (await response.json()) as ScreeningResult;
  return { ...data, engine: "python-api" };
}

export async function screenLocally(
  jdText: string,
  files: File[],
  onProgress?: (p: Progress) => void,
): Promise<ScreeningResult> {
  const started = performance.now();
  const jd = parseJobDescription(jdText.trim());

  const parsed: { filename: string; text: string; warnings: string[] }[] = [];
  const failed: { filename: string; error: string }[] = [];

  for (let i = 0; i < files.length; i += 1) {
    const file = files[i]!;
    onProgress?.({ current: i, total: files.length, label: `Parsing ${file.name}` });
    try {
      parsed.push(await parseFile(file));
    } catch (error) {
      // A single invalid file never aborts the batch.
      failed.push({
        filename: file.name,
        error: error instanceof ParseError ? error.message : `Unexpected error: ${(error as Error).message}`,
      });
    }
  }

  if (!parsed.length) {
    throw new Error(
      `No resume could be parsed. ${failed.map((f) => `${f.filename}: ${f.error}`).join(" | ")}`,
    );
  }

  onProgress?.({ current: files.length, total: files.length, label: "Computing embeddings & scores" });
  const vectors = encodeTfidf([jd.text, ...parsed.map((p) => p.text)]);
  const jdVector = vectors[0]!;

  const scored: ScoredCandidate[] = parsed.map((doc, index) => {
    const profile = extractProfile(doc.text);
    const similarity = Math.max(0, Math.min(1, cosineSimilarity(jdVector, vectors[index + 1]!)));
    return scoreCandidate({
      jd,
      profile,
      resumeText: doc.text,
      candidateId: `${index}-${doc.filename}`,
      filename: doc.filename,
      similarity,
      warnings: doc.warnings,
    });
  });

  const ranked = rankCandidates(scored);
  ranked.forEach((candidate) => {
    candidate.reasoning = templateReasoning(candidate, jd.title);
    candidate.reasoning_source = "fallback";
  });

  return {
    run_id: crypto.randomUUID(),
    job: jd,
    weights: { ...WEIGHTS },
    embedding_backend: "tfidf-browser-fallback",
    llm: { enabled: false, model: "none", mode: "template-fallback" },
    engine_info: {
      runtime: "browser-fallback (Python FastAPI service not reachable)",
      parser: "pdf.js (PDF) · mammoth (DOCX) · plain text (TXT/MD)",
      embeddings: "TF-IDF vectors computed in the browser — NOT Sentence Transformers",
      similarity: "Cosine similarity (TypeScript implementation)",
      scoring: "Same deterministic weighted rubric as the Python engine (/100)",
      llm: "Fallback mode: deterministic template reasoning (no OpenAI call)",
      fallback_mode: true,
    },
    processed: ranked.length,
    failed,
    duration_seconds: Math.round((performance.now() - started) / 10) / 100,
    candidates: ranked,
    summary: {
      shortlisted: ranked.filter((c) => c.shortlist).length,
      strong: ranked.filter((c) => c.match_level === "Strong").length,
      good: ranked.filter((c) => c.match_level === "Good").length,
      moderate: ranked.filter((c) => c.match_level === "Moderate").length,
      weak: ranked.filter((c) => c.match_level === "Weak").length,
      average_score:
        Math.round((ranked.reduce((sum, c) => sum + c.overall_score, 0) / ranked.length) * 100) / 100,
      highest_score: Math.max(...ranked.map((c) => c.overall_score)),
    },
    engine: "browser-fallback",
  };
}

export async function runScreening(
  jdText: string,
  files: File[],
  onProgress?: (p: Progress) => void,
): Promise<ScreeningResult> {
  onProgress?.({ current: 0, total: files.length, label: "Checking screening service" });
  if (await backendAvailable()) {
    try {
      onProgress?.({ current: 0, total: files.length, label: "Screening via Python service" });
      return await screenViaApi(jdText, files);
    } catch {
      // fall through to the in-browser engine
    }
  }
  return screenLocally(jdText, files, onProgress);
}
