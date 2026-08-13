/** Rule-based resume information extraction (mirror of backend/parsers/extractor.py). */

import { SKILL_ALIASES } from "./skills";
import type { CandidateProfile } from "./types";

const EMAIL_RE = /[\w.+-]+@[\w-]+\.[\w.-]+/;
const PHONE_RE = /(?:\+\d{1,3}[\s.-]?)?\d[\d\s().-]{7,15}\d/g;
const LINK_RE = /(?:https?:\/\/)?(?:www\.)?(?:linkedin\.com|github\.com)\/[\w\-./]+/gi;

const DEGREE_LEVELS: [number, RegExp][] = [
  [4, /\b(ph\.?d|doctorate|doctoral)\b/i],
  [3, /\b(m\.?tech|m\.?s\.?c|m\.?s\b|master'?s?|mba|m\.?e\b|m\.?c\.?a)\b/i],
  [2, /\b(b\.?tech|b\.?e\b|b\.?s\.?c|b\.?s\b|bachelor'?s?|b\.?c\.?a|b\.?com)\b/i],
  [1, /\b(diploma|associate degree|higher secondary|12th)\b/i],
];
export const DEGREE_NAMES: Record<number, string> = {
  0: "None detected",
  1: "Diploma",
  2: "Bachelor's",
  3: "Master's",
  4: "PhD",
};

const YEARS_RE =
  /(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:professional|industry|relevant|work)?\s*experience/gi;
const DATE_RANGE_RE = /(20\d{2}|19\d{2})\s*(?:-|–|—|to)\s*(present|current|now|20\d{2}|19\d{2})/gi;
const TITLE_RE =
  /\b((?:senior |junior |lead |principal |associate )?(?:ai|ml|machine learning|deep learning|data|software|backend|full[- ]stack|frontend|nlp|research|devops|mlops|cloud|python)[ -]?(?:engineer|scientist|developer|analyst|associate|researcher|intern|architect))\b/gi;

const SECTION_HEADS: Record<string, string> = {
  experience: "(?:work experience|professional experience|employment|experience)",
  education: "(?:education|academic)",
  skills: "(?:technical skills|skills|technologies|tech stack)",
  projects: "(?:projects|personal projects|academic projects)",
  certifications: "(?:certifications?|licenses?|courses?)",
};
const ALL_HEADS = Object.values(SECTION_HEADS).join("|");
const STOP_NAME_WORDS = new Set(["resume", "curriculum", "vitae", "cv", "profile", "summary"]);

export function cleanText(text: string): string {
  return text
    .replace(/\x00/g, " ")
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t\u00a0]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function section(text: string, key: string): string {
  const head = SECTION_HEADS[key];
  const block = new RegExp(
    `^[^\\S\\n]*${head}[^\\S\\n]*:?[^\\S\\n]*$([\\s\\S]*?)(?=^[^\\S\\n]*(?:${ALL_HEADS})[^\\S\\n]*:?[^\\S\\n]*$|$(?![\\s\\S]))`,
    "im",
  );
  const match = block.exec(text);
  if (match?.[1]) return match[1].trim();
  const inline = new RegExp(`${head}\\s*[:\\-]\\s*([\\s\\S]+?)(?:\\n\\n|$)`, "i").exec(text);
  return inline?.[1]?.trim() ?? "";
}

export function extractSkills(text: string): string[] {
  const lowered = ` ${text.toLowerCase().replace(/[^a-z0-9+#./ \n-]/g, " ")} `;
  const found = new Set<string>();
  for (const [alias, canonical] of Object.entries(SKILL_ALIASES)) {
    const escaped = alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    if (new RegExp(`(?<![a-z0-9])${escaped}(?![a-z0-9])`).test(lowered)) found.add(canonical);
  }
  return Array.from(found).sort();
}

export function extractYears(text: string, currentYear = 2026): number {
  let best = 0;
  for (const m of text.matchAll(YEARS_RE)) best = Math.max(best, parseFloat(m[1] ?? "0"));

  const spans: [number, number][] = [];
  for (const m of text.matchAll(DATE_RANGE_RE)) {
    const start = parseInt(m[1] ?? "0", 10);
    const endRaw = (m[2] ?? "").toLowerCase();
    const end = ["present", "current", "now"].includes(endRaw) ? currentYear : parseInt(endRaw, 10);
    if (start >= 1980 && end >= start && end <= currentYear) spans.push([start, end]);
  }
  let merged = 0;
  if (spans.length) {
    spans.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let [curS, curE] = spans[0]!;
    for (const [s, e] of spans.slice(1)) {
      if (s <= curE) curE = Math.max(curE, e);
      else {
        merged += curE - curS;
        curS = s;
        curE = e;
      }
    }
    merged += curE - curS;
  }
  return Math.round(Math.max(best, merged) * 10) / 10;
}

export function extractEducation(text: string): [number, string, string] {
  const sec = section(text, "education") || text;
  let level = 0;
  for (const [value, pattern] of DEGREE_LEVELS) {
    if (pattern.test(sec) || pattern.test(text)) level = Math.max(level, value);
  }
  const snippet = sec
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .join(" ")
    .slice(0, 240);
  return [level, DEGREE_NAMES[level] ?? "None detected", snippet];
}

function bullets(sec: string, limit = 6): string[] {
  const items: string[] = [];
  for (const raw of sec.split("\n")) {
    const line = raw.replace(/^[\s•\-*\t]+/, "").trim();
    if (line.length > 12) items.push(line.slice(0, 220));
    if (items.length >= limit) break;
  }
  return items;
}

function findPhone(text: string): string | null {
  for (const m of text.matchAll(PHONE_RE)) {
    const candidate = (m[0] ?? "").trim();
    const digits = candidate.replace(/\D/g, "");
    if (digits.length >= 10 && digits.length <= 13) return candidate;
  }
  return null;
}

function guessName(text: string, email: string | null): string {
  for (const raw of text.split("\n").slice(0, 8)) {
    const line = raw.trim().replace(/^[|\-•*_ ]+|[|\-•*_ ]+$/g, "");
    if (line.length < 3 || line.length > 45) continue;
    if (/\d/.test(line) || line.includes("@") || /http/i.test(line)) continue;
    const words = line.split(/\s+/);
    if (words.length < 2 || words.length > 4) continue;
    if (words.some((w) => STOP_NAME_WORDS.has(w.toLowerCase().replace(/[,.]/g, "")))) continue;
    const caps = words.filter((w) => /^[A-Z]/.test(w)).length;
    if (caps >= Math.max(2, words.length - 1)) return words.map((w) => w.replace(/[,.]/g, "")).join(" ");
  }
  if (email) {
    const local = email.split(/[.@_\-0-9]+/)[0];
    if (local) return local.charAt(0).toUpperCase() + local.slice(1);
  }
  return "Unknown Candidate";
}

export function extractProfile(text: string): CandidateProfile {
  const email = EMAIL_RE.exec(text)?.[0] ?? null;
  const [level, label, eduText] = extractEducation(text);
  const titles = Array.from(
    new Set(
      Array.from(text.matchAll(TITLE_RE), (m) =>
        (m[1] ?? "").trim().replace(/\b\w/g, (c) => c.toUpperCase()),
      ),
    ),
  ).slice(0, 6);

  return {
    name: guessName(text, email),
    email,
    phone: findPhone(text),
    links: Array.from(new Set(Array.from(text.matchAll(LINK_RE), (m) => m[0].replace(/[/.]+$/, "")))).slice(0, 4),
    skills: extractSkills(text),
    years_experience: extractYears(text),
    education_level: level,
    education_label: label,
    education_text: eduText,
    titles,
    projects: bullets(section(text, "projects")),
    certifications: bullets(section(text, "certifications"), 5),
  };
}
