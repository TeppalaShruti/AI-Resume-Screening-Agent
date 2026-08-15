import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  ArrowUpDown,
  Award,
  BrainCircuit,
  Download,
  FileText,
  Loader2,
  RotateCcw,
  Search,
  Sparkles,
  Upload,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";

import { runScreening, type Progress as RunProgress } from "@/lib/screening/engine";
import { downloadCsv, downloadJson } from "@/lib/screening/exporters";
import { validateFile } from "@/lib/screening/parse";
import { COMPONENT_LABELS, type ScoredCandidate, type ScreeningResult } from "@/lib/screening/types";

const sampleResumes = import.meta.glob("../../backend/samples/resumes/*.txt", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;
const sampleJd = Object.values(
  import.meta.glob("../../backend/samples/job_description.txt", {
    query: "?raw",
    import: "default",
    eager: true,
  }) as Record<string, string>,
)[0] as string;

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AI Resume Screening Agent | Rank candidates against any JD" },
      {
        name: "description",
        content:
          "Upload a job description and 10+ resumes to get NLP-based relevance scores, ranked candidates, gaps and explainable AI reasoning with CSV/JSON export.",
      },
      { property: "og:title", content: "AI Resume Screening Agent" },
      {
        property: "og:description",
        content:
          "Deterministic 100-point scoring, semantic embeddings and AI reasoning for fast, explainable resume screening.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ScreeningDashboard,
});

const LEVEL_CLASS: Record<string, string> = {
  Strong: "bg-strong text-strong-foreground",
  Good: "bg-good text-good-foreground",
  Moderate: "bg-moderate text-moderate-foreground",
  Weak: "bg-weak text-weak-foreground",
};

function ScoreBar({ label, value, max }: { label: string; value: number; max: number }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">
          {value.toFixed(1)} / {max}
        </span>
      </div>
      <Progress value={(value / max) * 100} className="h-2" />
    </div>
  );
}

function CandidateDetail({ candidate, weights }: { candidate: ScoredCandidate; weights: Record<string, number> }) {
  const p = candidate.profile;
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Badge className={LEVEL_CLASS[candidate.match_level]}>{candidate.match_level} match</Badge>
        <span className="font-display text-3xl font-bold tabular-nums">{candidate.overall_score}</span>
        <span className="text-muted-foreground">/ 100</span>
        <Badge variant={candidate.shortlist ? "default" : "outline"}>{candidate.recommendation}</Badge>
      </div>

      <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
        <span>{p.email ?? "No email detected"}</span>
        <span>{p.phone ?? "No phone detected"}</span>
        <span>{p.years_experience} years experience</span>
        <span>{p.education_label}</span>
        {p.links.map((link) => (
          <span key={link}>{link}</span>
        ))}
      </div>

      <section className="rounded-lg border bg-surface-muted p-4">
        <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold">
          <BrainCircuit className="size-4" /> AI reasoning
          <Badge variant="outline" className="ml-auto text-xs">
            {candidate.reasoning_source === "openai" ? "OpenAI" : "Deterministic fallback"}
          </Badge>
        </h3>
        <p className="text-sm leading-relaxed">{candidate.reasoning}</p>
      </section>

      <section className="space-y-3">
        <h3 className="text-sm font-semibold">Score breakdown</h3>
        {(Object.keys(COMPONENT_LABELS) as (keyof typeof COMPONENT_LABELS)[]).map((key) => (
          <ScoreBar
            key={key}
            label={COMPONENT_LABELS[key]}
            value={candidate.breakdown[key]}
            max={weights[key] ?? 0}
          />
        ))}
        <p className="text-xs text-muted-foreground">
          Cosine similarity to JD: {(candidate.semantic_similarity * 100).toFixed(1)}%
        </p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2">
        <section>
          <h3 className="mb-2 text-sm font-semibold">Matched skills ({candidate.matched_skills.length})</h3>
          <div className="flex flex-wrap gap-1.5">
            {candidate.matched_skills.map((s) => (
              <Badge key={s} className="bg-strong text-strong-foreground">
                {s}
              </Badge>
            ))}
            {!candidate.matched_skills.length && <span className="text-sm text-muted-foreground">None</span>}
          </div>
        </section>
        <section>
          <h3 className="mb-2 text-sm font-semibold">Missing skills ({candidate.missing_skills.length})</h3>
          <div className="flex flex-wrap gap-1.5">
            {candidate.missing_skills.map((s) => (
              <Badge key={s} variant="outline">
                {s}
              </Badge>
            ))}
            {!candidate.missing_skills.length && <span className="text-sm text-muted-foreground">None</span>}
          </div>
        </section>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <section className="rounded-lg border p-4">
          <h3 className="mb-1 text-sm font-semibold">Experience &amp; education</h3>
          <p className="text-sm text-muted-foreground">{candidate.experience_match}</p>
          <p className="text-sm text-muted-foreground">{candidate.education_match}</p>
        </section>
        <section className="rounded-lg border p-4">
          <h3 className="mb-1 text-sm font-semibold">Required technologies</h3>
          <p className="text-sm text-muted-foreground">
            Matched: {candidate.matched_required_tech.join(", ") || "none"}
          </p>
          <p className="text-sm text-muted-foreground">
            Missing: {candidate.missing_required_tech.join(", ") || "none"}
          </p>
        </section>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <section>
          <h3 className="mb-2 text-sm font-semibold">Strengths</h3>
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            {candidate.strengths.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </section>
        <section>
          <h3 className="mb-2 text-sm font-semibold">Gaps</h3>
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            {candidate.gaps.map((g) => (
              <li key={g}>{g}</li>
            ))}
            {!candidate.gaps.length && <li>No material gaps detected</li>}
          </ul>
        </section>
      </div>

      {!!candidate.profile.projects.length && (
        <section>
          <h3 className="mb-2 text-sm font-semibold">Projects detected</h3>
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            {candidate.profile.projects.map((project) => (
              <li key={project}>{project}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function EngineTransparency({ result }: { result: ScreeningResult }) {
  const info = result.engine_info;
  const rows: [string, string][] = info
    ? [
        ["Runtime", info.runtime],
        ["Document parser", info.parser],
        ["Embeddings / NLP model", info.embeddings],
        ["Similarity method", info.similarity],
        ["Scoring", info.scoring],
        ["LLM reasoning", info.llm],
      ]
    : [
        ["Runtime", result.engine],
        ["Embeddings / NLP model", result.embedding_backend],
        ["Similarity method", "Cosine similarity"],
        ["Scoring", "Deterministic weighted rubric (/100)"],
        ["LLM reasoning", result.llm.mode],
      ];
  const fallback = info?.fallback_mode ?? result.engine === "browser-fallback";

  return (
    <Card className="shadow-card">
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <CardTitle className="flex items-center gap-2 text-base">
          <BrainCircuit className="size-4" /> Engine transparency
        </CardTitle>
        <Badge variant={fallback ? "outline" : "default"}>
          {fallback ? "Fallback mode" : "Full stack: Sentence Transformers + OpenAI"}
        </Badge>
      </CardHeader>
      <CardContent className="grid gap-2 text-sm sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="rounded-md border bg-surface-muted px-3 py-2">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
            <p className="mt-0.5">{value}</p>
          </div>
        ))}
        <div className="rounded-md border bg-surface-muted px-3 py-2">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">Run</p>
          <p className="mt-0.5">
            Role: {result.job.title} · {result.processed} resumes in {result.duration_seconds}s
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function ScoringMethodology({ weights }: { weights: Record<string, number> }) {
  return (
    <Card className="shadow-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <ArrowUpDown className="size-4" /> Scoring methodology
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm text-muted-foreground">
        <p>
          Every score is computed deterministically in code — the language model only writes the
          explanation and never produces or alters a number.
        </p>
        <div className="grid gap-2 sm:grid-cols-3">
          {(Object.keys(COMPONENT_LABELS) as (keyof typeof COMPONENT_LABELS)[]).map((key) => (
            <div key={key} className="rounded-md border px-3 py-2">
              <p className="font-medium text-foreground">{COMPONENT_LABELS[key]}</p>
              <p className="text-xs">{weights[key] ?? 0}% of the final score</p>
            </div>
          ))}
        </div>
        <ul className="list-disc space-y-1 pl-5 text-xs">
          <li>Skills: overlap between JD skills and resume skills from a 120+ term taxonomy with aliases.</li>
          <li>Semantic: cosine similarity between JD and resume embeddings, clamped to [0, 1].</li>
          <li>Experience: detected years vs. the JD minimum, with partial credit below it.</li>
          <li>Education: highest detected qualification vs. the JD requirement.</li>
          <li>Technologies: explicitly required tools present in the resume.</li>
          <li>Project / domain relevance: domain keywords found in project and summary sections.</li>
          <li>Match levels: Strong ≥ 75, Good ≥ 60, Moderate ≥ 45, otherwise Weak; shortlist at ≥ 60.</li>
        </ul>
      </CardContent>
    </Card>
  );
}

function ScreeningDashboard() {
  const [jdText, setJdText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const [result, setResult] = useState<ScreeningResult | null>(null);
  const [selected, setSelected] = useState<ScoredCandidate | null>(null);
  const [query, setQuery] = useState("");
  const [levelFilter, setLevelFilter] = useState("all");
  const [sortKey, setSortKey] = useState<"rank" | "score" | "name" | "experience">("rank");

  function addFiles(list: FileList | null) {
    if (!list) return;
    const accepted: File[] = [];
    for (const file of Array.from(list)) {
      const error = validateFile(file);
      if (error) toast.error(`${file.name}: ${error}`);
      else accepted.push(file);
    }
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...accepted.filter((f) => !names.has(f.name))];
    });
  }

  function loadSample() {
    setJdText(sampleJd);
    const sampleFiles = Object.entries(sampleResumes).map(
      ([path, content]) =>
        new File([content], path.split("/").pop() ?? "resume.txt", { type: "text/plain" }),
    );
    setFiles(sampleFiles);
    toast.success(`Loaded sample JD and ${sampleFiles.length} resumes`);
  }

  async function handleRun() {
    if (jdText.trim().length < 50) {
      toast.error("Provide a job description of at least 50 characters.");
      return;
    }
    if (!files.length) {
      toast.error("Upload at least one resume.");
      return;
    }
    setRunning(true);
    setResult(null);
    try {
      const run = await runScreening(jdText, files, setProgress);
      setResult(run);
      run.failed.forEach((f) => toast.warning(`Skipped ${f.filename}: ${f.error}`));
      toast.success(`Screened ${run.processed} candidates in ${run.duration_seconds}s`);
    } catch (error) {
      toast.error((error as Error).message);
    } finally {
      setRunning(false);
      setProgress(null);
    }
  }

  function reset() {
    setJdText("");
    setFiles([]);
    setResult(null);
    setSelected(null);
    setQuery("");
    setLevelFilter("all");
    setSortKey("rank");
  }

  const rows = useMemo(() => {
    if (!result) return [];
    const filtered = result.candidates.filter((c) => {
      const matchesLevel = levelFilter === "all" || c.match_level === levelFilter;
      const q = query.trim().toLowerCase();
      const matchesQuery =
        !q ||
        c.profile.name.toLowerCase().includes(q) ||
        c.filename.toLowerCase().includes(q) ||
        c.matched_skills.some((s) => s.toLowerCase().includes(q));
      return matchesLevel && matchesQuery;
    });
    const sorters = {
      rank: (a: ScoredCandidate, b: ScoredCandidate) => a.rank - b.rank,
      score: (a: ScoredCandidate, b: ScoredCandidate) => b.overall_score - a.overall_score,
      name: (a: ScoredCandidate, b: ScoredCandidate) => a.profile.name.localeCompare(b.profile.name),
      experience: (a: ScoredCandidate, b: ScoredCandidate) =>
        b.profile.years_experience - a.profile.years_experience,
    };
    return [...filtered].sort(sorters[sortKey]);
  }, [result, query, levelFilter, sortKey]);

  return (
    <main className="min-h-screen bg-background">
      <Toaster position="top-right" />

      <header className="gradient-header px-6 py-10 text-primary-foreground">
        <div className="mx-auto max-w-6xl">
          <p className="text-sm uppercase tracking-widest opacity-80">ROOMAN Technologies Challenge</p>
          <h1 className="mt-2 text-3xl font-bold sm:text-4xl">AI Resume Screening Agent</h1>
          <p className="mt-3 max-w-2xl text-sm opacity-90 sm:text-base">
            Parse resumes, measure semantic relevance to a job description with embeddings and cosine
            similarity, score deterministically out of 100, then explain every ranking.
          </p>
        </div>
      </header>

      <div className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6">
        <div className="grid gap-6 lg:grid-cols-2">
          <Card className="shadow-card">
            <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
              <CardTitle className="flex items-center gap-2 text-base">
                <FileText className="size-4" /> Job description
              </CardTitle>
              <Button variant="secondary" size="sm" onClick={loadSample}>
                <Sparkles className="size-4" /> Load sample
              </Button>
            </CardHeader>
            <CardContent>
              <Textarea
                value={jdText}
                onChange={(event) => setJdText(event.target.value)}
                placeholder="Paste the job description here (role, responsibilities, required skills, experience, education)..."
                className="min-h-56 resize-y"
              />
              <p className="mt-2 text-xs text-muted-foreground">{jdText.length} characters</p>
            </CardContent>
          </Card>

          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Upload className="size-4" /> Resumes ({files.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <label
                className="flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed bg-surface-muted px-4 py-8 text-center transition-colors hover:border-primary"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  addFiles(event.dataTransfer.files);
                }}
              >
                <Upload className="mb-2 size-6 text-muted-foreground" />
                <span className="text-sm font-medium">Drop 10+ resumes here or click to browse</span>
                <span className="mt-1 text-xs text-muted-foreground">PDF, DOCX, TXT or MD - max 10 MB each</span>
                <input
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt,.md"
                  className="hidden"
                  onChange={(event) => addFiles(event.target.files)}
                />
              </label>

              <div className="max-h-40 space-y-1 overflow-y-auto">
                {files.map((file) => (
                  <div
                    key={file.name}
                    className="flex items-center justify-between rounded-md border px-3 py-1.5 text-sm"
                  >
                    <span className="truncate">{file.name}</span>
                    <button
                      type="button"
                      aria-label={`Remove ${file.name}`}
                      onClick={() => setFiles((prev) => prev.filter((f) => f.name !== file.name))}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      <X className="size-4" />
                    </button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={handleRun} disabled={running} size="lg">
            {running ? <Loader2 className="size-4 animate-spin" /> : <Search className="size-4" />}
            {running ? "Screening..." : "Screen candidates"}
          </Button>
          <Button variant="outline" onClick={reset} disabled={running}>
            <RotateCcw className="size-4" /> New screening
          </Button>
          {result && (
            <>
              <Button variant="secondary" onClick={() => downloadCsv(result)}>
                <Download className="size-4" /> CSV
              </Button>
              <Button variant="secondary" onClick={() => downloadJson(result)}>
                <Download className="size-4" /> JSON
              </Button>
            </>
          )}
        </div>

        {progress && (
          <Card>
            <CardContent className="space-y-2 py-4">
              <div className="flex justify-between text-sm">
                <span>{progress.label}</span>
                <span className="tabular-nums">
                  {progress.current}/{progress.total}
                </span>
              </div>
              <Progress value={progress.total ? (progress.current / progress.total) * 100 : 10} />
            </CardContent>
          </Card>
        )}

        {result && (
          <>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              {[
                { label: "Candidates processed", value: result.processed },
                { label: "Shortlisted", value: result.summary.shortlisted },
                { label: "Average score", value: result.summary.average_score },
                {
                  label: "Highest score",
                  value:
                    result.summary.highest_score ??
                    Math.max(...result.candidates.map((c) => c.overall_score)),
                },
                { label: "Files skipped", value: result.failed.length },
              ].map((stat) => (
                <Card key={stat.label} className="shadow-card">
                  <CardContent className="py-5">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">{stat.label}</p>
                    <p className="font-display text-2xl font-bold tabular-nums">{stat.value}</p>
                  </CardContent>
                </Card>
              ))}
            </div>

            <EngineTransparency result={result} />
            <ScoringMethodology weights={result.weights} />

            <Card className="shadow-card">
              <CardHeader className="gap-3 sm:flex-row sm:items-center sm:justify-between">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Award className="size-4" /> Ranked candidates
                </CardTitle>
                <div className="flex flex-wrap gap-2">
                  <Input
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search name or skill"
                    className="w-48"
                  />
                  <Select value={levelFilter} onValueChange={setLevelFilter}>
                    <SelectTrigger className="w-36">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All levels</SelectItem>
                      <SelectItem value="Strong">Strong</SelectItem>
                      <SelectItem value="Good">Good</SelectItem>
                      <SelectItem value="Moderate">Moderate</SelectItem>
                      <SelectItem value="Weak">Weak</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={sortKey} onValueChange={(value) => setSortKey(value as typeof sortKey)}>
                    <SelectTrigger className="w-40">
                      <ArrowUpDown className="size-4" />
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="rank">Sort by rank</SelectItem>
                      <SelectItem value="score">Sort by score</SelectItem>
                      <SelectItem value="name">Sort by name</SelectItem>
                      <SelectItem value="experience">Sort by experience</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-sm">
                  <thead>
                    <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <th className="py-2 pr-3">#</th>
                      <th className="py-2 pr-3">Candidate</th>
                      <th className="py-2 pr-3">Score</th>
                      <th className="py-2 pr-3">Match</th>
                      <th className="py-2 pr-3">Exp</th>
                      <th className="py-2 pr-3">Education</th>
                      <th className="py-2 pr-3">Top matched skills</th>
                      <th className="py-2 pr-3">Shortlist</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((candidate) => (
                      <tr
                        key={candidate.candidate_id}
                        onClick={() => setSelected(candidate)}
                        className="cursor-pointer border-b transition-colors last:border-0 hover:bg-surface-muted"
                      >
                        <td className="py-3 pr-3 font-medium tabular-nums">{candidate.rank}</td>
                        <td className="py-3 pr-3">
                          <div className="font-medium">{candidate.profile.name}</div>
                          <div className="text-xs text-muted-foreground">{candidate.filename}</div>
                        </td>
                        <td className="py-3 pr-3">
                          <div className="flex items-center gap-2">
                            <span className="font-display font-semibold tabular-nums">
                              {candidate.overall_score}
                            </span>
                            <Progress value={candidate.overall_score} className="h-1.5 w-20" />
                          </div>
                        </td>
                        <td className="py-3 pr-3">
                          <Badge className={LEVEL_CLASS[candidate.match_level]}>{candidate.match_level}</Badge>
                        </td>
                        <td className="py-3 pr-3 tabular-nums">{candidate.profile.years_experience}y</td>
                        <td className="py-3 pr-3">{candidate.profile.education_label}</td>
                        <td className="py-3 pr-3">
                          <div className="flex flex-wrap gap-1">
                            {candidate.matched_skills.slice(0, 3).map((skill) => (
                              <Badge key={skill} variant="secondary" className="text-xs">
                                {skill}
                              </Badge>
                            ))}
                            {candidate.matched_skills.length > 3 && (
                              <span className="text-xs text-muted-foreground">
                                +{candidate.matched_skills.length - 3}
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="py-3 pr-3">{candidate.shortlist ? "Yes" : "No"}</td>
                      </tr>
                    ))}
                    {!rows.length && (
                      <tr>
                        <td colSpan={8} className="py-6 text-center text-muted-foreground">
                          No candidates match the current filters.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </CardContent>
            </Card>

            {!!result.failed.length && (
              <Card className="border-weak/40">
                <CardHeader>
                  <CardTitle className="text-base">Skipped files ({result.failed.length})</CardTitle>
                </CardHeader>
                <CardContent className="space-y-1 text-sm text-muted-foreground">
                  {result.failed.map((f) => (
                    <p key={f.filename}>
                      <strong>{f.filename}</strong>: {f.error}
                    </p>
                  ))}
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>

      <Dialog open={!!selected} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-xl">
              #{selected?.rank} {selected?.profile.name}
            </DialogTitle>
          </DialogHeader>
          {selected && result && <CandidateDetail candidate={selected} weights={result.weights} />}
        </DialogContent>
      </Dialog>
    </main>
  );
}
