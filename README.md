# AI Resume Screening Agent

An end-to-end agent for the **ROOMAN Technologies — Junior AI Research Associate 24-Hour AI Agent Challenge**.

Give it **1 job description + 10 or more resumes** (PDF / DOCX / TXT / MD). It parses every document,
extracts structured candidate data, measures semantic relevance with Sentence Transformer embeddings and
cosine similarity, scores each candidate deterministically out of 100, ranks them, explains every ranking
with an LLM, and exports CSV / JSON.

---

## 1. Architecture

```text
                 ┌──────────────────────── React + TypeScript + Tailwind dashboard ───────────────────────┐
JD text/file ───▶│ upload → progress → ranked table → candidate detail → CSV / JSON export               │
10+ resumes  ───▶└───────────────┬──────────────────────────────────────────────────────────────────────-┘
                                 │ POST /screen (multipart)
                 ┌───────────────▼───────────────── Python FastAPI backend ────────────────────────────────┐
                 │ parsers/    PyMuPDF (PDF) · python-docx (DOCX) · plain text (TXT/MD) → clean text       │
                 │ parsers/extractor.py  name, email, phone, links, skills, years, education, projects     │
                 │ nlp/embeddings.py     Sentence Transformers all-MiniLM-L6-v2  (TF-IDF fallback)         │
                 │ nlp/similarity.py     scikit-learn cosine_similarity (pure-python fallback)             │
                 │ scoring/scorer.py     deterministic weighted rubric → score /100 → rank                 │
                 │ llm/reasoning.py      OpenAI: strengths, gaps, reasoning, recommendation (NO numbers)   │
                 │ exporters.py          CSV / JSON                                                        │
                 └─────────────────────────────────────────────────────────────────────────────────────────┘
```

If the Python service is not running, the dashboard runs the **same deterministic rubric** in the browser
(TF-IDF + cosine + template reasoning) and labels itself **Fallback mode** everywhere — it never claims
Sentence Transformers or OpenAI when they were not used.

### Repository layout

```text
backend/
  main.py                 FastAPI app: /health, /screen, /export/csv, /export/json
  run_samples.py          CLI: screen the bundled JD + 13 resumes, write sample output
  parsers/                document_parser.py (PDF/DOCX/TXT), extractor.py (profile extraction)
  nlp/                    embeddings.py (Sentence Transformers | TF-IDF), similarity.py (cosine)
  scoring/                scorer.py (weighted rubric), skills.py (120+ skill taxonomy + aliases)
  llm/                    reasoning.py (OpenAI explanations + deterministic template fallback)
  tests/                  22 pytest tests across parsers, NLP and scoring
  samples/                job_description.txt, resumes/ (13), output/ (results.json, results.csv)
  .env.example, requirements.txt
src/
  routes/index.tsx        HR dashboard (upload, stats, search/filter/sort, detail modal, exports)
  lib/screening/          TypeScript mirror of the engine used in browser-fallback mode
```

---

## 2. Setup and run

### Backend (real NLP path — recommended)

```sh
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # optional: add OPENAI_API_KEY
uvicorn main:app --reload --port 8000
```

`GET http://localhost:8000/health` reports the live engine: parser, embedding model, similarity method,
scoring method and LLM status.

### Frontend

```sh
npm install
npm run dev            # http://localhost:8080
```

Set `VITE_API_URL=http://localhost:8000` (default) so the dashboard uses the Python engine. Without a
reachable backend it automatically switches to the in-browser fallback engine.

### CLI (no UI, reproducible run)

```sh
cd backend && python run_samples.py
# writes samples/output/results.json and samples/output/results.csv
```

### Tests

```sh
cd backend && python -m pytest -q      # 22 passed
```

---

## 3. Scoring formula (deterministic, /100)

| Component | Weight | How it is computed |
|---|---|---|
| Skills match | **30** | Overlap of JD skills with resume skills, normalised by the JD skill count (taxonomy + aliases, e.g. `sklearn → scikit-learn`) |
| Semantic similarity | **25** | Cosine similarity between the JD embedding and the resume embedding, clamped to `[0, 1]` |
| Relevant experience | **20** | Detected years vs. the JD minimum; partial credit below the minimum, capped bonus above it |
| Education | **10** | Highest detected qualification level vs. the level the JD asks for |
| Required technologies | **10** | Share of explicitly required tools present in the resume |
| Project / domain relevance | **5** | Domain keywords found in project / summary sections, small bonus for concrete projects |

```text
overall = 30·skills + 25·semantic + 20·experience + 10·education + 10·required_tech + 5·projects
          (each component is a 0–1 ratio, rounded to 2 decimals at the end)

Strong ≥ 80 · Good ≥ 65 · Moderate ≥ 45 · otherwise Weak
Shortlisted when overall ≥ 65 and not every required technology is missing
```

Scoring lives entirely in `backend/scoring/scorer.py`. It is pure, side-effect free and fully unit tested,
so the same inputs always yield the same ranking.

---

## 4. NLP methodology

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalised). The JD and all resumes are
  encoded in a single batch, so one run costs one forward pass over `n + 1` documents.
- **Similarity:** `sklearn.metrics.pairwise.cosine_similarity` between the JD vector and every resume
  vector. Negative cosine is clamped to 0 (negative relevance is simply no relevance).
- **Fallback:** if the model cannot be downloaded (offline / air-gapped), a deterministic TF-IDF vectoriser
  over the current batch is used. The response and the UI both say so.
- **Extraction:** rule-based and explainable — regex for email/phone/links, section detection for
  education, projects and certifications, taxonomy lookup with aliases and word-boundary matching for
  skills, and multiple year patterns (`4+ years`, date ranges, `since 2019`) for experience.

## 5. Role of the LLM

The LLM is an **explainer, never a scorer**. It receives the already-computed numbers and must reuse them
verbatim to produce: strengths, gaps, matched/missing skill narrative, the reasoning paragraph and the
shortlist recommendation. Configure via `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Without a key — or if the API call fails — a deterministic template produces the same shape of
explanation, tagged `Deterministic fallback` in the UI and `reasoning_source: "fallback"` in the JSON.

---

## 6. Sample results

`backend/samples/` ships one JD (Junior AI Research Associate) and **13 resumes**, including a deliberately
corrupt PDF to prove the batch never aborts. A TF-IDF-fallback run produces:

```text
Role: Junior AI Research Associate
 1. Aarav Sharma      68.43/100  Good
 2. Fatima Khan       66.42/100  Good
 3. Priya Nair        59.62/100  Moderate
 4. Karthik Reddy     54.22/100  Moderate
 5. Ananya Das        53.79/100  Moderate
 ...
13. Divya Rao         21.47/100  Weak
Skipped: corrupt_sample.pdf — could not be read
```

Full artefacts: `backend/samples/output/results.json`, `backend/samples/output/results.csv`.
(Scores shift slightly upward for genuinely relevant candidates when the Sentence Transformer model is
available, because dense embeddings capture paraphrase that TF-IDF misses.)

---

## 7. Reliability

- File validation: extension allow-list, non-empty check, 10 MB cap.
- Corrupt, scanned or empty documents are collected into `failed[]` with the reason and shown in the
  dashboard — the rest of the batch still completes.
- Missing fields (no email, no dates, no education section) degrade gracefully to neutral component scores
  instead of raising.
- LLM/API failures fall back to template reasoning per candidate; a network outage cannot fail a run.

## 8. Tradeoffs and limitations

- Extraction is rule-based, not a trained NER model: unusual resume layouts can mis-detect a name or miss
  a year range. It is transparent and debuggable, which suits a scoring rubric that must be auditable.
- No OCR: scanned image-only PDFs are reported as unreadable rather than silently scored on empty text.
- `all-MiniLM-L6-v2` truncates long inputs (~256 word pieces), so very long resumes are compared on their
  leading content. Section-wise embedding would improve this.
- The skill taxonomy is curated (120+ terms) and English-only; niche tools outside it are counted only
  through semantic similarity.
- The last run is kept in memory for the export endpoints — fine for a single-user demo, not multi-tenant.
- Browser-fallback mode uses TF-IDF, so its absolute numbers are lower than the Sentence Transformer path;
  the ranking logic is identical.

## 9. ROOMAN requirement mapping

| Requirement | Where it is implemented |
|---|---|
| 1 JD + 10+ resumes, PDF/DOCX/TXT | `parsers/document_parser.py`, dashboard uploader, 13 sample resumes |
| Candidate information extraction | `parsers/extractor.py` (contact, skills, experience, education, projects) |
| NLP relevance calculation | `nlp/embeddings.py` + `nlp/similarity.py` (Sentence Transformers + cosine) |
| Deterministic scoring 30/25/20/10/10/5 | `scoring/scorer.py`, mirrored in the dashboard methodology panel |
| Ranking | `rank_candidates()` — stable sort by score, rank assigned server-side |
| AI reasoning | `llm/reasoning.py` (OpenAI via `.env`, template fallback) |
| JSON / CSV output | `exporters.py`, `/export/json`, `/export/csv`, dashboard download buttons |
| HR dashboard | `src/routes/index.tsx`: stats, search, filter, sort, score bars, progress, detail modal |
| Explainability & transparency | Engine transparency panel + scoring methodology section + per-candidate breakdown |
| Reproducibility | Pure scoring functions, 22 pytest tests, `run_samples.py`, committed sample output |
