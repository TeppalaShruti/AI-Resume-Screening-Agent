"""FastAPI application for the AI Resume Screening Agent.

Run:  uvicorn main:app --reload --port 8000   (from the backend/ directory)
"""

from __future__ import annotations

import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from exporters import to_csv
from llm.reasoning import generate_batch_reasoning, llm_status
from nlp.embeddings import get_backend
from nlp.similarity import cosine_similarity_matrix, similarity_backend
from parsers.document_parser import ParseError, SUPPORTED_EXTENSIONS, parse_bytes
from parsers.extractor import extract_profile
from scoring.scorer import WEIGHTS, parse_job_description, rank_candidates, score_candidate

load_dotenv()

app = FastAPI(title="AI Resume Screening Agent", version="1.0.0")

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LAST_RUN: dict = {}


def engine_info() -> dict:
    """Honest description of what actually ran (never claims unused components)."""
    backend = get_backend()
    llm = llm_status()
    return {
        "runtime": "python-fastapi",
        "parser": "PyMuPDF (PDF) · python-docx (DOCX) · plain text (TXT/MD)",
        "embeddings": (
            f"Sentence Transformers ({backend.name.split(':', 1)[-1]})"
            if backend.name.startswith("sentence-transformers")
            else "TF-IDF fallback (Sentence Transformers unavailable)"
        ),
        "similarity": similarity_backend(),
        "scoring": "Deterministic weighted rubric in scoring/scorer.py (Python, /100)",
        "llm": (
            f"OpenAI {llm['model']} — explanations only, never numbers"
            if llm["enabled"]
            else "Fallback mode: deterministic template reasoning (no OPENAI_API_KEY set)"
        ),
        "fallback_mode": (not llm["enabled"]) or not backend.name.startswith("sentence-transformers"),
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm": llm_status(),
        "embedding_backend": get_backend().name,
        "engine_info": engine_info(),
        "weights": WEIGHTS,
        "supported_files": sorted(SUPPORTED_EXTENSIONS),
    }


@app.post("/screen")
async def screen(
    jd_text: str = Form(default=""),
    jd_file: UploadFile | None = File(default=None),
    resumes: list[UploadFile] = File(default=[]),
    use_llm: bool = Form(default=True),
) -> dict:
    started = time.time()

    if jd_file is not None:
        try:
            jd_text = parse_bytes(jd_file.filename or "jd.txt", await jd_file.read()).text
        except ParseError as exc:
            raise HTTPException(status_code=400, detail=f"Job description: {exc}")
    if not jd_text or len(jd_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Provide a job description of at least 50 characters.")
    if not resumes:
        raise HTTPException(status_code=400, detail="Upload at least one resume.")

    jd = parse_job_description(jd_text.strip())

    parsed, failures = [], []
    for upload in resumes:
        name = upload.filename or "resume"
        try:
            doc = parse_bytes(name, await upload.read())
            profile = extract_profile(doc.text)
            parsed.append((doc, profile))
        except ParseError as exc:
            failures.append({"filename": name, "error": str(exc)})
        except Exception as exc:  # a single bad file never aborts the batch
            failures.append({"filename": name, "error": f"Unexpected error: {exc}"})

    if not parsed:
        raise HTTPException(status_code=422, detail={"message": "No resume could be parsed.", "failures": failures})

    # Batch-encode once: JD first, then every resume -> cosine similarity.
    backend = get_backend()
    vectors = backend.encode([jd.text] + [doc.text for doc, _ in parsed])
    jd_vec, resume_vecs = vectors[0], vectors[1:]
    similarities = cosine_similarity_matrix(jd_vec, resume_vecs)

    scored = []
    for (doc, profile), raw_similarity in zip(parsed, similarities):
        similarity = max(0.0, min(1.0, raw_similarity))
        scored.append(
            score_candidate(
                jd=jd,
                profile=profile,
                resume_text=doc.text,
                candidate_id=str(uuid.uuid4()),
                filename=doc.filename,
                similarity=similarity,
                warnings=doc.warnings,
            )
        )

    ranked = rank_candidates(scored)
    if use_llm:
        generate_batch_reasoning(ranked, jd.title)
    else:
        from llm.reasoning import template_reasoning

        for candidate in ranked:
            candidate.reasoning = template_reasoning(candidate, jd.title)
            candidate.reasoning_source = "fallback"

    result = {
        "run_id": str(uuid.uuid4()),
        "job": jd.to_dict(),
        "weights": WEIGHTS,
        "embedding_backend": backend.name,
        "llm": llm_status(),
        "engine_info": engine_info(),
        "processed": len(ranked),
        "failed": failures,
        "duration_seconds": round(time.time() - started, 2),
        "candidates": [c.to_dict() for c in ranked],
        "summary": {
            "shortlisted": sum(1 for c in ranked if c.shortlist),
            "strong": sum(1 for c in ranked if c.match_level == "Strong"),
            "good": sum(1 for c in ranked if c.match_level == "Good"),
            "moderate": sum(1 for c in ranked if c.match_level == "Moderate"),
            "weak": sum(1 for c in ranked if c.match_level == "Weak"),
            "average_score": round(sum(c.overall_score for c in ranked) / len(ranked), 2),
            "highest_score": max(c.overall_score for c in ranked),
        },
    }
    _LAST_RUN.clear()
    _LAST_RUN.update({"result": result, "ranked": ranked})
    return result


@app.get("/export/csv", response_class=PlainTextResponse)
def export_csv() -> str:
    if "ranked" not in _LAST_RUN:
        raise HTTPException(status_code=404, detail="No screening run available. POST /screen first.")
    return to_csv(_LAST_RUN["ranked"])


@app.get("/export/json")
def export_json() -> dict:
    if "result" not in _LAST_RUN:
        raise HTTPException(status_code=404, detail="No screening run available. POST /screen first.")
    return _LAST_RUN["result"]
