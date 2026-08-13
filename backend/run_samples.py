"""CLI: screen the bundled sample JD + resumes and write CSV/JSON outputs.

    cd backend && python run_samples.py
"""

from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv

from exporters import to_csv, to_json
from llm.reasoning import generate_batch_reasoning, llm_status
from nlp.embeddings import get_backend
from nlp.similarity import cosine_similarity
from parsers.document_parser import ParseError, parse_document
from parsers.extractor import extract_profile
from scoring.scorer import WEIGHTS, parse_job_description, rank_candidates, score_candidate

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")
OUT = os.path.join(SAMPLES, "output")


def main() -> None:
    load_dotenv()
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(SAMPLES, "job_description.txt"), encoding="utf-8") as fh:
        jd = parse_job_description(fh.read())

    folder = os.path.join(SAMPLES, "resumes")
    parsed, failures = [], []
    for filename in sorted(os.listdir(folder)):
        try:
            doc = parse_document(os.path.join(folder, filename))
            parsed.append((doc, extract_profile(doc.text)))
        except ParseError as exc:
            failures.append({"filename": filename, "error": str(exc)})

    backend = get_backend()
    vectors = backend.encode([jd.text] + [doc.text for doc, _ in parsed])
    jd_vec, resume_vecs = vectors[0], vectors[1:]

    scored = [
        score_candidate(
            jd=jd,
            profile=profile,
            resume_text=doc.text,
            candidate_id=str(uuid.uuid4()),
            filename=doc.filename,
            similarity=max(0.0, min(1.0, cosine_similarity(jd_vec, vec))),
            warnings=doc.warnings,
        )
        for (doc, profile), vec in zip(parsed, resume_vecs)
    ]
    ranked = rank_candidates(scored)
    generate_batch_reasoning(ranked, jd.title)

    payload = {
        "job": jd.to_dict(),
        "weights": WEIGHTS,
        "embedding_backend": backend.name,
        "llm": llm_status(),
        "failed": failures,
        "candidates": [c.to_dict() for c in ranked],
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        fh.write(to_json(payload))
    with open(os.path.join(OUT, "results.csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write(to_csv(ranked))

    print(f"Role: {jd.title} | backend: {backend.name} | llm: {llm_status()['mode']}")
    for c in ranked:
        print(f"{c.rank:>2}. {c.profile.name:<26} {c.overall_score:>6}/100  {c.match_level:<8} {c.filename}")
    if failures:
        print("Skipped:", failures)
    print(f"\nWrote {OUT}/results.json and results.csv")


if __name__ == "__main__":
    main()
