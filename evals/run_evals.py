"""Hands-on eval harness.

Runs real questions through the real embedding and LLM providers (reading
.env, exactly like the running app) against the fixed sample resume/job set
in samples/, then checks each answer for required and forbidden substrings.

Deliberately separate from tests/ and from CI: it costs real API calls and
money, and it measures answer *quality* (does retrieval surface the right
chunks, does the model ground its answer, does it stay honest about a bad
fit) rather than code correctness, which the mocked pytest suite already
covers. Run it manually after any change to chunking, retrieval, or prompts:

    python -m evals.run_evals
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.domain.models import DocumentType
from app.infra.embeddings.openai_embedder import OpenAIEmbedder
from app.infra.llm.factory import get_llm_provider
from app.infra.vectorstore.numpy_store import NumpyVectorStore
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from evals.cases import CASES, RESUME_PATH, Requirement

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EVAL_STORE_DIR = Path(__file__).resolve().parent / ".eval_store"
_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "latest.json"


def _requirement_met(requirement: Requirement, answer_lower: str) -> bool:
    alternatives = requirement if isinstance(requirement, tuple) else (requirement,)
    return any(alt.lower() in answer_lower for alt in alternatives)


def _requirement_label(requirement: Requirement) -> str:
    return " / ".join(requirement) if isinstance(requirement, tuple) else requirement


def _reset_eval_store() -> None:
    if _EVAL_STORE_DIR.exists():
        for path in _EVAL_STORE_DIR.glob("*"):
            path.unlink()


def _load_fixtures(ingestion: IngestionService, retrieval: RetrievalService) -> dict[str, str]:
    """Ingest the resume and every job description in samples/.

    Returns a map of job file path (as referenced by evals.cases) -> doc_id,
    so cases can scope retrieval to a single job the same way the UI's
    "Focus on job" selector does.
    """
    job_dir = _PROJECT_ROOT / "samples" / "job_descriptions"
    doc_ids: dict[str, str] = {}

    resume_bytes = (_PROJECT_ROOT / RESUME_PATH).read_bytes()
    resume_chunks = ingestion.ingest("jordan_avery_resume.txt", resume_bytes, DocumentType.RESUME)
    retrieval.index_chunks(resume_chunks)

    for job_path in sorted(job_dir.glob("*.txt")):
        content = job_path.read_bytes()
        chunks = ingestion.ingest(job_path.name, content, DocumentType.JOB_DESCRIPTION, title=job_path.stem)
        retrieval.index_chunks(chunks)
        doc_ids[f"samples/job_descriptions/{job_path.name}"] = chunks[0].doc_id

    return doc_ids


def main() -> int:
    settings = get_settings()
    _reset_eval_store()

    embedder = OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.openai_embedding_model)
    llm = get_llm_provider(settings)
    store = NumpyVectorStore(str(_EVAL_STORE_DIR))
    retrieval = RetrievalService(embedder=embedder, store=store)
    ingestion = IngestionService()
    chat_service = ChatService(retrieval=retrieval, llm=llm, top_k=settings.max_context_chunks)

    doc_ids = _load_fixtures(ingestion, retrieval)

    results = []
    passed = 0
    for case in CASES:
        doc_id = doc_ids[case.job_file] if case.job_file else None
        start = time.perf_counter()
        response = chat_service.ask(case.question, mode=case.mode, doc_id=doc_id)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        answer_lower = response.text.lower()
        missing = [_requirement_label(r) for r in case.must_include if not _requirement_met(r, answer_lower)]
        leaked = [s for s in case.must_not_include if s.lower() in answer_lower]
        ok = not missing and not leaked
        passed += ok

        results.append(
            {
                "id": case.id,
                "passed": ok,
                "missing_required": missing,
                "unexpected_present": leaked,
                "latency_ms": elapsed_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model": response.model,
                "answer": response.text,
            }
        )

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.id} ({elapsed_ms}ms, {response.input_tokens}+{response.output_tokens} tok)")
        if missing:
            print(f"         missing required substrings: {missing}")
        if leaked:
            print(f"         unexpectedly present substrings: {leaked}")

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{passed}/{len(CASES)} cases passed. Full answers written to {_RESULTS_PATH}.")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
