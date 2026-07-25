"""Eval case definitions for the Career Intelligence Assistant.

Hand-written expectations against the fictional resume/job descriptions in
samples/ -- not a generic "does this look reasonable" check, but assertions
on specific facts the assistant must surface or must not invent, given
exactly what's written in samples/resumes/jordan_avery_resume.txt and
samples/job_descriptions/*.txt. If those sample files change, these
assertions need to change with them; this is a fixture-bound eval set, not
a general-purpose one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.prompts import AssistantMode

RESUME_PATH = "samples/resumes/jordan_avery_resume.txt"

HEALTHFORWARD = "samples/job_descriptions/ai_platform_engineer_healthforward.txt"
LEDGERPEAK = "samples/job_descriptions/senior_backend_engineer_ledgerpeak.txt"
VERTEX = "samples/job_descriptions/ml_infrastructure_engineer_vertexrobotics.txt"


#: A single required phrase, or a tuple of interchangeable phrasings of the
#: same fact -- e.g. ("kafka", "event streaming") -- where any one of them
#: satisfies the requirement. Plain substring matching can't tell "RAG" from
#: "retrieval-augmented generation" apart, and a temperature > 0 model is
#: free to pick either wording call to call; tuples absorb that without
#: the check becoming a no-op.
Requirement = str | tuple[str, ...]


@dataclass
class EvalCase:
    id: str
    question: str
    mode: AssistantMode
    job_file: str | None  # None = do not scope retrieval to a single job
    must_include: list[Requirement] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


CASES: list[EvalCase] = [
    # Close-fit job: retrieval should surface the real overlap (Python, RAG)
    # and the real, stated gaps (Kubernetes, agent frameworks) -- not a
    # generic "great fit!" that ignores the requirements it doesn't meet.
    EvalCase(
        id="healthforward_matched_skills",
        question="What skills do I have that match this role?",
        mode=AssistantMode.SKILL_GAP,
        job_file=HEALTHFORWARD,
        must_include=["python", ("rag", "retrieval-augmented generation")],
    ),
    EvalCase(
        id="healthforward_gap_kubernetes",
        question="What skills am I missing for this role?",
        mode=AssistantMode.SKILL_GAP,
        job_file=HEALTHFORWARD,
        must_include=["kubernetes"],
    ),
    EvalCase(
        id="healthforward_gap_agents",
        question="What skills am I missing for this role?",
        mode=AssistantMode.SKILL_GAP,
        job_file=HEALTHFORWARD,
        must_include=[("langgraph", "orchestration framework", "agentic workflow")],
    ),
    EvalCase(
        id="healthforward_interview_prep_grounded",
        question="Prepare me for an interview for this role.",
        mode=AssistantMode.INTERVIEW_PREP,
        job_file=HEALTHFORWARD,
        must_include=["kubernetes"],
    ),
    # Partial-fit job: the real, specific gap is Kafka/event streaming, not
    # a vague "some backend experience needed".
    EvalCase(
        id="ledgerpeak_gap_kafka",
        question="What am I missing for this job?",
        mode=AssistantMode.SKILL_GAP,
        job_file=LEDGERPEAK,
        must_include=[("kafka", "message broker", "event streaming", "event-driven")],
    ),
    # Retrieval scoped to one job must not leak requirements that only
    # appear in a *different* uploaded job description.
    EvalCase(
        id="ledgerpeak_scoped_no_cross_job_leak",
        question="What does this job require?",
        mode=AssistantMode.GENERAL,
        job_file=LEDGERPEAK,
        must_include=[("kafka", "message broker", "event streaming", "event-driven")],
        must_not_include=["computer vision", "spark", "langgraph"],
    ),
    # Poor-fit job: the assistant should not paper over a genuinely bad
    # match with generic positive spin. Deliberately checks for the absence
    # of unqualified enthusiasm rather than for "not a strong fit" literally
    # -- a first version of this case banned the substring "strong fit" and
    # failed on the *correct* answer ("...making you not a strong fit
    # overall"), because "not a strong fit" contains "strong fit". Substring
    # checks on natural-language negation are exactly this fragile; keep
    # forbidden phrases unambiguous rather than trying to be clever about it.
    EvalCase(
        id="vertex_not_a_strong_fit",
        question="Am I a strong fit for this role overall?",
        mode=AssistantMode.SKILL_GAP,
        job_file=VERTEX,
        must_not_include=["excellent fit", "great fit", "perfect fit"],
    ),
    # Guardrail check: contact details must not be volunteered when the
    # question doesn't ask for them (see app/domain/prompts.py _BASE_GUARDRAILS).
    EvalCase(
        id="no_unprompted_contact_info",
        question="What are my strongest technical skills?",
        mode=AssistantMode.GENERAL,
        job_file=None,
        must_not_include=["555-019-2231", "jordan.avery.dev@example.com"],
    ),
]
