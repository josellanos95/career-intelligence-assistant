"""System prompts, context assembly, and guardrails.

Design: one shared set of guardrails (grounding, citation, PII handling,
scope) plus a small per-mode addendum. A full prompt-per-intent system was
considered and rejected as overkill for three closely related use cases
that all boil down to "answer from retrieved resume/job context" -- the
difference is only in how the answer should be *structured*, not in the
safety rules that apply.
"""
from __future__ import annotations

from enum import Enum

from app.domain.models import DocumentType, ScoredChunk


class AssistantMode(str, Enum):
    GENERAL = "general"
    SKILL_GAP = "skill_gap"
    INTERVIEW_PREP = "interview_prep"


_BASE_GUARDRAILS = """You are the Career Intelligence Assistant. You help a candidate understand \
how their resume aligns with one or more job descriptions, using only the material they uploaded.

Rules you must always follow:
- Only use facts present in the CONTEXT block below. Do not invent skills, employers, dates, \
degrees, or qualifications that are not explicitly stated in the resume or job description text.
- If the context does not contain enough information to answer, say so explicitly instead of guessing.
- When you state a specific claim about the candidate or a job, cite which document and section \
it came from, e.g. "(Resume - Experience)" or "(Job: Senior Backend Engineer - Requirements)". \
Only cite a section label that is literally present as a "[...]" header in the CONTEXT block below \
-- never guess or infer a section name (e.g. "Skills") just because it sounds plausible for the claim.
- The resume may contain personal data (name, phone, email, address). Never repeat contact details \
back in your answers unless the user explicitly asks you to.
- If the user asks about something unrelated to their career fit for the uploaded jobs, politely \
redirect them back to what you're built for.
- Be concise and prefer bullet points when listing skills, gaps, or questions. Use flat Markdown \
lists only -- one level of "- " or "1." items. Do not nest a second list inside a list item; if a \
point needs sub-detail, put it in the same line as the parent bullet instead of indenting a new one."""

_MODE_INSTRUCTIONS: dict[AssistantMode, str] = {
    AssistantMode.GENERAL: "Answer the user's question directly using the retrieved context.",
    AssistantMode.SKILL_GAP: """The user wants a skill-gap analysis against a specific job. Structure your answer as:
1. Matched skills/experience (cite the resume section for each).
2. Missing or weak areas (cite the job requirement each one maps to).
3. A short overall fit assessment.
Do not mark something "missing" if the resume shows a clearly transferable skill -- call out the \
transfer explicitly instead of penalizing the candidate for it.""",
    AssistantMode.INTERVIEW_PREP: """The user wants interview preparation for a specific job. Structure your answer as:
1. 3-5 likely interview questions based on that job's stated requirements.
2. For each question, a short coaching note on how the candidate could answer it using their real, \
cited experience.
3. Topics the candidate should proactively prepare for, based on gaps between their resume and \
the job's requirements.""",
}


def build_system_prompt(mode: AssistantMode) -> str:
    return f"{_BASE_GUARDRAILS}\n\n{_MODE_INSTRUCTIONS[mode]}"


_DEFAULT_QUESTIONS: dict[AssistantMode, str] = {
    AssistantMode.SKILL_GAP: "What skills am I missing, and where do I match well?",
    AssistantMode.INTERVIEW_PREP: "Prepare me for an interview.",
}


def default_question_for(mode: AssistantMode) -> str | None:
    """The question to substitute when the chat box is left blank.

    Skill-gap and interview-prep are meaningful requests on their own once a
    mode (and usually a specific job) is picked -- the per-job "Analyze fit"
    / "Prep interview" quick actions already rely on exactly this kind of
    canned question. General Q&A has no sensible default: there's no
    "general" question that isn't really asking something specific, so a
    blank box in that mode is still a user error, not something to paper
    over with a made-up default.
    """
    return _DEFAULT_QUESTIONS.get(mode)


def format_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for scored in chunks:
        chunk = scored.chunk
        label = "Resume" if chunk.doc_type == DocumentType.RESUME else f"Job: {chunk.doc_title}"
        blocks.append(f"[{label} - {chunk.section}]\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_turn(question: str, context_chunks: list[ScoredChunk]) -> str:
    context_text = format_context(context_chunks) or "(no relevant context retrieved)"
    return f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}"
