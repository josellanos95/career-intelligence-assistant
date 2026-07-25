"""Renders the LLM's Markdown answers to sanitized HTML for the chat UI.

The Messages/Chat Completions APIs return plain text; both providers'
default prompting style is Markdown (headers, bold, bullet lists). Showing
that text as-is in a `white-space: pre-wrap` div left literal "**"/"-"/"#"
characters on screen instead of formatted output. Rendering happens here,
not in the JSON API -- API consumers get the raw text and can render it
however they want.

Sanitized rather than trusted outright: the guardrails constrain the model,
but nothing stops a prompt-injected job description from asking it to emit
`<script>` in its answer.
"""
from __future__ import annotations

import bleach
import markdown

_ALLOWED_TAGS = [
    "p", "br", "strong", "em", "code", "pre", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "a",
]
_ALLOWED_ATTRIBUTES = {"a": ["href", "title", "rel"]}


def render_markdown(text: str) -> str:
    # nl2br: LLM answers often nest "- sub-bullet" lines under a numbered
    # point with 2-3 spaces of indent, which is one space short of what
    # python-markdown needs to parse them as a real nested <ul> -- they'd
    # otherwise collapse into a single run-on line of HTML whitespace.
    # nl2br keeps each line visually separated even when that happens.
    html = markdown.markdown(text, extensions=["extra", "sane_lists", "nl2br"])
    return bleach.clean(html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRIBUTES, strip=True)
