"""Reject STS sides that still look like SOP letterhead or empty tables."""

from __future__ import annotations

import re

PAGE_RE = re.compile(r"Page\s+\d+\s+of", re.IGNORECASE)
SOP_NUMBER_RE = re.compile(r"SOP-Number", re.IGNORECASE)
PAGE_COMMENT_RE = re.compile(r"<!--\s*page(?:-break)?[^>]*-->", re.IGNORECASE)
IMG_CAPTION_RE = re.compile(r"\[IMG:\d+\]")
# Markdown table separator or a row with no word characters.
PIPE_ONLY_ROW_RE = re.compile(r"^\s*\|[\s\-:|]*\|\s*$")
PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9\u00C0-\u024F\u4E00-\u9FFF\u0E00-\u0E7F]+")


def strip_headers(text: str) -> str:
    """Drop page comments, page-n-of, SOP-Number, and pipe-only table rows.

    Used as TF-IDF input and as a gist fallback. Seed keep/skip is otherwise
    out of scope: remaining letterhead still goes to the LLM.
    """
    cleaned = PAGE_COMMENT_RE.sub("", text)
    cleaned = IMG_CAPTION_RE.sub("", cleaned)
    lines: list[str] = []
    for raw in cleaned.splitlines():
        line = PAGE_RE.sub("", raw)
        line = SOP_NUMBER_RE.sub("", line)
        if PIPE_ONLY_ROW_RE.match(line):
            continue
        if PIPE_ROW_RE.match(line) and not WORD_RE.search(line.replace("|", " ")):
            continue
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def has_pipe_only_table(text: str) -> bool:
    rows = [ln for ln in text.splitlines() if "|" in ln]
    if not rows:
        return False
    pipe_rows = [ln for ln in rows if PIPE_ROW_RE.match(ln) or PIPE_ONLY_ROW_RE.match(ln)]
    if not pipe_rows:
        return False
    # A pipe-only table is a table block whose cells have no real words.
    wordy = 0
    for ln in pipe_rows:
        cells = ln.replace("|", " ")
        if PIPE_ONLY_ROW_RE.match(ln):
            continue
        if WORD_RE.search(cells):
            wordy += 1
    return wordy == 0 and len(pipe_rows) >= 2


GENERIC_QUESTION_RE = re.compile(
    r"(?is)^\s*(?:"
    r"what\s+(?:is|are)\s+(?:this|the)\s+(?:document|text|passage|chunk|paper|sop|section)\b"
    r"|what\s+is\s+the\s+main\s+(?:idea|topic|research|theme|purpose of this)"
    r"|summarize\s+(?:this|the)\b"
    r"|what\s+does\s+this\s+(?:text|passage|document|chunk)\s+(?:say|talk|discuss|cover)"
    r"|what\s+is\s+the\s+title\s+of\s+(?:this|the)\s+(?:document|paper|text)"
    r")"
)


def reject_reason(text: str, *, min_chars: int = 20) -> str | None:
    if not text or not text.strip():
        return "empty"
    if PAGE_RE.search(text):
        return "page_marker"
    if SOP_NUMBER_RE.search(text):
        return "sop_number"
    if has_pipe_only_table(text):
        return "pipe_only_table"
    if len(text.strip()) < min_chars:
        return "too_short"
    return None


def question_reject_reason(text: str) -> str | None:
    """Letterhead junk plus generic / summary questions."""
    reason = reject_reason(text, min_chars=12)
    if reason:
        return reason
    if GENERIC_QUESTION_RE.search(text):
        return "generic"
    return None


def pair_reject_reason(sentence1: str, sentence2: str) -> str | None:
    for side, label in ((sentence1, "sentence1"), (sentence2, "sentence2")):
        reason = reject_reason(side)
        if reason:
            return f"{label}:{reason}"
    if sentence1.strip() == sentence2.strip():
        return "identical"
    return None


def split_sentences(text: str, max_sentences: int = 8) -> str:
    stripped = strip_headers(text)
    if not stripped:
        return ""
    parts = re.split(r"(?<=[.!?。！？])\s+", stripped)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return stripped
    return " ".join(parts[:max_sentences])
