from __future__ import annotations


def question_prompt(seed: str) -> str:
    """One specific question answerable only from this chunk."""
    return f"""Based on the following text, generate exactly one specific question
that can be answered using only this text.
Return only the question itself, no explanations or preamble.

The question must ask for a fact, step, number, name, or rule that is in the
text — the kind of thing a staff member would ask to find this passage.

The question MUST be in the language of the Text below (English Text → English
question, Chinese Text → Chinese question, Malay Text → Malay question).
Do not translate the question into a different language.

Do not mention page numbers, SOP-Number, letterhead, figures with no content,
or empty tables. Do not ask what the document or passage is "about", and do
not ask for a summary.

Text:
---
{seed}
---
"""
