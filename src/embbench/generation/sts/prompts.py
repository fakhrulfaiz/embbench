from __future__ import annotations

HIGH_SCORE_RULES = {
    5.0: "Same meaning; paraphrase only. Do not add or drop facts.",
    4.0: "Same meaning; at most one extra or omitted detail.",
    3.0: (
        "Same SOP topic, but a different step or adjacent rule "
        "(e.g. leave vs sick leave; proforma vs PI approval). Not a paraphrase."
    ),
}


def high_score_prompt(seed: str, score: float) -> str:
    rule = HIGH_SCORE_RULES[score]
    return f"""You write STS evaluation pairs from an SOP / procedure chunk.

Seed chunk:
---
{seed}
---

Target score: {score} on a 0–5 scale.

Steps:
1. Extract a short gist of the seed (2–8 sentences of actual procedure).
   Drop letterhead, page markers (Page N of), SOP-Number, empty markdown
   tables, and pipe-only tables.
   Write in the same language as the seed.
2. sentence1 = that gist. It must be entailed by the chunk.
3. sentence2 follows this rule: {rule}

Do not copy "Page N of", "SOP-Number", or markdown tables into either sentence.
Do not label two raw page-slices. You are writing both sides from this one seed.

Return JSON only:
{{"sentence1": "...", "sentence2": "...", "score": {score}}}
"""


def gist_prompt(passage: str) -> str:
    return f"""Rewrite this SOP / procedure passage into a short gist (2–8 sentences).

Drop letterhead, page markers (Page N of), SOP-Number, empty markdown
tables, and pipe-only tables.
Keep the actual procedure. Write in the same language as the passage.

Passage:
---
{passage}
---

Return JSON only:
{{"gist": "..."}}
"""
