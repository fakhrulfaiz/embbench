from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass(frozen=True)
class Chunk:
    chunk_id: UUID
    doc_id: UUID
    content: str
    language: str
    chunker_profile: str
    created_at: datetime | None = None


@dataclass
class PairDraft:
    sentence1: str
    sentence2: str
    score: float
    pair_kind: str
    language: str
    source_chunk_id: UUID | None = None
    source_chunk_id_2: UUID | None = None
    pair_id: UUID = field(default_factory=uuid4)


@dataclass
class QuestionDraft:
    question_text: str
    gold_chunk_id: UUID
    language: str
    question_id: UUID = field(default_factory=uuid4)


HIGH_SCORES = (3.0, 4.0, 5.0)
LOW_SCORES = (0.0, 1.0, 2.0)

KIND_BY_SCORE = {
    5.0: "paraphrase",
    4.0: "generated",
    3.0: "generated",
    2.0: "chunk_chunk",
    1.0: "chunk_chunk",
    0.0: "chunk_chunk",
}

# Target mix from docs/sts-generation.md
SCORE_SHARES: dict[float, float] = {
    0.0: 0.10,
    1.0: 0.20,
    2.0: 0.20,
    3.0: 0.20,
    4.0: 0.20,
    5.0: 0.10,
}

LANG_EXPORT_ALIAS = {
    "en": "eng-Latn",
    "eng": "eng-Latn",
    "zh": "cmn-Hans",
    "zho": "cmn-Hans",
    "cmn": "cmn-Hans",
    "ms": "zsm-Latn",
    "msa": "zsm-Latn",
    "zlm": "zsm-Latn",
    "zsm": "zsm-Latn",
    "may": "zsm-Latn",
    "all": "mul",
    "*": "mul",
    "mul": "mul",
}


def is_all_languages(language: str | None) -> bool:
    token = (language or "").strip().lower()
    return token in {"", "all", "*", "any", "mul"}


def language_filter_aliases(language: str | None) -> tuple[str, ...] | None:
    """DB `language` prefixes to match, or None to include every language."""
    if is_all_languages(language):
        return None
    return db_language_aliases(language)


def export_language(language: str) -> str:
    token = language.strip()
    if "-" in token:
        return token
    return LANG_EXPORT_ALIAS.get(token.lower(), token)


def db_language_aliases(language: str) -> tuple[str, ...]:
    """Accept both DB (`en`) and export (`eng-Latn`) codes when filtering."""
    token = language.strip().lower().split("-")[0]
    groups = {
        "en": ("en", "eng"),
        "eng": ("en", "eng"),
        "zh": ("zh", "zho", "cmn"),
        "zho": ("zh", "zho", "cmn"),
        "cmn": ("zh", "zho", "cmn"),
        "ms": ("ms", "msa", "zlm", "zsm", "may"),
        "msa": ("ms", "msa", "zlm", "zsm", "may"),
        "zlm": ("ms", "msa", "zlm", "zsm", "may"),
        "zsm": ("ms", "msa", "zlm", "zsm", "may"),
        "may": ("ms", "msa", "zlm", "zsm", "may"),
    }
    return groups.get(token, (token,))
