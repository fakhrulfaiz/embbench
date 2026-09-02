"""Scores 0–2: two passages. TF-IDF is allowed only in this band."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from uuid import UUID

from embbench.generation.models import KIND_BY_SCORE, Chunk, PairDraft
from embbench.generation.sts.filters import split_sentences, strip_headers

TOKEN_RE = re.compile(r"[A-Za-z0-9\u4E00-\u9FFF]+")
WEAK_COSINE = (0.08, 0.38)
# Consecutive 500/50 windows share a ~50-token overlap; skip immediate neighbours.
MIN_INDEX_GAP = 2


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(strip_headers(text))]


def tfidf_vectors(docs: Sequence[str]) -> list[dict[str, float]]:
    tokenized = [tokenize(d) for d in docs]
    df: Counter[str] = Counter()
    for toks in tokenized:
        df.update(set(toks))
    n = len(tokenized) or 1
    idf = {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}
    vectors: list[dict[str, float]] = []
    for toks in tokenized:
        tf = Counter(toks)
        length = sum(tf.values()) or 1
        vec = {term: (count / length) * idf.get(term, 0.0) for term, count in tf.items()}
        vectors.append(vec)
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def group_by_doc(chunks: Sequence[Chunk]) -> dict[UUID, list[Chunk]]:
    grouped: dict[UUID, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id].append(chunk)
    for doc_id in grouped:
        grouped[doc_id].sort(
            key=lambda c: (c.created_at is None, c.created_at, str(c.chunk_id))
        )
    return dict(grouped)


def _side_text(chunk: Chunk) -> str:
    gist = split_sentences(chunk.content, max_sentences=6)
    return gist or strip_headers(chunk.content)


def _draft(left: Chunk, right: Chunk, score: float, language: str) -> PairDraft:
    return PairDraft(
        sentence1=_side_text(left),
        sentence2=_side_text(right),
        score=score,
        pair_kind=KIND_BY_SCORE[score],
        language=language or left.language,
        source_chunk_id=left.chunk_id,
        source_chunk_id_2=right.chunk_id,
    )


def mine_score_2(
    chunks: Sequence[Chunk],
    n: int,
    *,
    language: str,
    rng: random.Random,
) -> list[PairDraft]:
    """Same doc, skip the 50-token overlap partner (adjacent windows)."""
    by_doc = group_by_doc(chunks)
    eligible = [chs for chs in by_doc.values() if len(chs) >= MIN_INDEX_GAP + 1]
    if not eligible or n <= 0:
        return []
    out: list[PairDraft] = []
    seen: set[tuple[UUID, UUID]] = set()
    attempts = n * 40
    while len(out) < n and attempts > 0:
        attempts -= 1
        chs = rng.choice(eligible)
        i = rng.randrange(len(chs))
        j = rng.randrange(len(chs))
        if abs(i - j) < MIN_INDEX_GAP:
            continue
        a, b = (chs[i], chs[j]) if i < j else (chs[j], chs[i])
        key = (a.chunk_id, b.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(_draft(a, b, 2.0, language))
    return out


def mine_score_1(
    chunks: Sequence[Chunk],
    n: int,
    *,
    language: str,
    rng: random.Random,
) -> list[PairDraft]:
    """Same doc, weak TF-IDF overlap on header-stripped text."""
    by_doc = group_by_doc(chunks)
    lo, hi = WEAK_COSINE
    out: list[PairDraft] = []
    seen: set[tuple[UUID, UUID]] = set()
    docs = [chs for chs in by_doc.values() if len(chs) >= 2]
    rng.shuffle(docs)
    for chs in docs:
        if len(out) >= n:
            break
        vecs = tfidf_vectors([c.content for c in chs])
        in_band: list[tuple[float, int, int]] = []
        rest: list[tuple[float, int, int]] = []
        for i in range(len(chs)):
            for j in range(i + 1, len(chs)):
                sim = cosine(vecs[i], vecs[j])
                item = (sim, i, j)
                if lo <= sim <= hi:
                    in_band.append(item)
                elif abs(i - j) >= MIN_INDEX_GAP or len(chs) == 2:
                    rest.append(item)
        rng.shuffle(in_band)
        rest.sort(key=lambda row: row[0])
        for _, i, j in in_band + rest:
            if len(out) >= n:
                break
            a, b = chs[i], chs[j]
            key = (a.chunk_id, b.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(_draft(a, b, 1.0, language))
    return out


def mine_score_0(
    chunks: Sequence[Chunk],
    n: int,
    *,
    language: str,
    rng: random.Random,
) -> list[PairDraft]:
    """Different doc_id (or random chunks if only one document exists)."""
    if n <= 0 or len(chunks) < 2:
        return []
    by_doc = group_by_doc(chunks)
    doc_ids = list(by_doc)
    out: list[PairDraft] = []
    seen: set[tuple[UUID, UUID]] = set()
    attempts = n * 40
    multi = len(doc_ids) >= 2
    while len(out) < n and attempts > 0:
        attempts -= 1
        if multi:
            d1, d2 = rng.sample(doc_ids, 2)
            a = rng.choice(by_doc[d1])
            b = rng.choice(by_doc[d2])
        else:
            a, b = rng.sample(list(chunks), 2)
        if a.chunk_id == b.chunk_id:
            continue
        key = (min(a.chunk_id, b.chunk_id), max(a.chunk_id, b.chunk_id))
        if key in seen:
            continue
        seen.add(key)
        out.append(_draft(a, b, 0.0, language))
    return out


def mine_low_scores(
    chunks: Sequence[Chunk],
    counts: dict[float, int],
    *,
    language: str,
    rng: random.Random | None = None,
) -> list[PairDraft]:
    rng = rng or random.Random()
    pairs: list[PairDraft] = []
    pairs.extend(mine_score_0(chunks, counts.get(0.0, 0), language=language, rng=rng))
    pairs.extend(mine_score_1(chunks, counts.get(1.0, 0), language=language, rng=rng))
    pairs.extend(mine_score_2(chunks, counts.get(2.0, 0), language=language, rng=rng))
    return pairs
