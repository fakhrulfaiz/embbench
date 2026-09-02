from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from uuid import UUID

import httpx

from embbench.generation.config import Settings, get_settings
from embbench.generation.models import HIGH_SCORES, KIND_BY_SCORE, Chunk, PairDraft
from embbench.generation.sts.filters import pair_reject_reason, reject_reason, split_sentences
from embbench.generation.sts.llm import HttpChatLLM, LLMClient
from embbench.generation.sts.miner import mine_low_scores
from embbench.generation.sts.mix import allocate_counts
from embbench.generation.sts.prompts import gist_prompt, high_score_prompt

logger = logging.getLogger("embbench.generation.sts")

PersistFn = Callable[[PairDraft], None]


@dataclass
class GenerateResult:
    requested: int
    written: int
    rejected: int
    by_score: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def generate_sts_pairs(
    chunks: Sequence[Chunk],
    *,
    count: int,
    language: str,
    llm: LLMClient | None = None,
    rewrite_gists: bool = True,
    rng: random.Random | None = None,
    settings: Settings | None = None,
    persist: PersistFn | None = None,
) -> tuple[list[PairDraft], GenerateResult]:
    """Build the target mix. persist(pair) is called per accepted pair (DB insert)."""
    rng = rng or random.Random()
    counts = allocate_counts(count)
    warnings: list[str] = []
    accepted: list[PairDraft] = []
    rejected = 0

    low = mine_low_scores(chunks, counts, language=language, rng=rng)
    for score in (0.0, 1.0, 2.0):
        got = sum(1 for p in low if p.score == score)
        want = counts.get(score, 0)
        if got < want:
            warnings.append(f"score {score}: mined {got}/{want} pairs")

    if rewrite_gists and low:
        client = llm or HttpChatLLM(settings or get_settings())
        rewritten, extra_reject = _rewrite_low(low, client)
        rejected += extra_reject
        low = rewritten
    else:
        kept: list[PairDraft] = []
        for pair in low:
            if pair_reject_reason(pair.sentence1, pair.sentence2):
                rejected += 1
                continue
            kept.append(pair)
        low = kept

    for pair in low:
        if persist:
            persist(pair)
        accepted.append(pair)

    seeds = list(chunks)
    rng.shuffle(seeds)
    client = llm
    for score in HIGH_SCORES:
        want = counts.get(score, 0)
        if want <= 0:
            continue
        if client is None:
            client = HttpChatLLM(settings or get_settings())
        built, extra_reject = _generate_high(
            seeds, score=score, n=want, language=language, llm=client
        )
        rejected += extra_reject
        if len(built) < want:
            warnings.append(f"score {score}: generated {len(built)}/{want} pairs")
        for pair in built:
            if persist:
                persist(pair)
            accepted.append(pair)

    by_score: dict[str, int] = {}
    for pair in accepted:
        key = f"{pair.score:g}"
        by_score[key] = by_score.get(key, 0) + 1

    result = GenerateResult(
        requested=count,
        written=len(accepted),
        rejected=rejected,
        by_score=by_score,
        warnings=warnings,
    )
    return accepted, result


def _generate_high(
    seeds: Sequence[Chunk],
    *,
    score: float,
    n: int,
    language: str,
    llm: LLMClient,
    attempts_per_seed: int = 2,
) -> tuple[list[PairDraft], int]:
    out: list[PairDraft] = []
    rejected = 0
    used: set[UUID] = set()
    for chunk in seeds:
        if len(out) >= n:
            break
        if chunk.chunk_id in used:
            continue
        pair = None
        for _ in range(attempts_per_seed):
            try:
                data = llm.complete_json(high_score_prompt(chunk.content, score))
                s1 = str(data.get("sentence1") or "").strip()
                s2 = str(data.get("sentence2") or "").strip()
            except (
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
                httpx.HTTPError,
                OSError,
            ) as exc:
                logger.warning("LLM JSON failed for chunk %s: %s", chunk.chunk_id, exc)
                rejected += 1
                continue
            reason = pair_reject_reason(s1, s2)
            if reason:
                logger.info("reject %s on %s", reason, chunk.chunk_id)
                rejected += 1
                continue
            pair = PairDraft(
                sentence1=s1,
                sentence2=s2,
                score=score,
                pair_kind=KIND_BY_SCORE[score],
                language=language or chunk.language,
                source_chunk_id=chunk.chunk_id,
            )
            break
        if pair is None:
            continue
        used.add(chunk.chunk_id)
        out.append(pair)
    return out, rejected


def _rewrite_low(
    pairs: Sequence[PairDraft], llm: LLMClient
) -> tuple[list[PairDraft], int]:
    out: list[PairDraft] = []
    rejected = 0
    for pair in pairs:
        s1 = _gist_or_fallback(pair.sentence1, llm)
        s2 = _gist_or_fallback(pair.sentence2, llm)
        reason = pair_reject_reason(s1, s2)
        if reason:
            rejected += 1
            continue
        pair.sentence1 = s1
        pair.sentence2 = s2
        out.append(pair)
    return out, rejected


def _gist_or_fallback(text: str, llm: LLMClient) -> str:
    try:
        data = llm.complete_json(gist_prompt(text))
        gist = str(data.get("gist") or data.get("sentence1") or "").strip()
        if gist and reject_reason(gist) is None:
            return gist
    except (
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        httpx.HTTPError,
        OSError,
    ) as exc:
        logger.info("gist rewrite failed, using stripped fallback: %s", exc)
    fallback = split_sentences(text, max_sentences=6)
    return fallback or text.strip()
