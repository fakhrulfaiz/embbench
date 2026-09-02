from __future__ import annotations

import json
import logging
import random
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from uuid import UUID

import httpx

from embbench.generation.config import Settings, get_settings
from embbench.generation.models import Chunk, QuestionDraft, is_all_languages
from embbench.generation.retrieval.prompts import question_prompt
from embbench.generation.sts.filters import question_reject_reason, strip_headers
from embbench.generation.sts.llm import HttpChatLLM, LLMClient, parse_json_object

logger = logging.getLogger("embbench.generation.retrieval")

PersistFn = Callable[[QuestionDraft], None]


def _progress(done: int, total: int) -> None:
    if not sys.stderr.isatty():
        return
    sys.stderr.write(f"\r  {done}/{total} questions")
    sys.stderr.flush()


def _progress_done() -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()


@dataclass
class QuestionGenerateResult:
    requested: int
    written: int
    rejected: int
    skipped_labelled: int = 0
    warnings: list[str] = field(default_factory=list)


def generate_questions(
    chunks: Sequence[Chunk],
    *,
    count: int | None,
    language: str,
    llm: LLMClient | None = None,
    labelled: set[UUID] | None = None,
    force: bool = False,
    rng: random.Random | None = None,
    settings: Settings | None = None,
    persist: PersistFn | None = None,
    attempts_per_seed: int = 2,
    concurrency: int = 1,
) -> tuple[list[QuestionDraft], QuestionGenerateResult]:
    rng = rng or random.Random()
    labelled = labelled or set()
    seeds, skipped = _select_seeds(chunks, labelled=labelled, force=force, rng=rng)
    target = len(seeds) if count is None else count

    client = llm or HttpChatLLM(settings or get_settings())
    workers = max(1, concurrency)
    accepted: list[QuestionDraft] = []
    rejected = 0
    warnings: list[str] = []
    lock = threading.Lock()

    def consider(question: QuestionDraft | None, n_rejected: int) -> None:
        nonlocal rejected
        with lock:
            rejected += n_rejected
            if question is None or len(accepted) >= target:
                return
            if persist:
                persist(question)
            accepted.append(question)
            _progress(len(accepted), target)

    _progress(0, target)
    idx = 0
    if workers == 1:
        while len(accepted) < target and idx < len(seeds):
            question, n_rejected = _question_from_chunk(
                client, seeds[idx], language=language, attempts=attempts_per_seed
            )
            consider(question, n_rejected)
            idx += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while len(accepted) < target and idx < len(seeds):
                with lock:
                    need = target - len(accepted)
                take = min(workers, need, len(seeds) - idx)
                batch = seeds[idx : idx + take]
                idx += take
                futures = [
                    pool.submit(
                        _question_from_chunk,
                        client,
                        chunk,
                        language,
                        attempts_per_seed,
                    )
                    for chunk in batch
                ]
                for fut in as_completed(futures):
                    consider(*fut.result())

    _progress_done()
    if len(accepted) < target:
        warnings.append(f"generated {len(accepted)}/{target} questions")

    result = QuestionGenerateResult(
        requested=target,
        written=len(accepted),
        rejected=rejected,
        skipped_labelled=skipped,
        warnings=warnings,
    )
    return accepted, result


def _question_from_chunk(
    client: LLMClient,
    chunk: Chunk,
    language: str,
    attempts: int,
) -> tuple[QuestionDraft | None, int]:
    seed_text = strip_headers(chunk.content) or chunk.content
    rejected = 0
    for _ in range(attempts):
        try:
            text = _ask_question(client, seed_text)
        except (
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            httpx.HTTPError,
            OSError,
        ) as exc:
            logger.warning("LLM failed for chunk %s: %s", chunk.chunk_id, exc)
            rejected += 1
            continue
        reason = question_reject_reason(text)
        if reason:
            logger.info("reject %s on %s", reason, chunk.chunk_id)
            rejected += 1
            continue
        logger.info("question ok chunk=%s", chunk.chunk_id)
        return (
            QuestionDraft(
                question_text=text,
                gold_chunk_id=chunk.chunk_id,
                language=_question_language(language, chunk),
            ),
            rejected,
        )
    return None, rejected


def _question_language(requested: str, chunk: Chunk) -> str:
    if is_all_languages(requested):
        return chunk.language or requested
    return requested or chunk.language


def _select_seeds(
    chunks: Sequence[Chunk],
    *,
    labelled: set[UUID],
    force: bool,
    rng: random.Random,
) -> tuple[list[Chunk], int]:
    """Round-robin by document. No wrap/reuse of a used chunk."""
    if force:
        pool = list(chunks)
        skipped = 0
    else:
        pool = [c for c in chunks if c.chunk_id not in labelled]
        skipped = len(chunks) - len(pool)

    by_doc: dict[UUID, list[Chunk]] = defaultdict(list)
    for chunk in pool:
        by_doc[chunk.doc_id].append(chunk)

    docs = list(by_doc.keys())
    rng.shuffle(docs)
    for doc_id in docs:
        rng.shuffle(by_doc[doc_id])

    seeds: list[Chunk] = []
    if not docs:
        return seeds, skipped

    cursor = dict.fromkeys(docs, 0)
    while True:
        added = False
        for doc_id in docs:
            available = by_doc[doc_id]
            idx = cursor[doc_id]
            if idx < len(available):
                seeds.append(available[idx])
                cursor[doc_id] = idx + 1
                added = True
        if not added:
            break
    return seeds, skipped


def _ask_question(client: LLMClient, seed: str) -> str:
    prompt = question_prompt(seed)
    complete = getattr(client, "complete", None)
    if callable(complete):
        return extract_question(complete(prompt))
    data = client.complete_json(prompt)
    raw = data.get("question") or data.get("question_text") or ""
    return extract_question(str(raw))


def extract_question(raw: str) -> str:
    """Accept plain question text or a JSON `{"question": "..."}` blob."""
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        obj = parse_json_object(text)
        nested = obj.get("question") or obj.get("question_text") or ""
        if nested:
            text = str(nested).strip()
    except (ValueError, json.JSONDecodeError):
        pass
    text = text.strip().strip('"').strip("'").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    for line in lines:
        cleaned = line.strip().strip('"').strip("'")
        if cleaned.endswith("?") or cleaned[:1].isupper():
            return cleaned
    return lines[0]
