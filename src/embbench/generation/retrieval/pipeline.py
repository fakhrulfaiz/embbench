"""Retrieval jobs that talk to Postgres. Pure construction lives in generate.py."""

from __future__ import annotations

import logging
import random

from embbench.generation.config import Settings, get_settings
from embbench.generation.db import (
    fetch_chunks,
    fetch_labelled_chunk_ids,
    fetch_questions,
    insert_export,
    insert_questions,
)
from embbench.generation.models import QuestionDraft
from embbench.generation.retrieval.export import RetrievalExportResult, export_retrieval
from embbench.generation.retrieval.generate import QuestionGenerateResult, generate_questions
from embbench.generation.sts.llm import LLMClient

logger = logging.getLogger("embbench.generation.retrieval")


def run_generate(
    *,
    count: int | None,
    language: str,
    profile: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    seed: int | None = None,
    concurrency: int = 1,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> QuestionGenerateResult:
    cfg = settings or get_settings()
    chunks = fetch_chunks(
        profile=profile or cfg.chunker_profile,
        language=language,
        settings=cfg,
    )
    if not chunks:
        raise RuntimeError(
            f"No chunks for profile={profile or cfg.chunker_profile} language={language}"
        )
    labelled = set() if force else fetch_labelled_chunk_ids(settings=cfg)
    already = 0 if force else sum(1 for c in chunks if c.chunk_id in labelled)
    logger.info(
        "loaded %s chunks, %s already have a question",
        len(chunks),
        already,
    )

    batch: list[QuestionDraft] = []

    def persist(question: QuestionDraft) -> None:
        batch.append(question)
        if len(batch) >= 10:
            insert_questions(batch, settings=cfg)
            batch.clear()

    _questions, result = generate_questions(
        chunks,
        count=count,
        language=language,
        llm=llm,
        labelled=labelled,
        force=force,
        rng=random.Random(seed),
        settings=cfg,
        persist=None if dry_run else persist,
        concurrency=concurrency,
    )
    if dry_run:
        result.written = len(_questions)
        return result
    if batch:
        insert_questions(batch, settings=cfg)
    return result


def run_export(
    *,
    name: str,
    language: str,
    revision: str | None = None,
    description: str | None = None,
    record: bool = True,
    settings: Settings | None = None,
) -> RetrievalExportResult:
    cfg = settings or get_settings()
    chunks = fetch_chunks(
        profile=cfg.chunker_profile,
        language=language,
        settings=cfg,
    )
    questions = fetch_questions(language=language, settings=cfg)
    result = export_retrieval(
        chunks,
        questions,
        name=name,
        language=language,
        revision=revision,
        description=description,
        settings=cfg,
    )
    if record:
        insert_export(
            name=result.name,
            task_type="retrieval",
            language=result.language,
            revision=result.revision,
            settings=cfg,
        )
    logger.info(
        "exported corpus=%s queries=%s qrels=%s to %s",
        result.corpus,
        result.queries,
        result.qrels,
        result.folder,
    )
    return result
