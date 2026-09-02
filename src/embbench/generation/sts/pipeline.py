"""STS jobs that talk to Postgres. Pure pair construction lives in generate.py."""

from __future__ import annotations

import logging
import random

from embbench.generation.config import Settings, get_settings
from embbench.generation.db import fetch_chunks, fetch_pairs, insert_export, insert_pairs
from embbench.generation.models import PairDraft
from embbench.generation.sts.export import ExportResult, export_sts
from embbench.generation.sts.generate import GenerateResult, generate_sts_pairs
from embbench.generation.sts.llm import LLMClient

logger = logging.getLogger("embbench.generation.sts")


def run_generate(
    *,
    count: int | None,
    language: str,
    profile: str | None = None,
    rewrite_gists: bool = True,
    dry_run: bool = False,
    seed: int | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> GenerateResult:
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
    logger.info("loaded %s chunks", len(chunks))

    pair_count = count if count is not None else len(chunks)
    if pair_count < 1:
        raise RuntimeError("No chunks to build STS pairs from")

    batch: list[PairDraft] = []

    def persist(pair: PairDraft) -> None:
        if dry_run:
            return
        batch.append(pair)
        if len(batch) >= 10:
            insert_pairs(batch, settings=cfg)
            batch.clear()

    rng = random.Random(seed)
    _pairs, result = generate_sts_pairs(
        chunks,
        count=pair_count,
        language=language,
        llm=llm,
        rewrite_gists=rewrite_gists,
        rng=rng,
        settings=cfg,
        persist=None if dry_run else persist,
    )
    if dry_run:
        result.written = len(_pairs)
        return result
    if batch:
        insert_pairs(batch, settings=cfg)
    return result


def run_export(
    *,
    name: str,
    language: str,
    revision: str | None = None,
    description: str | None = None,
    min_score: float = 0.0,
    max_score: float = 5.0,
    record: bool = True,
    settings: Settings | None = None,
) -> ExportResult:
    cfg = settings or get_settings()
    rows = fetch_pairs(language=language, settings=cfg)
    result = export_sts(
        rows,
        name=name,
        language=language,
        revision=revision,
        description=description,
        min_score=min_score,
        max_score=max_score,
        settings=cfg,
    )
    if record:
        insert_export(
            name=result.name,
            task_type="sts",
            language=result.language,
            revision=result.revision,
            settings=cfg,
        )
    logger.info("exported %s pairs to %s", result.pairs, result.folder)
    return result
