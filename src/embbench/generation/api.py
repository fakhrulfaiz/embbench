from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from embbench.generation import __version__
from embbench.generation.config import get_settings
from embbench.generation.db import store_stats
from embbench.generation.retrieval.pipeline import run_export as run_retrieval_export
from embbench.generation.retrieval.pipeline import run_generate as run_retrieval_generate
from embbench.generation.sts.pipeline import run_export as run_sts_export
from embbench.generation.sts.pipeline import run_generate as run_sts_generate

app = FastAPI(
    title="embbench",
    version=__version__,
    description=(
        "STS pairs and retrieval questions from Postgres chunks. "
        "Extract and chunking are out of scope this phase."
    ),
)


class StsGenerateRequest(BaseModel):
    count: int = Field(100, ge=1, le=50_000)
    all: bool = False
    language: str = "en"
    profile: str | None = None
    rewrite_gists: bool = True
    dry_run: bool = False
    seed: int | None = None


class StsGenerateResponse(BaseModel):
    requested: int
    written: int
    rejected: int
    by_score: dict[str, int]
    warnings: list[str]


class StsExportRequest(BaseModel):
    name: str
    language: str = "eng-Latn"
    revision: str | None = None
    description: str | None = None
    min_score: float = 0.0
    max_score: float = 5.0
    record: bool = True


class StsExportResponse(BaseModel):
    name: str
    language: str
    revision: str
    pairs: int
    folder: str


class RetrievalGenerateRequest(BaseModel):
    count: int = Field(100, ge=1, le=50_000)
    all: bool = False
    language: str = "en"
    profile: str | None = None
    force: bool = False
    dry_run: bool = False
    seed: int | None = None
    concurrency: int = Field(4, ge=1, le=64)


class RetrievalGenerateResponse(BaseModel):
    requested: int
    written: int
    rejected: int
    skipped_labelled: int
    warnings: list[str]


class RetrievalExportRequest(BaseModel):
    name: str
    language: str = "eng-Latn"
    revision: str | None = None
    description: str | None = None
    record: bool = True


class RetrievalExportResponse(BaseModel):
    name: str
    language: str
    revision: str
    corpus: int
    queries: int
    qrels: int
    folder: str


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "service": "embbench",
        "export_dir": str(settings.export_dir),
    }


@app.get("/stats")
def stats() -> dict[str, Any]:
    try:
        return store_stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/sts/stats")
def sts_stats() -> dict[str, Any]:
    return stats()


@app.post("/sts/generate", response_model=StsGenerateResponse)
def sts_generate(body: StsGenerateRequest) -> StsGenerateResponse:
    try:
        result = run_sts_generate(
            count=None if body.all else body.count,
            language=body.language,
            profile=body.profile,
            rewrite_gists=body.rewrite_gists,
            dry_run=body.dry_run,
            seed=body.seed,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StsGenerateResponse(
        requested=result.requested,
        written=result.written,
        rejected=result.rejected,
        by_score=result.by_score,
        warnings=result.warnings,
    )


@app.post("/sts/export", response_model=StsExportResponse)
def sts_export(body: StsExportRequest) -> StsExportResponse:
    try:
        result = run_sts_export(
            name=body.name,
            language=body.language,
            revision=body.revision,
            description=body.description,
            min_score=body.min_score,
            max_score=body.max_score,
            record=body.record,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StsExportResponse(
        name=result.name,
        language=result.language,
        revision=result.revision,
        pairs=result.pairs,
        folder=str(result.folder),
    )


@app.post("/retrieval/generate", response_model=RetrievalGenerateResponse)
def retrieval_generate(body: RetrievalGenerateRequest) -> RetrievalGenerateResponse:
    try:
        result = run_retrieval_generate(
            count=None if body.all else body.count,
            language=body.language,
            profile=body.profile,
            force=body.force,
            dry_run=body.dry_run,
            seed=body.seed,
            concurrency=body.concurrency,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RetrievalGenerateResponse(
        requested=result.requested,
        written=result.written,
        rejected=result.rejected,
        skipped_labelled=result.skipped_labelled,
        warnings=result.warnings,
    )


@app.post("/retrieval/export", response_model=RetrievalExportResponse)
def retrieval_export(body: RetrievalExportRequest) -> RetrievalExportResponse:
    try:
        result = run_retrieval_export(
            name=body.name,
            language=body.language,
            revision=body.revision,
            description=body.description,
            record=body.record,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RetrievalExportResponse(
        name=result.name,
        language=result.language,
        revision=result.revision,
        corpus=result.corpus,
        queries=result.queries,
        qrels=result.qrels,
        folder=str(result.folder),
    )
