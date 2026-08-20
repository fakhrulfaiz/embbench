"""Shared job contracts. CLI and API both build a JobSpec and persist a JobResult."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal["Retrieval", "STS"]
JobStatus = Literal["pending", "running", "completed", "failed", "skipped"]


class JobSpec(BaseModel):
    job_id: str
    model_id: str
    languages: list[str] = Field(default_factory=lambda: ["eng", "cmn", "zsm"])
    task_types: list[TaskType] = Field(default_factory=lambda: ["Retrieval", "STS"])
    include_heavy: bool = False
    include_local: bool = True
    include_mteb: bool = True
    k_values: list[int] = Field(default_factory=lambda: [10, 30])
    profile_ops: bool = False
    overwrite: bool = False
    encode_batch_size: int | None = None
    task_names: list[str] | None = None


class TaskScore(BaseModel):
    name: str
    task_type: TaskType
    language: str
    source: Literal["mteb", "local"]
    main_score: float | None = None
    main_score_name: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    ndcg: dict[str, float] = Field(default_factory=dict)
    recall: dict[str, float] = Field(default_factory=dict)
    peak_vram_gb: float | None = None
    elapsed_s: float | None = None
    error: str | None = None


class OpsProfile(BaseModel):
    n_docs: int
    embed_dim: int
    peak_vram_gb: float | None = None
    encode_docs_per_s: float | None = None
    encode_wall_s: float | None = None
    index_build_s: float | None = None
    index_size_bytes: int | None = None
    p50_query_ms: float | None = None
    p95_query_ms: float | None = None
    p99_query_ms: float | None = None
    exact_recall_at_30: float | None = None
    qdrant_recall_at_30: float | None = None
    ann_recall_delta: float | None = None
    backend: str | None = None
    notes: list[str] = Field(default_factory=list)


class JobResult(BaseModel):
    spec: JobSpec
    status: JobStatus = "pending"
    model_hf_name: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    tasks: list[TaskScore] = Field(default_factory=list)
    ops: OpsProfile | None = None
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ManifestEntry(BaseModel):
    model_id: str
    status: JobStatus
    job_id: str
    error: str | None = None
    finished_at: datetime | None = None


class RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    entries: list[ManifestEntry] = Field(default_factory=list)
