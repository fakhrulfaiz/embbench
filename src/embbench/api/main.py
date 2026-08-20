"""Phase 2 HTTP stub. Builds the same JobSpec the CLI uses."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from embbench.core.config import bootstrap_env, get_settings
from embbench.core.schemas import JobResult, JobSpec
from embbench.evaluation.report import collect_results
from embbench.evaluation.runner import result_path

bootstrap_env()

app = FastAPI(
    title="embbench",
    version="0.1.0",
    description=(
        "Phase 2 stub. POST /jobs persists a JobSpec; execute with the CLI "
        "orchestrator so only one model occupies the GPU. Optional ?execute=true "
        "calls run_job in-process (do not use alongside another GPU run)."
    ),
)


class CreateJobRequest(BaseModel):
    model_id: str
    languages: list[str] = Field(default_factory=lambda: ["eng", "cmn", "zsm"])
    task_types: list[str] = Field(default_factory=lambda: ["Retrieval", "STS"])
    include_heavy: bool = False
    include_local: bool = True
    profile_ops: bool = False
    overwrite: bool = False
    job_id: str | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
    spec: JobSpec


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "hf_home": str(settings.hf_home)}


@app.post("/jobs", response_model=CreateJobResponse)
def create_job(body: CreateJobRequest, execute: bool = False) -> CreateJobResponse:
    from datetime import datetime, timezone

    job_id = body.job_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{body.model_id}"
    spec = JobSpec(
        job_id=job_id,
        model_id=body.model_id,
        languages=body.languages,
        task_types=body.task_types,  # type: ignore[arg-type]
        include_heavy=body.include_heavy,
        include_local=body.include_local,
        profile_ops=body.profile_ops,
        overwrite=body.overwrite,
    )
    dest = result_path(spec).parent
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "spec.json").write_text(spec.model_dump_json(indent=2))

    if execute:
        from embbench.evaluation.runner import run_job

        result = run_job(spec)
        return CreateJobResponse(job_id=spec.job_id, status=result.status, spec=spec)

    return CreateJobResponse(job_id=spec.job_id, status="pending", spec=spec)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JobResult:
    settings = get_settings()
    path = settings.results_dir / job_id / "result.json"
    if not path.exists():
        spec_path = settings.results_dir / job_id / "spec.json"
        if spec_path.exists():
            spec = JobSpec.model_validate_json(spec_path.read_text())
            return JobResult(spec=spec, status="pending")
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
    return JobResult.model_validate_json(path.read_text())


@app.get("/results")
def list_results() -> list[JobResult]:
    return collect_results()
