"""Typer CLI. `run --all` orchestrates one subprocess per model."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Cache guard MUST run before torch / huggingface_hub / mteb.
from embbench.core.config import REPO_ROOT, bootstrap_env, get_settings

bootstrap_env()

import typer
from rich.console import Console
from rich.table import Table

from embbench.core.registry import get_model_config, load_models
from embbench.core.schemas import JobResult, JobSpec, ManifestEntry, RunManifest
from embbench.datasets.local_source import LocalSource
from embbench.datasets.mteb_source import MtebSource
from embbench.evaluation.report import render_report, write_report
from embbench.evaluation.runner import run_job
from embbench.generation.cli import retrieval_app, sts_app

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(retrieval_app, name="retrieval")
app.add_typer(sts_app, name="sts")
console = Console()
logger = logging.getLogger("embbench")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.callback()
def _root() -> None:
    _configure_logging()
    get_settings()


@app.command("list-models")
def list_models() -> None:
    table = Table(title="Models")
    table.add_column("id")
    table.add_column("hf_name")
    table.add_column("role")
    table.add_column("loader")
    table.add_column("max_seq")
    for model in load_models():
        table.add_row(model.id, model.hf_name, model.role, model.loader, str(model.max_seq_length))
    console.print(table)


@app.command("list-datasets")
def list_datasets(
    languages: str = typer.Option("eng,cmn,zsm"),
    include_heavy: bool = False,
) -> None:
    langs = [x.strip() for x in languages.split(",") if x.strip()]
    table = Table(title="Datasets")
    table.add_column("source")
    table.add_column("type")
    table.add_column("lang")
    table.add_column("name")
    table.add_column("heavy")
    for source_name, source in (("mteb", MtebSource()), ("local", LocalSource())):
        for task in source.list_tasks(langs, ["Retrieval", "STS"], include_heavy):
            table.add_row(source_name, task.task_type, task.language, task.name, str(task.heavy))
    console.print(table)
    if MtebSource().malay_sts_missing(langs, ["STS"]):
        local = LocalSource().list_tasks(["zsm"], ["STS"])
        if not local:
            console.print(
                "[yellow]No Malay STS dataset present. MTEB has none; "
                "drop a folder into data/sts/ to fill this slot.[/yellow]"
            )


@app.command("run")
def run(
    model: str = typer.Option("all", help="Model id, hf name, or 'all'."),
    languages: str = typer.Option("eng,cmn,zsm"),
    task_types: str = typer.Option("Retrieval,STS"),
    include_heavy: bool = False,
    include_local: bool = True,
    include_mteb: bool = True,
    task_names: str | None = typer.Option(
        None,
        help="Comma-separated task names. Restricts to matching MTEB and local folders.",
    ),
    profile_ops: bool = False,
    overwrite: bool = False,
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-run models that already have a completed result for this same scope. "
        "Without it, `--model all` resumes and skips finished work.",
    ),
    run_id: str | None = None,
) -> None:
    """Run one model, or all models sequentially (one subprocess each)."""
    langs = [x.strip() for x in languages.split(",") if x.strip()]
    types = [x.strip() for x in task_types.split(",") if x.strip()]  # type: ignore[var-annotated]
    names = [x.strip() for x in (task_names or "").split(",") if x.strip()] or None
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings = get_settings()
    models = load_models()
    if model != "all":
        cfg = get_model_config(model)
        models = [cfg]

    # Forcing a re-run must also bypass MTEB's own result cache, otherwise the job
    # would reload cached scores instead of actually re-encoding.
    overwrite = overwrite or force

    if model == "all" and os.environ.get("EMBBENCH_WORKER") != "1":
        _orchestrate(
            models,
            langs,
            types,
            include_heavy,
            include_local,
            include_mteb,
            names,
            profile_ops,
            overwrite,
            rid,
            skip_completed=not (force or overwrite),
        )
        return

    for cfg in models:
        spec = JobSpec(
            job_id=f"{rid}-{cfg.id}",
            model_id=cfg.id,
            languages=langs,
            task_types=types,  # type: ignore[arg-type]
            include_heavy=include_heavy,
            include_local=include_local,
            include_mteb=include_mteb,
            profile_ops=profile_ops,
            overwrite=overwrite,
            task_names=names,
        )
        console.print(f"[bold]Running[/bold] {cfg.id} ({cfg.hf_name})")
        result = run_job(spec)
        console.print(f"  status={result.status} tasks={len(result.tasks)} error={result.error}")
        if result.status == "failed":
            raise typer.Exit(code=1)

    dest = write_report(settings.results_dir / "report.md")
    console.print(f"Report: {dest}")


def _orchestrate(
    models,
    langs: list[str],
    types: list[str],
    include_heavy: bool,
    include_local: bool,
    include_mteb: bool,
    task_names: list[str] | None,
    profile_ops: bool,
    overwrite: bool,
    run_id: str,
    skip_completed: bool = True,
) -> None:
    settings = get_settings()
    manifest_path = settings.results_dir / run_id / "run-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(run_id=run_id, started_at=datetime.now(timezone.utc))

    env = os.environ.copy()
    env["EMBBENCH_WORKER"] = "1"
    python = sys.executable
    skipped = 0

    for cfg in models:
        job_id = f"{run_id}-{cfg.id}"
        wanted = JobSpec(
            job_id=job_id,
            model_id=cfg.id,
            languages=langs,
            task_types=types,  # type: ignore[arg-type]
            include_heavy=include_heavy,
            include_local=include_local,
            include_mteb=include_mteb,
            profile_ops=profile_ops,
            task_names=task_names,
        )
        done = _find_completed(cfg.id, wanted) if skip_completed else None
        if done is not None:
            skipped += 1
            console.print(
                f"[green]skip[/green] {cfg.id}: already completed "
                f"{len(done.tasks)} tasks in {done.spec.job_id}"
            )
            if profile_ops and done.ops is None:
                console.print(
                    f"  [yellow]note[/yellow] that job has no ops profile; "
                    f"rerun it alone with --profile-ops --force to measure one"
                )
            manifest.entries.append(
                ManifestEntry(
                    model_id=cfg.id,
                    status="skipped",
                    job_id=done.spec.job_id,
                    finished_at=done.finished_at,
                )
            )
            _write_manifest(manifest_path, manifest)
            continue

        cmd = [
            python,
            "-m",
            "embbench.cli",
            "run",
            "--model",
            cfg.id,
            "--languages",
            ",".join(langs),
            "--task-types",
            ",".join(types),
            "--run-id",
            run_id,
        ]
        if include_heavy:
            cmd.append("--include-heavy")
        if not include_local:
            cmd.append("--no-include-local")
        if not include_mteb:
            cmd.append("--no-include-mteb")
        if task_names:
            cmd.extend(["--task-names", ",".join(task_names)])
        if profile_ops:
            cmd.append("--profile-ops")
        if overwrite:
            cmd.append("--overwrite")

        console.print(f"[bold cyan]subprocess[/bold cyan] {' '.join(cmd)}")
        completed = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
        status = "completed" if completed.returncode == 0 else "failed"
        error = None if completed.returncode == 0 else f"exit {completed.returncode}"
        manifest.entries.append(
            ManifestEntry(
                model_id=cfg.id,
                status=status,  # type: ignore[arg-type]
                job_id=job_id,
                error=error,
                finished_at=datetime.now(timezone.utc),
            )
        )
        _write_manifest(manifest_path, manifest)
        if completed.returncode != 0:
            console.print(f"[red]fail-soft[/red] {cfg.id} exited {completed.returncode}; continuing")

    manifest.finished_at = datetime.now(timezone.utc)
    _write_manifest(manifest_path, manifest)
    if skipped:
        console.print(
            f"Reused {skipped} of {len(models)} models from earlier runs. "
            "Pass [bold]--force[/bold] to re-run everything."
        )
    dest = write_report(settings.results_dir / "report.md")
    console.print(f"Report: {dest}")


def _write_manifest(path: Path, manifest: RunManifest) -> None:
    path.write_text(manifest.model_dump_json(indent=2))


def _scope(spec: JobSpec) -> tuple:
    """What a job actually covered. Two jobs are interchangeable only if these match."""
    return (
        tuple(sorted(spec.languages)),
        tuple(sorted(spec.task_types)),
        spec.include_heavy,
        spec.include_local,
        spec.include_mteb,
        tuple(sorted(spec.task_names)) if spec.task_names else None,
    )


def _find_completed(model_id: str, wanted: JobSpec) -> JobResult | None:
    """Newest usable result for this model at this exact scope, from any earlier run.

    Scope must match exactly, so narrowing a run (say to `--languages eng`) never
    silently reuses a broader one. A job that failed, or whose every task errored,
    does not count as done and will be retried.
    """
    settings = get_settings()
    if not settings.results_dir.exists():
        return None

    best: JobResult | None = None
    for path in sorted(settings.results_dir.glob("*/result.json")):
        try:
            found = JobResult.model_validate_json(path.read_text())
        except Exception:
            continue
        if found.spec.model_id != model_id or found.status != "completed":
            continue
        if not found.tasks or all(task.error for task in found.tasks):
            continue
        if _scope(found.spec) != _scope(wanted):
            continue
        if best is None or _finished_key(found) > _finished_key(best):
            best = found
    return best


def _finished_key(result: JobResult) -> tuple[int, str]:
    return (1, result.finished_at.isoformat()) if result.finished_at else (0, "")


@app.command("report")
def report_cmd(out: Path | None = None) -> None:
    dest = write_report(out)
    console.print(render_report())
    console.print(f"Wrote {dest}")


@app.command("check-model")
def check_model(hf_name: str) -> None:
    """Report whether MTEB knows a model, and print the configs/models.yaml block to use.

    MTEB-registered models keep their own query/document prompt recipe, which
    instruction-aware models need to score honestly. Anything else falls back to
    a plain SentenceTransformer.
    """
    import mteb

    try:
        meta = mteb.get_model_meta(hf_name)
    except Exception as exc:
        console.print(f"[yellow]Not in the MTEB registry[/yellow] ({type(exc).__name__})")
        console.print("Use [bold]loader: sentence_transformers[/bold]; encode() is called directly.")
        console.print("If the repo ships custom modeling code, also set trust_remote_code: true.")
        _print_model_yaml(hf_name, loader="sentence_transformers")
        return

    loader_name = getattr(meta.loader, "__name__", str(meta.loader))
    kwargs = meta.loader_kwargs or {}
    trust = bool(kwargs.get("trust_remote_code"))

    table = Table(title=f"MTEB knows {hf_name}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("wrapper", loader_name)
    table.add_row("revision", str(meta.revision))
    table.add_row("uses instructions", str(meta.use_instructions))
    table.add_row("embedding dim", str(meta.embed_dim))
    table.add_row("parameters", f"{meta.n_parameters:,}" if meta.n_parameters else "—")
    table.add_row("max tokens", str(meta.max_tokens))
    table.add_row("trust_remote_code", str(trust))
    table.add_row("extra requirements", str(meta.extra_requirements_groups or "none"))
    console.print(table)

    if meta.use_instructions:
        console.print(
            "[green]Use loader: mteb.[/green] This model prepends its own query and document "
            "prompts; a plain encode() would score it below its real quality."
        )
    else:
        console.print(
            "[green]loader: mteb works.[/green] No instruction prompts, so "
            "sentence_transformers would score the same."
        )

    if meta.extra_requirements_groups:
        extras = ", ".join(meta.extra_requirements_groups)
        console.print(
            f"[yellow]Heads up:[/yellow] MTEB tags this with the extra [bold]{extras}[/bold]. "
            "That is usually a hosted-API SDK which local weights do not need. "
            "`load_encoder` already falls back to MTEB's SentenceTransformer wrapper instead of "
            "installing it."
        )
    if trust:
        console.print(
            "[yellow]Runs custom code from the Hub.[/yellow] Set trust_remote_code: true. "
            "If it was written for an older Transformers, expect signature errors like the "
            "create_causal_mask fix in core/encoders.py."
        )

    console.print(
        "To score a model already served by vLLM, use [bold]loader: openai_api[/bold] "
        "instead of loading weights in this process. See docs/adding-models.md."
    )

    _print_model_yaml(
        hf_name,
        loader="mteb",
        trust_remote_code=trust,
        use_instructions=bool(meta.use_instructions),
        n_parameters=meta.n_parameters,
    )


def _print_model_yaml(
    hf_name: str,
    *,
    loader: str,
    trust_remote_code: bool = False,
    use_instructions: bool = False,
    n_parameters: int | None = None,
) -> None:
    """Emit a paste-ready configs/models.yaml entry."""
    model_id = hf_name.split("/")[-1]
    batch = 32 if n_parameters and n_parameters < 200_000_000 else 16
    block = "\n".join(
        [
            f"  - id: {model_id}",
            f"    hf_name: {hf_name}",
            "    role: candidate",
            f"    loader: {loader}",
            f"    trust_remote_code: {str(trust_remote_code).lower()}",
            f"    use_instructions: {str(use_instructions).lower()}",
            "    max_seq_length: 512",
            f"    batch_size: {batch}",
            "    description: why this model is worth testing",
        ]
    )
    console.print("\nAdd to [bold]configs/models.yaml[/bold] under `models:`\n")
    console.print(block, highlight=False)
    console.print(
        f"\nThen smoke it on one task: [bold]uv run embbench run --model {model_id} "
        "--languages eng --task-types Retrieval --no-include-mteb --task-names SciFact[/bold]"
    )


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8501, help="Port to serve the dashboard on."),
    host: str = typer.Option("localhost", help="Address to bind."),
    open_browser: bool = typer.Option(
        False, help="Try to open a browser. Off by default because WSL has no default browser."
    ),
) -> None:
    """Serve the read-only Streamlit view of everything under results/."""
    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    settings = get_settings()
    console.print(f"[bold]Dashboard[/bold] http://{host}:{port} (reading {settings.results_dir})")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "false" if open_browser else "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    try:
        code = subprocess.call(command, env=os.environ.copy())
    except FileNotFoundError as exc:
        console.print(f"[red]Could not start Streamlit:[/red] {exc}")
        console.print("Run [bold]uv sync[/bold] to install it.")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("Dashboard stopped.")
        return
    raise typer.Exit(code=code)


@app.command("smoke")
def smoke() -> None:
    """Tiny SciFact run of the baseline, used to verify HF cache + VRAM."""
    spec = JobSpec(
        job_id=f"smoke-{uuid.uuid4().hex[:8]}",
        model_id="bce-embedding-base_v1",
        languages=["eng"],
        task_types=["Retrieval"],
        include_local=False,
        include_mteb=True,
        profile_ops=False,
        overwrite=True,
        task_names=["SciFact"],
    )
    # Restrict to SciFact only by temporarily using run_job on a patched spec:
    # we still go through run_job; SciFact is in the eng retrieval list.
    result = run_job(spec)
    settings = get_settings()
    console.print(f"status={result.status} hf_home={settings.hf_home}")
    if result.status != "completed":
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    app()
