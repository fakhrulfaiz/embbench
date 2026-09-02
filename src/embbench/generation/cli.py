from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import typer
import uvicorn

from embbench.core.config import REPO_ROOT
from embbench.generation.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
sts_app = typer.Typer(no_args_is_help=True, add_completion=False)
retrieval_app = typer.Typer(no_args_is_help=True, add_completion=False)

LOG_FILE = REPO_ROOT / "logs" / "generate.log"


def _configure_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_embbench_generate_configured", False):
        return
    root.setLevel(logging.INFO)
    root.handlers.clear()

    log_dir = LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(console)
    root._embbench_generate_configured = True  # type: ignore[attr-defined]


@app.callback()
def _root() -> None:
    _configure_logging()


@retrieval_app.callback()
def _retrieval_root() -> None:
    _configure_logging()


@sts_app.callback()
def _sts_root() -> None:
    _configure_logging()


@app.command("init-schema")
def cmd_init_schema() -> None:
    """CREATE IF NOT EXISTS for documents/chunks/labels/dataset_exports."""
    from embbench.generation.db import init_schema

    init_schema()
    typer.echo("schema ok")


@app.command("stats")
def cmd_stats() -> None:
    from embbench.generation.db import store_stats

    stats = store_stats()
    typer.echo(
        f"chunks={stats['chunks']} pairs={stats['pairs']} "
        f"questions={stats['questions']}"
    )
    for row in stats.get("by_language") or []:
        typer.echo(f"  lang={row['language'] or '?'} chunks={row['count']}")
    for row in stats["by_score"]:
        typer.echo(f"  score={row['score']:g} kind={row['pair_kind']} n={row['count']}")


@app.command("serve")
def cmd_serve(
    host: str | None = None,
    port: int | None = None,
) -> None:
    settings = get_settings()
    uvicorn.run(
        "embbench.generation.api:app",
        host=host or settings.gensvc_host,
        port=port or settings.gensvc_port,
    )


@sts_app.command("generate")
def cmd_sts_generate(
    count: int = typer.Option(100, help="How many new pairs to add (target mix)."),
    all_chunks: bool = typer.Option(
        False,
        "--all",
        help="Target mix sized to the number of matching chunks.",
    ),
    language: str = typer.Option(
        "en",
        help="Chunk/pair language filter: en, zh, ms, or all.",
    ),
    profile: str | None = typer.Option(None, help="Override CHUNKER_PROFILE."),
    rewrite_gists: bool = typer.Option(
        True,
        help="LLM-rewrite mined 0–2 sides into short gists.",
    ),
    dry_run: bool = typer.Option(False, help="Do not write sts_pairs."),
    seed: int | None = typer.Option(None, help="RNG seed for mining."),
) -> None:
    from embbench.generation.sts.pipeline import run_generate as run_sts_generate

    result = run_sts_generate(
        count=None if all_chunks else count,
        language=language,
        profile=profile,
        rewrite_gists=rewrite_gists,
        dry_run=dry_run,
        seed=seed,
    )
    typer.echo(
        f"wrote {result.written} pairs  "
        f"(requested {result.requested}, rejected {result.rejected})"
    )
    typer.echo(f"by_score={result.by_score}")
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")


@sts_app.command("export")
def cmd_sts_export(
    name: str = typer.Option(..., help="Task id / folder name, e.g. sop-sts-v1."),
    language: str = typer.Option("eng-Latn", help="meta.yaml language."),
    revision: str | None = typer.Option(None, help="Default: UTC date."),
    description: str | None = typer.Option(None),
    min_score: float = typer.Option(0.0),
    max_score: float = typer.Option(5.0),
    no_record: bool = typer.Option(False, help="Skip dataset_exports insert."),
) -> None:
    from embbench.generation.sts.pipeline import run_export as run_sts_export

    result = run_sts_export(
        name=name,
        language=language,
        revision=revision,
        description=description,
        min_score=min_score,
        max_score=max_score,
        record=not no_record,
    )
    typer.echo(f"wrote {result.pairs} pairs to {result.folder}")
    typer.echo(f"language={result.language} revision={result.revision}")


@retrieval_app.command("generate")
def cmd_retrieval_generate(
    count: int = typer.Option(100, help="How many new questions to add. Ignored with --all."),
    all_chunks: bool = typer.Option(
        False,
        "--all",
        help="One question per remaining unlabelled chunk (this language, or all).",
    ),
    language: str = typer.Option(
        "en",
        help="Chunk language filter: en, zh, ms, or all.",
    ),
    profile: str | None = typer.Option(None, help="Override CHUNKER_PROFILE."),
    force: bool = typer.Option(
        False,
        help="Also use chunks that already have a question.",
    ),
    dry_run: bool = typer.Option(False, help="Do not write retrieval_questions."),
    seed: int | None = typer.Option(None, help="RNG seed for chunk order."),
    concurrency: int = typer.Option(
        4,
        help="Parallel vLLM requests. Keep at or below LLM_MAX_REQ (compose).",
    ),
) -> None:
    from embbench.generation.retrieval.pipeline import run_generate as run_retrieval_generate

    result = run_retrieval_generate(
        count=None if all_chunks else count,
        language=language,
        profile=profile,
        force=force,
        dry_run=dry_run,
        seed=seed,
        concurrency=concurrency,
    )
    typer.echo(
        f"wrote {result.written} questions  "
        f"(requested {result.requested}, rejected {result.rejected}, "
        f"skipped {result.skipped_labelled} that already had a question)"
    )
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")


@retrieval_app.command("export")
def cmd_retrieval_export(
    name: str = typer.Option(
        ...,
        help="Task id / folder name, e.g. sop-handbook-v1.",
    ),
    language: str = typer.Option("eng-Latn", help="meta.yaml language."),
    revision: str | None = typer.Option(None, help="Default: UTC date."),
    description: str | None = typer.Option(None),
    no_record: bool = typer.Option(False, help="Skip dataset_exports insert."),
) -> None:
    from embbench.generation.retrieval.pipeline import run_export as run_retrieval_export

    result = run_retrieval_export(
        name=name,
        language=language,
        revision=revision,
        description=description,
        record=not no_record,
    )
    typer.echo(
        f"wrote corpus={result.corpus} queries={result.queries} "
        f"qrels={result.qrels} to {result.folder}"
    )
    typer.echo(f"language={result.language} revision={result.revision}")
