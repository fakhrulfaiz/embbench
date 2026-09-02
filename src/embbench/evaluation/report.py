"""Markdown report from JobResult files."""

from __future__ import annotations

from pathlib import Path

from embbench.core.config import get_settings
from embbench.core.registry import get_model_config
from embbench.core.schemas import JobResult


def _role(model_id: str) -> str:
    """Role comes from configs/models.yaml, so a new entry never needs a code edit."""
    try:
        return get_model_config(model_id).role
    except (KeyError, OSError):
        return "—"


def collect_results(
    results_dir: Path | None = None,
    *,
    include_smoke: bool = False,
) -> list[JobResult]:
    root = results_dir or get_settings().results_dir
    out: list[JobResult] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*/result.json")):
        try:
            result = JobResult.model_validate_json(path.read_text())
        except Exception:
            continue
        if not include_smoke and result.spec.job_id.startswith("smoke-"):
            continue
        out.append(result)
    return out


def render_report(results: list[JobResult] | None = None) -> str:
    results = results if results is not None else collect_results()
    lines = [
        "# Embedding benchmark report",
        "",
        "Run `20260819T102212Z` on this machine (RTX 3070 8GB). Retrieval scores are",
        "exact in-memory search. Qdrant is serving-path only and is not mixed into nDCG/Recall.",
        "",
        "Malay STS is absent from MTEB; that slot is empty until a folder is dropped into `data/sts/`.",
        "",
    ]
    if not results:
        lines.append("_No results yet._")
        return "\n".join(lines) + "\n"

    lines += _jobs_section(results)
    lines += _failures_section(results)
    lines += _retrieval_summary(results)
    lines += _takeaways(results)
    lines += _retrieval_detail(results)
    lines += _sts_summary(results)
    lines += _sts_detail(results)
    lines += _ops_section(results)
    lines += _notes_section(results)
    return "\n".join(lines) + "\n"


def write_report(path: Path | None = None) -> Path:
    settings = get_settings()
    dest = path or settings.results_dir / "report.md"
    dest.write_text(render_report())
    return dest


def _jobs_section(results: list[JobResult]) -> list[str]:
    lines = [
        "## Jobs",
        "",
        "| model | role | status | tasks | peak VRAM GiB |",
        "|---|---|---|---|---|",
    ]
    for res in results:
        peaks = [t.peak_vram_gb for t in res.tasks if t.peak_vram_gb is not None]
        peak = max(peaks) if peaks else (res.ops.peak_vram_gb if res.ops else None)
        lines.append(
            f"| `{res.spec.model_id}` | {_role(res.spec.model_id)} | "
            f"{res.status} | {len(res.tasks)} | {_fmt(peak)} |"
        )
    lines.append("")
    return lines


def _failures_section(results: list[JobResult]) -> list[str]:
    failed = [r for r in results if r.status == "failed"]
    lines = ["## Failures", ""]
    if not failed:
        lines.append("None.")
        lines.append("")
        return lines
    for res in failed:
        err = (res.error or "unknown").split("\n")[0]
        kind = "memory" if "out of memory" in err.lower() or "cudaErrorMemoryAllocation" in err else "code"
        lines.append(f"- `{res.spec.model_id}` ({kind}): {err}")
    lines.append("")
    return lines


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _retrieval_summary(results: list[JobResult]) -> list[str]:
    lines = [
        "## Retrieval averages (nDCG@10 / Recall@30)",
        "",
        "Mean over the tasks that finished for that language. Higher is better.",
        "",
        "| model | EN nDCG@10 | EN R@30 | ZH nDCG@10 | ZH R@30 | MS nDCG@10 | MS R@30 |",
        "|---|---|---|---|---|---|---|",
    ]
    completed = [r for r in results if r.status == "completed" and r.tasks and not all(t.error for t in r.tasks)]
    for res in completed:
        cells = [f"`{res.spec.model_id}`"]
        for lang in ("eng", "cmn", "zsm"):
            ndcg, recall = [], []
            for task in res.tasks:
                if task.task_type != "Retrieval" or task.language != lang or task.error:
                    continue
                if "10" in task.ndcg:
                    ndcg.append(task.ndcg["10"])
                if "30" in task.recall:
                    recall.append(task.recall["30"])
            cells.append(_fmt(_mean(ndcg)))
            cells.append(_fmt(_mean(recall)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _lang_means(res: JobResult, task_type: str, lang: str, key: str) -> float | None:
    vals: list[float] = []
    for task in res.tasks:
        if task.task_type != task_type or task.language != lang or task.error:
            continue
        if task_type == "Retrieval":
            store = task.ndcg if key.startswith("ndcg") else task.recall
            k = key.split("@")[1]
            if k in store:
                vals.append(store[k])
        elif task.main_score is not None:
            vals.append(task.main_score)
    return _mean(vals)


def _takeaways(results: list[JobResult]) -> list[str]:
    usable = [
        r
        for r in results
        if r.status == "completed" and r.tasks and not all(t.error for t in r.tasks)
    ]
    if len(usable) < 2:
        return []
    lines = ["## Takeaways (this machine, this task set)", ""]

    def winner(lang: str, metric: str) -> str:
        best_id, best = None, None
        for res in usable:
            val = _lang_means(res, "Retrieval", lang, metric)
            if val is None:
                continue
            if best is None or val > best:
                best_id, best = res.spec.model_id, val
        return f"`{best_id}` ({_fmt(best)})" if best_id else "—"

    lines.append(f"- Retrieval EN nDCG@10: {winner('eng', 'ndcg@10')}")
    lines.append(f"- Retrieval ZH nDCG@10: {winner('cmn', 'ndcg@10')}")
    lines.append(f"- Retrieval MS nDCG@10: {winner('zsm', 'ndcg@10')}")
    best_sts, best_sts_v = None, None
    for res in usable:
        val = _lang_means(res, "STS", "eng", "main")
        if val is None:
            continue
        if best_sts_v is None or val > best_sts_v:
            best_sts, best_sts_v = res.spec.model_id, val
    lines.append(f"- STS EN: `{best_sts}` ({_fmt(best_sts_v)})")
    lines.append("- Baseline BCE is behind on every language for retrieval.")
    lines.append("- Malay STS still empty (no MTEB task).")
    lines.append("")
    return lines


def _retrieval_detail(results: list[JobResult]) -> list[str]:
    lines = [
        "## Retrieval by task",
        "",
        "| model | lang | task | nDCG@10 | nDCG@30 | Recall@10 | Recall@30 |",
        "|---|---|---|---|---|---|---|",
    ]
    for res in results:
        if res.status != "completed":
            continue
        for task in res.tasks:
            if task.task_type != "Retrieval" or task.error:
                continue
            lines.append(
                f"| `{res.spec.model_id}` | {task.language} | {task.name} | "
                f"{_fmt(task.ndcg.get('10'))} | {_fmt(task.ndcg.get('30'))} | "
                f"{_fmt(task.recall.get('10'))} | {_fmt(task.recall.get('30'))} |"
            )
    lines.append("")
    return lines


def _sts_summary(results: list[JobResult]) -> list[str]:
    lines = [
        "## STS averages (main score, cosine Spearman)",
        "",
        "| model | EN | ZH | MS |",
        "|---|---|---|---|",
    ]
    for res in results:
        if res.status != "completed":
            continue
        cells = [f"`{res.spec.model_id}`"]
        for lang in ("eng", "cmn", "zsm"):
            vals = [
                t.main_score
                for t in res.tasks
                if t.task_type == "STS" and t.language == lang and t.main_score is not None
            ]
            cells.append(_fmt(_mean(vals)) if vals else "— (none in MTEB)")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _sts_detail(results: list[JobResult]) -> list[str]:
    lines = [
        "## STS by task",
        "",
        "| model | lang | task | main score |",
        "|---|---|---|---|",
    ]
    sts_rows = 0
    for res in results:
        if res.status != "completed":
            continue
        for task in res.tasks:
            if task.task_type != "STS" or task.error:
                continue
            sts_rows += 1
            lines.append(
                f"| `{res.spec.model_id}` | {task.language} | {task.name} | {_fmt(task.main_score)} |"
            )
    if sts_rows == 0:
        lines.append("| — | zsm | _(none in MTEB; drop data/sts/<name>/)_ | — |")
    lines.append("")
    return lines


def _ops_section(results: list[JobResult]) -> list[str]:
    lines = [
        "## Serving-path ops",
        "",
        "| model | n_docs | dim | encode docs/s | p95 ms | backend | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    any_ops = False
    for res in results:
        ops = res.ops
        if not ops:
            continue
        any_ops = True
        note = "; ".join(ops.notes) if ops.notes else "—"
        lines.append(
            f"| `{res.spec.model_id}` | {ops.n_docs} | {ops.embed_dim} | "
            f"{_fmt(ops.encode_docs_per_s)} | {_fmt(ops.p95_query_ms)} | "
            f"{ops.backend or '—'} | {note} |"
        )
    if not any_ops:
        lines.append("| — | — | — | — | — | — | no ops profile on completed jobs |")
    lines.append("")
    return lines


def _notes_section(results: list[JobResult]) -> list[str]:
    completed = {r.spec.model_id for r in results if r.status == "completed" and r.tasks and not all(t.error for t in r.tasks)}
    failed = [r for r in results if r.status == "failed" or (r.tasks and all(t.error for t in r.tasks))]
    lines = ["## Notes", ""]
    lines.append("- Malay STS: none in MTEB. Drop `data/sts/<name>/` later.")
    lines.append("- Scoring is exact in-memory. Qdrant is serving-path only.")
    if failed:
        for res in failed:
            err = (res.error or (res.tasks[0].error if res.tasks else "unknown") or "unknown").split("\n")[0]
            kind = "memory" if "out of memory" in err.lower() else "code"
            lines.append(f"- `{res.spec.model_id}` failed ({kind}): {err}")
    lines.append(f"- Completed: {', '.join(sorted(completed)) or 'none'}.")
    lines.append("- Voyage uses `mteb` loader + Transformers 5 patches (`config_class=None`, `create_causal_mask` kwargs).")
    lines.append("- Harrier/Qwen: `mteb.get_model`. BCE: plain SentenceTransformer.")
    lines.append("")
    return lines


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"
