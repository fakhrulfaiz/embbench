"""Single execution path: run_job(JobSpec) -> JobResult."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embbench.core.config import get_settings
from embbench.core.encoders import load_encoder
from embbench.core.registry import get_model_config
from embbench.core.schemas import JobResult, JobSpec, TaskScore
from embbench.datasets.local_source import LocalSource
from embbench.datasets.mteb_source import MtebSource
from embbench.evaluation.metrics import (
    depth_ok,
    metrics_from_prediction_file,
    split_ndcg_recall,
    _extract_run,
)

logger = logging.getLogger("embbench")


def job_dir(spec: JobSpec) -> Path:
    settings = get_settings()
    path = settings.results_dir / spec.job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(spec: JobSpec) -> Path:
    return job_dir(spec) / "result.json"


def run_job(spec: JobSpec) -> JobResult:
    """Evaluate one model. Caller must not have another model on the GPU."""
    started = datetime.now(timezone.utc)
    model_cfg = get_model_config(spec.model_id)
    result = JobResult(
        spec=spec,
        status="running",
        model_hf_name=model_cfg.hf_name,
        started_at=started,
    )
    out_dir = job_dir(spec)
    (out_dir / "manifest.json").write_text(spec.model_dump_json(indent=2))

    existing = result_path(spec)
    if existing.exists() and not spec.overwrite:
        loaded = JobResult.model_validate_json(existing.read_text())
        if loaded.status == "completed":
            logger.info("Skipping completed job %s", spec.job_id)
            return loaded

    tasks = _collect_tasks(spec)
    if spec.include_mteb:
        mteb_source = MtebSource()
        if mteb_source.malay_sts_missing(spec.languages, list(spec.task_types)):
            local_sts = [t for t in tasks if t.source == "local" and t.task_type == "STS" and t.language == "zsm"]
            if not local_sts:
                logger.warning(
                    "No Malay STS dataset present. MTEB has none; drop a folder into data/sts/ to fill this slot."
                )

    try:
        model = load_encoder(model_cfg)
    except Exception as exc:
        result.status = "failed"
        result.error = f"Failed to load {model_cfg.hf_name}: {exc}"
        result.finished_at = datetime.now(timezone.utc)
        _write_result(result)
        return result

    batch_size = spec.encode_batch_size or model_cfg.batch_size
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    import mteb

    cache = mteb.ResultCache(cache_path=str(get_settings().mteb_cache))
    overwrite = "always" if spec.overwrite else "only-missing"

    for resolved in tasks:
        score = TaskScore(
            name=resolved.name,
            task_type=resolved.task_type,
            language=resolved.language,
            source=resolved.source,
        )
        _reset_peak()
        t0 = time.perf_counter()
        try:
            source = LocalSource() if resolved.source == "local" else MtebSource()
            mteb_task = source.load_mteb_task(resolved)
            if resolved.task_type == "Retrieval" and hasattr(mteb_task, "k_values"):
                ks = {int(k) for k in mteb_task.k_values}
                ks.update(int(k) for k in spec.k_values)
                mteb_task.k_values = tuple(sorted(ks))
                mteb_task._top_k = max(mteb_task.k_values)
            eval_result = mteb.evaluate(
                model,
                tasks=[mteb_task],
                cache=cache,
                overwrite_strategy=overwrite,
                prediction_folder=str(pred_dir),
                encode_kwargs={"batch_size": batch_size, "show_progress_bar": True},
            )
            score.elapsed_s = time.perf_counter() - t0
            score.peak_vram_gb = _peak_vram_gb()
            _fill_scores(score, eval_result, pred_dir, spec.k_values, resolved.task_type)
        except Exception as exc:
            logger.exception("Task %s failed", resolved.name)
            score.error = str(exc)
            score.elapsed_s = time.perf_counter() - t0
            score.peak_vram_gb = _peak_vram_gb()
        result.tasks.append(score)
        _write_result(result)
        _empty_cache()

    if spec.profile_ops and any(not t.error for t in result.tasks):
        try:
            from embbench.ops.profile import profile_ops

            corpus = _ops_corpus(tasks)
            result.ops = profile_ops(
                model,
                corpus_texts=corpus,
                batch_size=batch_size,
                k=max(spec.k_values),
            )
        except Exception as exc:
            logger.exception("Ops profile failed")
            result.extra["ops_error"] = str(exc)

    if result.tasks and all(t.error for t in result.tasks):
        result.status = "failed"
        result.error = result.error or result.tasks[0].error
    elif result.error:
        result.status = "failed"
    else:
        result.status = "completed"
    result.finished_at = datetime.now(timezone.utc)
    _write_result(result)
    _empty_cache()
    del model
    _empty_cache()
    return result


def _collect_tasks(spec: JobSpec):
    tasks = []
    if spec.include_mteb:
        tasks.extend(
            MtebSource().list_tasks(spec.languages, list(spec.task_types), spec.include_heavy)
        )
    if spec.include_local:
        tasks.extend(
            LocalSource().list_tasks(spec.languages, list(spec.task_types), spec.include_heavy)
        )
    if spec.task_names:
        allow = set(spec.task_names)
        tasks = [t for t in tasks if t.name in allow]
    return tasks


def _fill_scores(
    score: TaskScore,
    eval_result: Any,
    pred_dir: Path,
    k_values: list[int],
    task_type: str,
) -> None:
    dumped = _result_as_dict(eval_result)
    if dumped:
        main = dumped.get("main_score") or dumped.get("score")
        if main is None:
            try:
                first = eval_result[0]
                main = first.get_score()
            except Exception:
                main = None
        if main is not None:
            score.main_score = float(main)
        score.main_score_name = dumped.get("main_score_name") or dumped.get("main_metric") or "main_score"
        scores = dumped.get("scores") or dumped.get("metrics") or {}
        if isinstance(scores, dict):
            flat = _flatten_scores(scores)
            keep = {
                k: v
                for k, v in flat.items()
                if k.startswith(("ndcg_at_", "recall_at_", "main_score", "cosine", "spearman", "pearson"))
                and not k.endswith(("_max", "_std", "_diff1"))
            }
            score.scores.update(keep)
            for k in k_values:
                ndcg = flat.get(f"ndcg_at_{k}") or flat.get(f"ndcg@{k}")
                rec = flat.get(f"recall_at_{k}") or flat.get(f"recall@{k}")
                if ndcg is not None:
                    score.ndcg[str(k)] = float(ndcg)
                if rec is not None:
                    score.recall[str(k)] = float(rec)

    pred_file = _find_prediction(pred_dir, score.name)
    if pred_file and task_type == "Retrieval":
        extra = metrics_from_prediction_file(pred_file, k_values=k_values)
        ndcg, recall = split_ndcg_recall(extra)
        score.ndcg.update(ndcg)
        score.recall.update(recall)
        score.scores.update(extra)
        payload = json.loads(pred_file.read_text())
        run, _ = _extract_run(payload)
        missing = [k for k in k_values if not depth_ok(run, k)]
        if missing:
            score.scores["depth_warning"] = float(min(missing))
            logger.warning(
                "Prediction dump for %s has depth < %s; @k metrics may be underestimates",
                score.name,
                missing,
            )


def _result_as_dict(eval_result: Any) -> dict[str, Any]:
    if eval_result is None:
        return {}
    if hasattr(eval_result, "__getitem__"):
        try:
            first = eval_result[0]
        except Exception:
            first = eval_result
    else:
        first = eval_result
    if hasattr(first, "to_dict"):
        return first.to_dict()
    if hasattr(first, "model_dump"):
        return first.model_dump()
    if isinstance(first, dict):
        return first
    scores = getattr(first, "scores", None)
    if scores is not None:
        return {"scores": scores, "main_score": getattr(first, "get_score", lambda: None)()}
    if hasattr(first, "get_score"):
        try:
            return {"main_score": float(first.get_score())}
        except Exception:
            return {}
    return {}


def _flatten_scores(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = str(key) if not prefix else f"{prefix}_{key}"
            out.update(_flatten_scores(value, name))
        return out
    if isinstance(obj, list):
        for item in obj:
            out.update(_flatten_scores(item, prefix))
        return out
    try:
        val = float(obj)
    except (TypeError, ValueError):
        return out
    if prefix:
        out[prefix] = val
        out[str(prefix.split("_")[-1] if prefix.split("_")[-1] not in {"10", "20", "30", "100", "1000"} else prefix)] = val
        # Always store the metric leaf: ndcg_at_10, recall_at_30, main_score, cosine_spearman
        parts = prefix.split("_")
        for i, part in enumerate(parts):
            if part in {
                "ndcg",
                "recall",
                "map",
                "precision",
                "mrr",
                "main",
                "cosine",
                "spearman",
                "pearson",
            }:
                out["_".join(parts[i:])] = val
                break
    return out


def _find_prediction(pred_dir: Path, task_name: str) -> Path | None:
    candidates = list(pred_dir.glob(f"*{task_name}*"))
    if candidates:
        return candidates[0]
    jsons = list(pred_dir.glob("*.json"))
    return jsons[-1] if jsons else None


def _ops_corpus(tasks) -> list[str]:
    """Borrow texts from the first local/public retrieval task; no chunking."""
    texts: list[str] = []
    for resolved in tasks:
        if resolved.task_type != "Retrieval":
            continue
        try:
            source = LocalSource() if resolved.source == "local" else MtebSource()
            task = source.load_mteb_task(resolved)
            task.load_data()
            dataset = getattr(task, "dataset", None)
            if not dataset:
                continue
            # Retrieval nesting: {subset: {split: RetrievalSplitData}}
            for subset in dataset.values() if isinstance(dataset, dict) else []:
                if not isinstance(subset, dict):
                    continue
                for split in subset.values():
                    corpus = getattr(split, "corpus", None)
                    if corpus is None:
                        continue
                    for row in corpus:
                        text = row.get("text") if isinstance(row, dict) else None
                        if text:
                            texts.append(str(text))
                        if len(texts) >= 4096:
                            return texts
        except Exception:
            continue
        if texts:
            return texts
    if not texts:
        texts = [f"placeholder document {i} about policies and procedures." for i in range(256)]
    return texts


def _write_result(result: JobResult) -> None:
    result_path(result.spec).write_text(result.model_dump_json(indent=2))


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        return None
    return None


def _reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
