"""Wrap a local folder as an MTEB AbsTaskSTS / AbsTaskRetrieval subclass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


def build_local_task(folder: Path, meta: dict[str, Any], task_type: str) -> Any:
    if task_type == "STS":
        return _build_sts(folder, meta)
    if task_type == "Retrieval":
        return _build_retrieval(folder, meta)
    raise ValueError(f"Unsupported local task type {task_type}")


def _eval_langs(meta: dict[str, Any]) -> list[str]:
    lang = str(meta.get("language") or "und")
    if "-" not in lang and lang in {"eng", "zsm", "msa", "zlm", "ind"}:
        return [f"{lang}-Latn"]
    if lang in {"cmn", "zho"}:
        return ["cmn-Hans"]
    return [lang]


def _build_sts(folder: Path, meta: dict[str, Any]):
    import mteb
    from mteb.abstasks import AbsTaskSTS

    name = str(meta["name"])
    revision = str(meta.get("revision") or "0")
    pairs_path = folder / "pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"{folder} is missing pairs.jsonl")

    class LocalSTS(AbsTaskSTS):
        min_score = float(meta.get("min_score", 0))
        max_score = float(meta.get("max_score", 5))
        column_names = ("sentence1", "sentence2")
        metadata = mteb.TaskMetadata(
            name=name,
            description=str(meta.get("description") or f"Local STS dataset {name}"),
            type="STS",
            category="t2t",
            modalities=["text"],
            main_score="cosine_spearman",
            eval_langs=_eval_langs(meta),
            eval_splits=["test"],
            dataset={"path": str(folder), "revision": revision},
            prompt=meta.get("prompt") or "Retrieve semantically similar text.",
            reference=None,
            license=str(meta.get("license") or "not specified"),
        )

        def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
            del num_proc, kwargs
            if self.data_loaded:
                return
            ds = Dataset.from_json(str(pairs_path))
            self.dataset = DatasetDict({"test": ds})
            self.data_loaded = True

    LocalSTS.__name__ = name
    return LocalSTS()


def _build_retrieval(folder: Path, meta: dict[str, Any]):
    import mteb
    from mteb.abstasks import AbsTaskRetrieval

    try:
        from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
    except ImportError:
        from mteb.abstasks.retrieval import RetrievalSplitData  # type: ignore[attr-defined]


    name = str(meta["name"])
    revision = str(meta.get("revision") or "0")
    corpus_path = folder / "corpus.jsonl"
    queries_path = folder / "queries.jsonl"
    qrels_path = folder / "qrels.tsv"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"{folder} is missing {path.name}")

    class LocalRetrieval(AbsTaskRetrieval):
        metadata = mteb.TaskMetadata(
            name=name,
            description=str(meta.get("description") or f"Local retrieval dataset {name}"),
            type="Retrieval",
            category="t2t",
            modalities=["text"],
            main_score="ndcg_at_10",
            eval_langs=_eval_langs(meta),
            eval_splits=["test"],
            dataset={"path": str(folder), "revision": revision},
            prompt=meta.get("prompt") or "Retrieve relevant passages that answer the query.",
            reference=None,
            license=str(meta.get("license") or "not specified"),
        )

        def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
            del num_proc, kwargs
            if self.data_loaded:
                return
            corpus_rows = [
                _corpus_row(obj) for obj in _iter_jsonl(corpus_path)
            ]
            query_rows = [
                _query_row(obj) for obj in _iter_jsonl(queries_path)
            ]
            qrels = _load_qrels(qrels_path)
            self.dataset = {
                "default": {
                    "test": RetrievalSplitData(
                        corpus=Dataset.from_list(corpus_rows),
                        queries=Dataset.from_list(query_rows),
                        relevant_docs=qrels,
                        top_ranked=None,
                    )
                }
            }
            self.data_loaded = True

    LocalRetrieval.__name__ = name
    return LocalRetrieval()


def _iter_jsonl(path: Path):
    import json

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _corpus_row(obj: dict[str, Any]) -> dict[str, Any]:
    doc_id = str(obj.get("_id") or obj.get("id"))
    title = obj.get("title") or ""
    text = obj.get("text") or obj.get("contents") or ""
    return {"id": doc_id, "title": title, "text": text}


def _query_row(obj: dict[str, Any]) -> dict[str, Any]:
    query_id = str(obj.get("_id") or obj.get("id"))
    text = obj.get("text") or obj.get("query") or ""
    return {"id": query_id, "text": text}


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", "\t").split("\t") if "\t" in line or "," in line else line.split()
            if i == 0 and parts[0].lower() in {"query-id", "query_id", "qid"}:
                continue
            if len(parts) == 4:
                # TREC: qid iter docid rel
                qid, _, docid, rel = parts
            elif len(parts) == 3:
                qid, docid, rel = parts
            else:
                raise ValueError(f"Cannot parse qrels line: {line!r}")
            qrels.setdefault(str(qid), {})[str(docid)] = int(float(rel))
    return qrels
