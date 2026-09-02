from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from embbench.generation.config import Settings, get_settings
from embbench.generation.models import Chunk, export_language


@dataclass
class RetrievalExportResult:
    name: str
    language: str
    revision: str
    corpus: int
    queries: int
    qrels: int
    folder: Path


def export_retrieval(
    chunks: list[Chunk],
    questions: list[dict[str, Any]],
    *,
    name: str,
    language: str,
    revision: str | None = None,
    description: str | None = None,
    export_dir: Path | None = None,
    settings: Settings | None = None,
) -> RetrievalExportResult:
    cfg = settings or get_settings()
    root = Path(export_dir or cfg.export_dir)
    lang = export_language(language)
    rev = revision or datetime.now(UTC).strftime("%Y-%m-%d")
    folder = root / "retrieval" / name
    folder.mkdir(parents=True, exist_ok=True)

    corpus_ids: set[UUID] = set()
    with (folder / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            text = chunk.content.strip()
            if not text:
                continue
            corpus_ids.add(chunk.chunk_id)
            fh.write(
                json.dumps(
                    {"_id": str(chunk.chunk_id), "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )

    query_n = 0
    qrel_n = 0
    with (
        (folder / "queries.jsonl").open("w", encoding="utf-8") as qf,
        (folder / "qrels.tsv").open("w", encoding="utf-8") as rf,
    ):
        rf.write("query-id\tcorpus-id\tscore\n")
        for row in questions:
            text = str(row["question_text"]).strip()
            if not text:
                continue
            qid = str(row["question_id"])
            gold = row["gold_chunk_id"]
            if not isinstance(gold, UUID):
                gold = UUID(str(gold))
            qf.write(
                json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n"
            )
            query_n += 1
            if gold in corpus_ids:
                rf.write(f"{qid}\t{gold}\t1\n")
                qrel_n += 1

    meta = {
        "name": name,
        "language": lang,
        "revision": rev,
        "description": description or f"Generated retrieval task ({name}).",
        "prompt": (
            "Given a question, retrieve the handbook passage that answers it."
        ),
    }
    (folder / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return RetrievalExportResult(
        name=name,
        language=lang,
        revision=rev,
        corpus=len(corpus_ids),
        queries=query_n,
        qrels=qrel_n,
        folder=folder,
    )
