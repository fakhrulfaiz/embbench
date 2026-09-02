from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from embbench.core.config import REPO_ROOT
from embbench.generation.config import Settings, get_settings
from embbench.generation.models import Chunk, PairDraft, QuestionDraft

_PACKAGE_DIR = Path(__file__).resolve().parent


def _schema_path() -> Path:
    candidates = [
        REPO_ROOT / "sql" / "schema.sql",
        _PACKAGE_DIR / "sql" / "schema.sql",
        Path.cwd() / "sql" / "schema.sql",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"sql/schema.sql not found (tried {_PACKAGE_DIR / 'sql'})")


@contextmanager
def connect(settings: Settings | None = None) -> Iterator[psycopg.Connection]:
    cfg = settings or get_settings()
    with psycopg.connect(cfg.require_database_url(), row_factory=dict_row) as conn:
        yield conn


def init_schema(settings: Settings | None = None, schema_path: Path | None = None) -> None:
    path = schema_path or _schema_path()
    sql = path.read_text(encoding="utf-8")
    with connect(settings) as conn:
        conn.execute(sql)
        conn.commit()


def fetch_chunks(
    *,
    profile: str,
    language: str | None = None,
    settings: Settings | None = None,
) -> list[Chunk]:
    sql = """
        SELECT chunk_id, doc_id, content, language,
               chunker AS chunker_profile, created_at
        FROM chunks
        WHERE chunker = %(profile)s
          AND content IS NOT NULL
          AND length(trim(content)) > 0
    """
    params: dict[str, Any] = {"profile": profile}
    if language:
        from embbench.generation.models import language_filter_aliases

        aliases = language_filter_aliases(language)
        if aliases:
            sql += " AND lower(split_part(language, '-', 1)) = ANY(%(langs)s)"
            params["langs"] = list(aliases)
    sql += " ORDER BY doc_id, created_at, chunk_id"

    with connect(settings) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_chunk_from_row(r) for r in rows]


def insert_pairs(pairs: Sequence[PairDraft], settings: Settings | None = None) -> int:
    if not pairs:
        return 0
    sql = """
        INSERT INTO sts_pairs (
            pair_id, sentence1, sentence2, score,
            source_chunk_id, pair_kind, language
        )
        VALUES (
            %(pair_id)s, %(sentence1)s, %(sentence2)s, %(score)s,
            %(source_chunk_id)s, %(pair_kind)s, %(language)s
        )
    """
    with connect(settings) as conn:
        with conn.cursor() as cur:
            for pair in pairs:
                cur.execute(
                    sql,
                    {
                        "pair_id": pair.pair_id,
                        "sentence1": pair.sentence1,
                        "sentence2": pair.sentence2,
                        "score": pair.score,
                        "source_chunk_id": pair.source_chunk_id,
                        "pair_kind": pair.pair_kind,
                        "language": pair.language,
                    },
                )
        conn.commit()
    return len(pairs)


def fetch_pairs(
    *,
    language: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    sql = """
        SELECT pair_id, sentence1, sentence2, score, source_chunk_id, pair_kind, language
        FROM sts_pairs
    """
    params: dict[str, Any] = {}
    if language:
        from embbench.generation.models import language_filter_aliases

        aliases = language_filter_aliases(language)
        if aliases:
            sql += " WHERE lower(split_part(language, '-', 1)) = ANY(%(langs)s)"
            params["langs"] = list(aliases)
    sql += " ORDER BY created_at, pair_id"
    with connect(settings) as conn:
        return list(conn.execute(sql, params).fetchall())


def fetch_labelled_chunk_ids(settings: Settings | None = None) -> set[UUID]:
    with connect(settings) as conn:
        rows = conn.execute("SELECT gold_chunk_id FROM retrieval_questions").fetchall()
    return {row["gold_chunk_id"] for row in rows}


def insert_questions(
    questions: Sequence[QuestionDraft], settings: Settings | None = None
) -> int:
    if not questions:
        return 0
    sql = """
        INSERT INTO retrieval_questions (
            question_id, gold_chunk_id, question_text, language
        )
        VALUES (
            %(question_id)s, %(gold_chunk_id)s, %(question_text)s, %(language)s
        )
    """
    with connect(settings) as conn:
        with conn.cursor() as cur:
            for question in questions:
                cur.execute(
                    sql,
                    {
                        "question_id": question.question_id,
                        "gold_chunk_id": question.gold_chunk_id,
                        "question_text": question.question_text,
                        "language": question.language,
                    },
                )
        conn.commit()
    return len(questions)


def fetch_questions(
    *,
    language: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    sql = """
        SELECT question_id, gold_chunk_id, question_text, language
        FROM retrieval_questions
    """
    params: dict[str, Any] = {}
    if language:
        from embbench.generation.models import language_filter_aliases

        aliases = language_filter_aliases(language)
        if aliases:
            sql += " WHERE lower(split_part(language, '-', 1)) = ANY(%(langs)s)"
            params["langs"] = list(aliases)
    sql += " ORDER BY created_at, question_id"
    with connect(settings) as conn:
        return list(conn.execute(sql, params).fetchall())


def store_stats(settings: Settings | None = None) -> dict[str, Any]:
    with connect(settings) as conn:
        chunk_n = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        pair_n = conn.execute("SELECT count(*) AS n FROM sts_pairs").fetchone()["n"]
        question_n = conn.execute(
            "SELECT count(*) AS n FROM retrieval_questions"
        ).fetchone()["n"]
        by_lang = conn.execute(
            """
            SELECT lower(split_part(language, '-', 1)) AS lang, count(*) AS n
            FROM chunks
            GROUP BY 1
            ORDER BY n DESC, lang
            """
        ).fetchall()
        by_score = conn.execute(
            """
            SELECT score, pair_kind, count(*) AS n
            FROM sts_pairs
            GROUP BY score, pair_kind
            ORDER BY score, pair_kind
            """
        ).fetchall()
    return {
        "chunks": int(chunk_n),
        "pairs": int(pair_n),
        "questions": int(question_n),
        "by_language": [
            {"language": r["lang"] or "", "count": int(r["n"])} for r in by_lang
        ],
        "by_score": [
            {
                "score": float(r["score"]),
                "pair_kind": r["pair_kind"],
                "count": int(r["n"]),
            }
            for r in by_score
        ],
    }


def pair_stats(settings: Settings | None = None) -> dict[str, Any]:
    return store_stats(settings)


def insert_export(
    *,
    name: str,
    task_type: str,
    language: str,
    revision: str,
    settings: Settings | None = None,
) -> UUID:
    export_id = uuid4()
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO dataset_exports (export_id, name, task_type, language, revision)
            VALUES (%(export_id)s, %(name)s, %(task_type)s, %(language)s, %(revision)s)
            """,
            {
                "export_id": export_id,
                "name": name,
                "task_type": task_type,
                "language": language,
                "revision": revision,
            },
        )
        conn.commit()
    return export_id


def _chunk_from_row(row: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        content=row["content"],
        language=row.get("language") or "",
        chunker_profile=row.get("chunker_profile") or "",
        created_at=row.get("created_at"),
    )
