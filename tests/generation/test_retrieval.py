from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from embbench.generation.models import Chunk, QuestionDraft
from embbench.generation.retrieval.export import export_retrieval
from embbench.generation.retrieval.generate import extract_question, generate_questions


def _chunk(text: str, *, doc: UUID | None = None) -> Chunk:
    return Chunk(
        chunk_id=uuid4(),
        doc_id=doc or uuid4(),
        content=text,
        language="en",
        chunker_profile="bce_500_50_min100",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


class FakeQuestionLLM:
    def complete_json(self, prompt: str) -> dict[str, Any]:
        del prompt
        return {"question": "How does an investigator obtain a USD proforma?"}


class DirtyQuestionLLM:
    def complete_json(self, prompt: str) -> dict[str, Any]:
        del prompt
        return {"question": "See Page 1 of 12 and SOP-Number for procurement."}


def test_generate_skips_labelled_chunks() -> None:
    a = _chunk("Investigator obtains a USD proforma and checks type and count.")
    b = _chunk("Annual leave is fourteen days per calendar year with HR approval.")
    questions, result = generate_questions(
        [a, b],
        count=5,
        language="en",
        llm=FakeQuestionLLM(),
        labelled={a.chunk_id},
        force=False,
    )
    assert result.skipped_labelled == 1
    assert len(questions) == 1
    assert questions[0].gold_chunk_id == b.chunk_id
    assert questions[0].question_text.startswith("How does")


def test_generate_all_remaining_chunks() -> None:
    a = _chunk("Investigator obtains a USD proforma and checks type and count.")
    b = _chunk("Annual leave is fourteen days per calendar year with HR approval.")
    questions, result = generate_questions(
        [a, b],
        count=None,
        language="en",
        llm=FakeQuestionLLM(),
        labelled={a.chunk_id},
        force=False,
    )
    assert result.requested == 1
    assert len(questions) == 1
    assert questions[0].gold_chunk_id == b.chunk_id


def test_language_all_keeps_chunk_language() -> None:
    from embbench.generation.models import language_filter_aliases

    assert language_filter_aliases("all") is None
    assert language_filter_aliases("zh") == ("zh", "zho", "cmn")
    zh_chunk = _chunk("年假每年十四天，须经人力资源部批准。")
    zh_chunk = Chunk(
        chunk_id=zh_chunk.chunk_id,
        doc_id=zh_chunk.doc_id,
        content=zh_chunk.content,
        language="zh",
        chunker_profile=zh_chunk.chunker_profile,
        created_at=zh_chunk.created_at,
    )
    questions, _result = generate_questions(
        [zh_chunk],
        count=None,
        language="all",
        llm=FakeQuestionLLM(),
        labelled=set(),
    )
    assert questions[0].language == "zh"
    a = _chunk("Investigator obtains a USD proforma and checks type and count.")
    questions, result = generate_questions(
        [a],
        count=1,
        language="en",
        llm=FakeQuestionLLM(),
        labelled={a.chunk_id},
        force=True,
    )
    assert result.skipped_labelled == 0
    assert len(questions) == 1
    assert questions[0].gold_chunk_id == a.chunk_id


def test_dirty_question_is_rejected() -> None:
    chunk = _chunk("Investigator obtains a USD proforma and checks type and count.")
    questions, result = generate_questions(
        [chunk],
        count=1,
        language="en",
        llm=DirtyQuestionLLM(),
        labelled=set(),
    )
    assert questions == []
    assert result.rejected > 0


def test_generic_question_is_rejected() -> None:
    class GenericLLM:
        def complete_json(self, prompt: str) -> dict[str, Any]:
            del prompt
            return {"question": "What is this document about?"}

    chunk = _chunk("Investigator obtains a USD proforma and checks type and count.")
    questions, result = generate_questions(
        [chunk],
        count=1,
        language="en",
        llm=GenericLLM(),
        labelled=set(),
    )
    assert questions == []
    assert result.rejected > 0


def test_round_robin_spreads_across_docs() -> None:
    doc_a, doc_b = uuid4(), uuid4()
    chunks = [
        _chunk(
            "Investigator obtains a USD proforma and checks type and count.",
            doc=doc_a,
        ),
        _chunk("Then the PI signs the procurement pack before purchase.", doc=doc_a),
        _chunk(
            "Annual leave is fourteen days per calendar year with HR approval.",
            doc=doc_b,
        ),
        _chunk(
            "Sick leave longer than three days needs a medical certificate.",
            doc=doc_b,
        ),
    ]

    class EchoLLM:
        def complete_json(self, prompt: str) -> dict[str, Any]:
            if "proforma" in prompt or "procurement" in prompt:
                return {
                    "question": "How does an investigator obtain a USD proforma?"
                }
            return {
                "question": "How many days of annual leave does HR approve?"
            }

    questions, _result = generate_questions(
        chunks,
        count=2,
        language="en",
        llm=EchoLLM(),
        labelled=set(),
        rng=random.Random(0),
    )
    gold_docs = {
        next(c.doc_id for c in chunks if c.chunk_id == q.gold_chunk_id)
        for q in questions
    }
    assert len(questions) == 2
    assert gold_docs == {doc_a, doc_b}


def test_plain_text_complete_like_vllm() -> None:
    class PlainTextLLM:
        def complete(self, prompt: str) -> str:
            del prompt
            return "How does an investigator obtain a USD proforma?"

        def complete_json(self, prompt: str) -> dict[str, Any]:
            del prompt
            raise AssertionError("question gen should use complete()")

    chunk = _chunk("Investigator obtains a USD proforma and checks type and count.")
    questions, result = generate_questions(
        [chunk],
        count=1,
        language="en",
        llm=PlainTextLLM(),
        labelled=set(),
    )
    assert result.written == 1
    assert questions[0].question_text.endswith("proforma?")


def test_concurrent_generate_writes_count() -> None:
    doc = uuid4()
    chunks = [
        _chunk("Investigator obtains a USD proforma and checks type and count.", doc=doc),
        _chunk("Annual leave is fourteen days per calendar year with HR approval.", doc=doc),
        _chunk("Sick leave longer than three days needs a medical certificate.", doc=doc),
        _chunk("Goods receipt is recorded in the store ledger with the delivery note.", doc=doc),
    ]
    questions, result = generate_questions(
        chunks,
        count=3,
        language="en",
        llm=FakeQuestionLLM(),
        labelled=set(),
        concurrency=3,
        rng=random.Random(0),
    )
    assert result.written == 3
    assert len(questions) == 3
    assert len({q.gold_chunk_id for q in questions}) == 3


def test_extract_question_plain_and_json() -> None:
    assert (
        extract_question('{"question": "How many days of annual leave?"}')
        == "How many days of annual leave?"
    )
    assert (
        extract_question("How many days of annual leave?")
        == "How many days of annual leave?"
    )
    assert (
        extract_question('"How many days of annual leave?"')
        == "How many days of annual leave?"
    )


def test_export_retrieval_files(tmp_path: Path) -> None:
    chunk = _chunk(
        "Investigator obtains a USD proforma and checks type and count.",
    )
    question = QuestionDraft(
        question_text="How does an investigator obtain a USD proforma?",
        gold_chunk_id=chunk.chunk_id,
        language="en",
    )
    result = export_retrieval(
        [chunk],
        [
            {
                "question_id": question.question_id,
                "gold_chunk_id": question.gold_chunk_id,
                "question_text": question.question_text,
            }
        ],
        name="sop-handbook-v1",
        language="en",
        revision="2026-09-01",
        export_dir=tmp_path,
    )
    folder = tmp_path / "retrieval" / "sop-handbook-v1"
    assert result.corpus == 1
    assert result.queries == 1
    assert result.qrels == 1
    corpus = (folder / "corpus.jsonl").read_text(encoding="utf-8")
    assert str(chunk.chunk_id) in corpus
    queries = (folder / "queries.jsonl").read_text(encoding="utf-8")
    assert "USD proforma" in queries
    qrels = (folder / "qrels.tsv").read_text(encoding="utf-8")
    assert f"{question.question_id}\t{chunk.chunk_id}\t1" in qrels
    meta = (folder / "meta.yaml").read_text(encoding="utf-8")
    assert "name: sop-handbook-v1" in meta
    assert "language: eng-Latn" in meta
