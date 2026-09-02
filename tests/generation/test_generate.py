from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from embbench.generation.models import Chunk
from embbench.generation.sts.generate import generate_sts_pairs
from embbench.generation.sts.llm import parse_json_object

PROFORMA = "Investigator obtains a USD or TZS proforma and checks type and count."


def _chunks() -> list[Chunk]:
    docs = [uuid4(), uuid4()]
    texts_a = [
        (
            "Investigator obtains a USD or TZS proforma, checks type and count, "
            "and includes bank SWIFT details."
        ),
        (
            "After the proforma is approved, the PI checks the pack against "
            "the grant budget before purchase."
        ),
        (
            "Goods receipt is recorded in the store ledger with the delivery "
            "note attached to the file."
        ),
    ]
    texts_b = [
        (
            "Annual leave is fourteen days and must be requested two weeks "
            "in advance through HR."
        ),
        (
            "Sick leave longer than three days requires a medical certificate "
            "from a registered clinic."
        ),
    ]
    base = datetime(2026, 9, 1, tzinfo=UTC)
    out: list[Chunk] = []
    for doc, texts in ((docs[0], texts_a), (docs[1], texts_b)):
        for i, text in enumerate(texts):
            out.append(
                Chunk(
                    chunk_id=uuid4(),
                    doc_id=doc,
                    content=text,
                    language="en",
                    chunker_profile="bce_500_50_min100",
                    created_at=base.replace(minute=i),
                )
            )
    return out


class FakeLLM:
    def complete_json(self, prompt: str) -> dict[str, Any]:
        if "Target score: 5.0" in prompt:
            return {
                "sentence1": PROFORMA,
                "sentence2": (
                    "The investigator gets a proforma in USD or TZS and "
                    "verifies quantity and type."
                ),
                "score": 5.0,
            }
        if "Target score: 4.0" in prompt:
            return {
                "sentence1": PROFORMA,
                "sentence2": (
                    "The investigator gets a proforma and, when possible, "
                    "compares about three quotes."
                ),
                "score": 4.0,
            }
        if "Target score: 3.0" in prompt:
            return {
                "sentence1": PROFORMA,
                "sentence2": (
                    "The PI checks the procurement pack against the grant "
                    "budget before approval."
                ),
                "score": 3.0,
            }
        snippet = prompt.replace("\n", " ")[:80]
        return {
            "gist": (
                f"Procedure gist for this passage: {snippet}. "
                "Follow the documented step."
            )
        }


class DirtyLLM:
    def complete_json(self, prompt: str) -> dict[str, Any]:
        del prompt
        return {
            "sentence1": "Page 1 of 12 shows the procurement header.",
            "sentence2": "SOP-Number continues on the next leaf.",
            "score": 5.0,
        }


def test_parse_json_object_strips_fences() -> None:
    raw = '```json\n{"gist": "hello world this is a gist."}\n```'
    assert parse_json_object(raw)["gist"].startswith("hello")


def test_generate_mix_with_fake_llm() -> None:
    pairs, result = generate_sts_pairs(
        _chunks(),
        count=10,
        language="en",
        llm=FakeLLM(),
        rewrite_gists=True,
        rng=random.Random(0),
    )
    assert result.written == len(pairs)
    assert result.written > 0
    kinds = {p.pair_kind for p in pairs}
    assert "paraphrase" in kinds or "generated" in kinds or "chunk_chunk" in kinds
    scores = {p.score for p in pairs}
    assert scores <= {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}


def test_dirty_llm_output_is_rejected() -> None:
    pairs, result = generate_sts_pairs(
        _chunks(),
        count=10,
        language="en",
        llm=DirtyLLM(),
        rewrite_gists=False,
        rng=random.Random(0),
    )
    high = [p for p in pairs if p.score >= 3.0]
    assert high == []
    assert result.rejected > 0
