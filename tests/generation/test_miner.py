from __future__ import annotations

import random
from datetime import UTC, datetime
from uuid import UUID, uuid4

from embbench.generation.models import Chunk
from embbench.generation.sts.miner import mine_score_0, mine_score_1, mine_score_2

_DOCS: dict[str, UUID] = {}


def _stable_doc(name: str) -> UUID:
    if name not in _DOCS:
        _DOCS[name] = uuid4()
    return _DOCS[name]


def _seq(doc: str, texts: list[str]) -> list[Chunk]:
    base = datetime(2026, 9, 1, tzinfo=UTC)
    out: list[Chunk] = []
    for i, text in enumerate(texts):
        out.append(
            Chunk(
                chunk_id=uuid4(),
                doc_id=_stable_doc(doc),
                content=text,
                language="en",
                chunker_profile="bce_500_50_min100",
                created_at=base.replace(minute=i),
            )
        )
    return out


def test_score_2_skips_adjacent_windows() -> None:
    overlap = " ".join(["overlap"] * 40)
    chunks = _seq(
        "proc",
        [
            "alpha start " + overlap,
            overlap + " alpha middle",
            "omega later step about budget approval and PI sign-off.",
        ],
    )
    rng = random.Random(0)
    pairs = mine_score_2(chunks, 20, language="en", rng=rng)
    ids = [(p.source_chunk_id, p.source_chunk_id_2) for p in pairs]
    adjacent = {
        (chunks[0].chunk_id, chunks[1].chunk_id),
        (chunks[1].chunk_id, chunks[2].chunk_id),
    }
    assert pairs
    assert all(p.score == 2.0 and p.pair_kind == "chunk_chunk" for p in pairs)
    assert not adjacent.intersection(ids)


def test_score_0_uses_different_docs() -> None:
    a = _seq("a", ["Leave policy: annual leave is fourteen days per calendar year."])
    b = _seq("b", ["Procurement: obtain three quotes and attach the proforma invoice."])
    pairs = mine_score_0(a + b, 5, language="en", rng=random.Random(1))
    assert pairs
    for p in pairs:
        assert p.score == 0.0
        left = next(c for c in a + b if c.chunk_id == p.source_chunk_id)
        right = next(c for c in a + b if c.chunk_id == p.source_chunk_id_2)
        assert left.doc_id != right.doc_id


def test_score_1_weak_overlap_same_doc() -> None:
    chunks = _seq(
        "hr",
        [
            "Annual leave accrual uses the calendar year and requires "
            "manager approval.",
            "Sick leave needs a medical certificate after three consecutive days.",
            "Parking permits are issued by facilities on the ground floor lobby.",
        ],
    )
    pairs = mine_score_1(chunks, 5, language="en", rng=random.Random(2))
    assert pairs
    assert all(p.score == 1.0 for p in pairs)
    ids = {c.chunk_id: c for c in chunks}
    for p in pairs:
        assert ids[p.source_chunk_id].doc_id == ids[p.source_chunk_id_2].doc_id
