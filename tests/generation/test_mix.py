from __future__ import annotations

from embbench.generation.sts.mix import allocate_counts


def test_allocate_counts_sums_to_total() -> None:
    for total in (0, 1, 10, 17, 200):
        counts = allocate_counts(total)
        assert sum(counts.values()) == total
        assert set(counts) == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}


def test_allocate_counts_ten_matches_documented_mix() -> None:
    counts = allocate_counts(10)
    assert counts[0.0] == 1
    assert counts[5.0] == 1
    assert counts[1.0] == 2
    assert counts[2.0] == 2
    assert counts[3.0] == 2
    assert counts[4.0] == 2
