from __future__ import annotations

from embbench.generation.models import SCORE_SHARES


def allocate_counts(total: int, shares: dict[float, float] | None = None) -> dict[float, int]:
    """Largest-remainder allocation so counts sum to `total`."""
    if total < 0:
        raise ValueError("total must be >= 0")
    table = shares or SCORE_SHARES
    if total == 0:
        return {score: 0 for score in table}
    raw = {score: total * share for score, share in table.items()}
    counts = {score: int(value) for score, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda s: (raw[s] - counts[s], s), reverse=True)
    for score in order[:remainder]:
        counts[score] += 1
    return counts
