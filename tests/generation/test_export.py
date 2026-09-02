from __future__ import annotations

from pathlib import Path

from embbench.generation.sts.export import export_sts


def test_export_writes_pairs_and_meta(tmp_path: Path) -> None:
    rows = [
        {
            "sentence1": "Investigator obtains a proforma.",
            "sentence2": "The investigator gets a proforma invoice.",
            "score": 5.0,
        },
        {
            "sentence1": "empty",
            "sentence2": "",
            "score": 4.0,
        },
        {
            "sentence1": "Leave is fourteen days.",
            "sentence2": "Parking is in the lobby.",
            "score": 0.0,
        },
    ]
    result = export_sts(
        rows,
        name="sop-sts-v1",
        language="en",
        revision="2026-09-01",
        export_dir=tmp_path,
    )
    assert result.language == "eng-Latn"
    assert result.pairs == 2
    folder = tmp_path / "sts" / "sop-sts-v1"
    pairs = (folder / "pairs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(pairs) == 2
    meta = (folder / "meta.yaml").read_text(encoding="utf-8")
    assert "name: sop-sts-v1" in meta
    assert "language: eng-Latn" in meta
    assert "2026-09-01" in meta


def test_export_skips_out_of_range_scores(tmp_path: Path) -> None:
    rows = [{"sentence1": "a" * 30, "sentence2": "b" * 30, "score": 9.0}]
    result = export_sts(rows, name="x", language="eng-Latn", export_dir=tmp_path)
    assert result.pairs == 0
