from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from embbench.generation.config import Settings, get_settings
from embbench.generation.models import export_language


@dataclass
class ExportResult:
    name: str
    language: str
    revision: str
    pairs: int
    folder: Path


def export_sts(
    rows: list[dict[str, Any]],
    *,
    name: str,
    language: str,
    revision: str | None = None,
    description: str | None = None,
    min_score: float = 0.0,
    max_score: float = 5.0,
    export_dir: Path | None = None,
    settings: Settings | None = None,
) -> ExportResult:
    cfg = settings or get_settings()
    root = Path(export_dir or cfg.export_dir)
    lang = export_language(language)
    rev = revision or datetime.now(UTC).strftime("%Y-%m-%d")
    folder = root / "sts" / name
    folder.mkdir(parents=True, exist_ok=True)

    pairs_path = folder / "pairs.jsonl"
    kept = 0
    with pairs_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            score = float(row["score"])
            if score < min_score or score > max_score:
                continue
            s1 = str(row["sentence1"]).strip()
            s2 = str(row["sentence2"]).strip()
            if not s1 or not s2:
                continue
            fh.write(
                json.dumps(
                    {"sentence1": s1, "sentence2": s2, "score": score},
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    meta = {
        "name": name,
        "language": lang,
        "min_score": min_score,
        "max_score": max_score,
        "revision": rev,
        "description": description or f"Generated STS pairs ({name}).",
    }
    (folder / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return ExportResult(name=name, language=lang, revision=rev, pairs=kept, folder=folder)
