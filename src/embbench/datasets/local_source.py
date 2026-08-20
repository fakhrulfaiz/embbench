"""Discover drop-in datasets under data/sts/ and data/retrieval/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from embbench.core.config import get_settings
from embbench.datasets.adapters import build_local_task
from embbench.datasets.base import ResolvedTask


def _read_meta(folder: Path) -> dict[str, Any]:
    meta_path = folder / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    raw = yaml.safe_load(meta_path.read_text()) or {}
    raw.setdefault("name", folder.name)
    raw.setdefault("revision", "0")
    return raw


class LocalSource:
    def __init__(self, data_dir: Path | None = None) -> None:
        settings = get_settings()
        self.data_dir = data_dir or settings.data_dir

    def list_tasks(
        self,
        languages: list[str],
        task_types: list[str],
        include_heavy: bool = False,
    ) -> list[ResolvedTask]:
        del include_heavy
        out: list[ResolvedTask] = []
        mapping = {"STS": self.data_dir / "sts", "Retrieval": self.data_dir / "retrieval"}
        lang_aliases = _expand_langs(languages)
        for task_type in task_types:
            root = mapping[task_type]
            if not root.exists():
                continue
            for folder in sorted(p for p in root.iterdir() if p.is_dir()):
                try:
                    meta = _read_meta(folder)
                except FileNotFoundError:
                    continue
                language = str(meta.get("language") or meta.get("lang") or "und")
                if lang_aliases and not _lang_matches(language, lang_aliases):
                    continue
                out.append(
                    ResolvedTask(
                        name=str(meta["name"]),
                        task_type=task_type,  # type: ignore[arg-type]
                        language=_canonical_lang(language),
                        source="local",
                        folder=folder,
                        prompt=meta.get("prompt"),
                        extra=meta,
                    )
                )
        return out

    def load_mteb_task(self, resolved: ResolvedTask) -> Any:
        if resolved.folder is None:
            raise ValueError(f"Local task {resolved.name} has no folder")
        meta = _read_meta(resolved.folder)
        return build_local_task(resolved.folder, meta, resolved.task_type)


def _expand_langs(languages: list[str]) -> set[str]:
    aliases = {
        "eng": {"eng", "en", "eng-latn"},
        "cmn": {"cmn", "zh", "zho", "cmn-hans", "zho-hans"},
        "zsm": {"zsm", "msa", "zlm", "ms", "may", "zsm-latn", "msa-latn", "zlm-latn"},
    }
    out: set[str] = set()
    for lang in languages:
        key = lang.lower()
        out |= aliases.get(key, {key})
        out.add(key)
    return {x.lower() for x in out}


def _lang_matches(language: str, aliases: set[str]) -> bool:
    token = language.lower().replace("_", "-")
    if token in aliases:
        return True
    head = token.split("-")[0]
    return head in aliases


def _canonical_lang(language: str) -> str:
    token = language.lower().split("-")[0]
    mapping = {"en": "eng", "zh": "cmn", "zho": "cmn", "ms": "zsm", "msa": "zsm", "zlm": "zsm", "may": "zsm"}
    return mapping.get(token, token)
