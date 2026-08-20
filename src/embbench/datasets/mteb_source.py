"""Public MTEB tasks filtered by language and type from configs/tasks.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from embbench.core.config import get_settings
from embbench.datasets.base import ResolvedTask


class MtebSource:
    def __init__(self, tasks_yaml: Path | None = None) -> None:
        settings = get_settings()
        path = tasks_yaml or settings.configs_dir / "tasks.yaml"
        self._raw = yaml.safe_load(path.read_text())

    @property
    def k_values(self) -> list[int]:
        return list(self._raw.get("k_values") or [10, 30])

    def list_tasks(
        self,
        languages: list[str],
        task_types: list[str],
        include_heavy: bool = False,
    ) -> list[ResolvedTask]:
        out: list[ResolvedTask] = []
        lang_map = self._raw.get("languages") or {}
        for lang in languages:
            block = lang_map.get(lang)
            if not block:
                continue
            for task_type in task_types:
                key = "retrieval" if task_type == "Retrieval" else "sts"
                entries = block.get(key) or []
                if task_type == "STS" and not entries:
                    continue
                for entry in entries:
                    heavy = bool(entry.get("heavy"))
                    if heavy and not include_heavy:
                        continue
                    iso = entry.get("languages")
                    if iso is None:
                        iso_field = block.get("iso", lang)
                        iso = iso_field if isinstance(iso_field, list) else [iso_field]
                    out.append(
                        ResolvedTask(
                            name=entry["name"],
                            task_type=task_type,  # type: ignore[arg-type]
                            language=lang,
                            source="mteb",
                            heavy=heavy,
                            mteb_languages=list(iso),
                        )
                    )
        return out

    def load_mteb_task(self, resolved: ResolvedTask) -> Any:
        import mteb

        kwargs: dict[str, Any] = {}
        if resolved.mteb_languages:
            kwargs["languages"] = resolved.mteb_languages
        return mteb.get_task(resolved.name, **kwargs)

    def malay_sts_missing(self, languages: list[str], task_types: list[str]) -> bool:
        return "zsm" in languages and "STS" in task_types
