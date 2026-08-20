"""Dataset source protocol and resolved-task records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ResolvedTask(BaseModel):
    name: str
    task_type: Literal["Retrieval", "STS"]
    language: str
    source: Literal["mteb", "local"]
    heavy: bool = False
    mteb_languages: list[str] | None = None
    folder: Path | None = None
    prompt: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


@runtime_checkable
class DatasetSource(Protocol):
    def list_tasks(
        self,
        languages: list[str],
        task_types: list[str],
        include_heavy: bool = False,
    ) -> list[ResolvedTask]: ...

    def load_mteb_task(self, resolved: ResolvedTask) -> Any: ...
