"""Model registry loaded from configs/models.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from embbench.core.config import get_settings


class ModelConfig(BaseModel):
    id: str
    hf_name: str
    role: str
    loader: Literal["mteb", "sentence_transformers", "openai_api"] = "sentence_transformers"
    trust_remote_code: bool = False
    use_instructions: bool = False
    max_seq_length: int = 512
    batch_size: int = 16
    description: str = ""
    dtype: str = "float16"
    endpoint_url: str | None = None
    api_key: str | None = None
    use_chat_template: bool = False
    instruction_template: str | None = None


class ModelsFile(BaseModel):
    max_seq_length: int = 512
    dtype: str = "float16"
    models: list[ModelConfig] = Field(default_factory=list)


def load_models(path: Path | None = None) -> list[ModelConfig]:
    settings = get_settings()
    yaml_path = path or settings.configs_dir / "models.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    parsed = ModelsFile.model_validate(raw)
    for model in parsed.models:
        if not model.max_seq_length:
            model.max_seq_length = parsed.max_seq_length
        if not model.dtype:
            model.dtype = parsed.dtype
    return parsed.models


def get_model_config(model_id: str, path: Path | None = None) -> ModelConfig:
    models = load_models(path)
    for model in models:
        if model.id == model_id or model.hf_name == model_id:
            return model
    known = ", ".join(m.id for m in models)
    raise KeyError(f"Unknown model {model_id!r}. Known ids: {known}")
