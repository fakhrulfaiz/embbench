from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from embbench.core.config import REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", str(REPO_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = ""
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen2.5-7b-instruct"
    llm_timeout_s: float = 120.0
    chunker_profile: str = "bce_500_50_min100"
    export_dir: Path = Field(default_factory=lambda: REPO_ROOT / "data")
    gensvc_host: str = "0.0.0.0"
    gensvc_port: int = 8080

    def require_database_url(self) -> str:
        if not self.database_url.strip():
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill the DSN."
            )
        return self.database_url


def get_settings() -> Settings:
    return Settings()
