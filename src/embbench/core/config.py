"""Settings and HuggingFace cache bootstrap.

Import and call `bootstrap_env()` before any `torch` / `transformers` /
`huggingface_hub` / `mteb` import.

On WSL, Hub files must live under `/mnt/c` (Windows drive), not ext4.
On native Linux, the default is `~/.cache/huggingface`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
WSL_HF_PREFIX = "/mnt/c"


def _on_wsl_windows_drive() -> bool:
    root = Path(WSL_HF_PREFIX)
    try:
        return root.is_dir() and os.access(root, os.W_OK)
    except OSError:
        return False


def bootstrap_env(repo_root: Path | None = None) -> Path:
    """Load `.env`, set HF cache (WSL `/mnt/c` or Linux `~/.cache`), create dirs."""
    root = repo_root or REPO_ROOT
    load_dotenv(root / ".env", override=False)

    if _on_wsl_windows_drive():
        hf_home = "/mnt/c/ml-cache/huggingface"
    else:
        hf_home = str(Path.home() / ".cache" / "huggingface")

    defaults = {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": f"{hf_home}/hub",
        "HF_DATASETS_CACHE": f"{hf_home}/datasets",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "MTEB_CACHE": str(root / "results" / "mteb-cache"),
        "EMBBENCH_RESULTS_DIR": str(root / "results"),
        "EMBBENCH_DATA_DIR": str(root / "data"),
        "EMBBENCH_CONFIGS_DIR": str(root / "configs"),
        "EMBBENCH_EMBEDDINGS_URL": "http://127.0.0.1:8001",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    hf_home_str = os.path.normpath(str(Path(os.environ["HF_HOME"]).expanduser()))
    if _on_wsl_windows_drive() and not hf_home_str.startswith(WSL_HF_PREFIX):
        raise RuntimeError(
            f"HF_HOME must live under {WSL_HF_PREFIX} so Hub files land on "
            f"the Windows drive, not WSL ext4. Got {hf_home_str!r}. "
            "Fix .env (see .env.example)."
        )

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)

    for rel_key in ("MTEB_CACHE", "EMBBENCH_RESULTS_DIR", "EMBBENCH_DATA_DIR", "EMBBENCH_CONFIGS_DIR"):
        value = Path(os.environ[rel_key]).expanduser()
        if not value.is_absolute():
            value = root / value
            os.environ[rel_key] = str(value)
        Path(os.environ[rel_key]).mkdir(parents=True, exist_ok=True)
    return root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    hf_home: Path = Field(alias="HF_HOME")
    hf_hub_cache: Path = Field(alias="HF_HUB_CACHE")
    hf_datasets_cache: Path = Field(alias="HF_DATASETS_CACHE")
    mteb_cache: Path = Field(alias="MTEB_CACHE")
    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    embeddings_url: str = Field(
        default="http://127.0.0.1:8001", alias="EMBBENCH_EMBEDDINGS_URL"
    )
    embeddings_api_key: str | None = Field(default=None, alias="EMBBENCH_EMBEDDINGS_API_KEY")
    results_dir: Path = Field(alias="EMBBENCH_RESULTS_DIR")
    data_dir: Path = Field(alias="EMBBENCH_DATA_DIR")
    configs_dir: Path = Field(alias="EMBBENCH_CONFIGS_DIR")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        bootstrap_env()
        _settings = Settings()
    return _settings
