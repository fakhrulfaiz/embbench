"""Settings and the /mnt/c HuggingFace cache guard.

Import and call `bootstrap_env()` before any `torch` / `transformers` /
`huggingface_hub` / `mteb` import so Hub traffic cannot land on WSL ext4.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_HF_PREFIX = "/mnt/c"


def bootstrap_env(repo_root: Path | None = None) -> Path:
    """Load `.env`, force HF cache onto /mnt/c, and refuse to continue otherwise."""
    root = repo_root or REPO_ROOT
    load_dotenv(root / ".env", override=False)

    defaults = {
        "HF_HOME": "/mnt/c/ml-cache/huggingface",
        "HF_HUB_CACHE": "/mnt/c/ml-cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/mnt/c/ml-cache/huggingface/datasets",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "MTEB_CACHE": str(root / "results" / "mteb-cache"),
        "EMBBENCH_RESULTS_DIR": str(root / "results"),
        "EMBBENCH_DATA_DIR": str(root / "data"),
        "EMBBENCH_CONFIGS_DIR": str(root / "configs"),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    hf_home = Path(os.environ["HF_HOME"]).expanduser()
    # Do not resolve() through /mnt/c symlinks in a way that leaves the prefix.
    hf_home_str = os.path.normpath(str(hf_home))
    if not hf_home_str.startswith(REQUIRED_HF_PREFIX):
        raise RuntimeError(
            f"HF_HOME must live under {REQUIRED_HF_PREFIX} so Hub files land on "
            f"the Windows drive, not WSL ext4. Got {hf_home_str!r}. "
            "Fix .env (see .env.example) and do not point HF_HOME at ~/.cache."
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
