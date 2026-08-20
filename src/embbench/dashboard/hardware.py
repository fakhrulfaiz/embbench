"""GPU detection for the ops page.

Queried through `nvidia-smi` rather than torch on purpose: this starts no CUDA
context and allocates no VRAM, so opening the dashboard cannot disturb a
benchmark that is running on the same card.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Gpu:
    name: str
    total_memory_gb: float

    @property
    def label(self) -> str:
        return f"{self.name} {self.total_memory_gb:.0f}GB"


@st.cache_data(show_spinner=False, ttl=300)
def detect_gpu() -> Gpu | None:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        raw = subprocess.run(
            [binary, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None

    line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    name, _, memory = line.partition(",")
    try:
        total_mib = float(memory.strip())
    except ValueError:
        return None
    return Gpu(name=name.strip(), total_memory_gb=total_mib / 1024)
