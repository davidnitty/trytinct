"""Hardware / VRAM detection.

Lightweight by design: we only probe for GPUs when a train/eval command
actually runs, and it degrades gracefully to CPU when no accelerator is found.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareInfo:
    has_cuda: bool
    device_count: int
    device_names: list[str]
    total_vram_gb: Optional[float]
    nvidia_driver: Optional[str]


def detect() -> HardwareInfo:
    """Probe for CUDA without importing torch at module load time.

    Returns a best-effort :class:`HardwareInfo`. GPU enumeration requires
    ``nvidia-smi``; when absent we report CPU-only without error.
    """
    names: list[str] = []
    total_vram: Optional[float] = None
    has_cuda = "CUDA" in os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        smi = shutil.which("nvidia-smi")
        if smi:
            out = os.popen(f'"{smi}" --query-gpu=name,memory.total --format=csv,noheader,nounits')  # noqa: S605
            lines = [ln.strip() for ln in out.read().splitlines() if ln.strip()]
            out.close()
            for ln in lines:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) == 2:
                    names.append(parts[0])
                    total_vram = float(parts[1]) / 1024.0 if total_vram is None else total_vram
    except Exception:  # pragma: no cover - never fail a command over probing
        names = []

    driver = None
    try:
        nv = shutil.which("nvidia-smi")
        if nv:
            driver = os.popen(f'"{nv}" --query-gpu=driver_version --format=csv,noheader').read().strip()  # noqa: S605
    except Exception:  # pragma: no cover
        driver = None

    return HardwareInfo(
        has_cuda=bool(names) or has_cuda,
        device_count=len(names),
        device_names=names,
        total_vram_gb=total_vram,
        nvidia_driver=(driver or None),
    )


def suggest_quantization(hw: HardwareInfo) -> str:
    """Suggest ``qlora`` for <= 12 GB VRAM, else ``lora``."""
    if hw.total_vram_gb is not None and hw.total_vram_gb <= 12:
        return "qlora"
    return "lora"


def has_nvidia_gpu() -> bool:
    return detect().device_count > 0
