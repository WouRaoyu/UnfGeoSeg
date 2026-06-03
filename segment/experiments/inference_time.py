"""Computational-efficiency table (per TSP volume).

Times each pipeline stage and records GPU memory. Stages can be passed as
callables (so ``run_all`` can wrap the real nnU-Net commands) or the module can
simply time an external ``nnUNetv2_predict`` invocation via :func:`time_command`.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .report import write_table


def _gpu_mem_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return None


def time_callable(fn: Callable[[], None]) -> float:
    t = time.time()
    fn()
    return time.time() - t


def time_command(cmd: List[str]) -> float:
    t = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t


def run(
    stage_times: Dict[str, float],
    hardware: str = "RTX 3060 12GB / Xeon",
    voxel_patch: str = "128^3 patch",
    n_volumes: int = 1,
    out_dir="results/inference_time",
) -> List[Dict]:
    """Assemble the efficiency table from measured ``stage_times`` (seconds)."""
    rows: List[Dict] = []
    total = 0.0
    for stage, secs in stage_times.items():
        rows.append({
            "Stage": stage,
            "Time (s)": secs / max(n_volumes, 1),
            "Hardware": hardware,
            "Voxel/patch setting": voxel_patch,
            "GPU mem (MB)": _gpu_mem_mb(),
        })
        total += secs
    rows.append({
        "Stage": "Total per volume",
        "Time (s)": total / max(n_volumes, 1),
        "Hardware": hardware,
        "Voxel/patch setting": voxel_patch,
        "GPU mem (MB)": _gpu_mem_mb(),
    })
    write_table(rows, Path(out_dir) / "inference_time",
                columns=["Stage", "Time (s)", "Hardware", "Voxel/patch setting", "GPU mem (MB)"])
    return rows
