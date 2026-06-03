"""Training-curve extraction and plotting utilities."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return [float(v) for v in list(values)]


def _mean_dice(values: Any) -> list[float]:
    rows = values.tolist() if isinstance(values, np.ndarray) else list(values or [])
    means: list[float] = []
    for row in rows:
        arr = np.asarray(row, dtype=float)
        if arr.size:
            means.append(float(np.nanmean(arr)))
    return means


def _load_checkpoint_curves(run_dir: Path) -> dict[str, list[float]]:
    checkpoint_paths = [
        run_dir / "checkpoint_final.pth",
        run_dir / "checkpoint_best.pth",
        run_dir / "checkpoint_latest.pth",
    ]
    checkpoint_paths.extend(sorted(run_dir.glob("checkpoint_*.pth")))
    for path in checkpoint_paths:
        if not path.exists():
            continue
        try:
            import torch

            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue

        curves: dict[str, list[float]] = {}
        logging = ckpt.get("logging") if isinstance(ckpt, dict) else None
        if logging is not None:
            for attr, name in (
                ("train_losses", "train_loss"),
                ("val_losses", "val_loss"),
                ("mean_fg_dice", "mean_fg_dice"),
                ("ema_fg_dice", "ema_fg_dice"),
            ):
                curves[name] = _as_float_list(getattr(logging, attr, None))

        plot_stuff = ckpt.get("plot_stuff") if isinstance(ckpt, dict) else None
        if isinstance(plot_stuff, (list, tuple)) and len(plot_stuff) >= 4:
            curves.setdefault("train_loss", _as_float_list(plot_stuff[0]))
            curves.setdefault("val_loss", _as_float_list(plot_stuff[1]))
            curves.setdefault("val_loss_train_mode", _as_float_list(plot_stuff[2]))
            curves.setdefault("mean_fg_dice", _mean_dice(plot_stuff[3]))

        if any(curves.values()):
            return curves
    return {}


def _load_log_curves(run_dir: Path) -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {}
    patterns = {
        "train_loss": re.compile(r"(?:train(?:ing)? loss|train_loss)\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.I),
        "val_loss": re.compile(r"(?:validation loss|val_loss)\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.I),
        "mean_fg_dice": re.compile(r"(?:mean_fg_dice|mean foreground dice|pseudo dice)\D+(-?\d+(?:\.\d+)?)", re.I),
    }
    log_files = sorted(run_dir.glob("*.txt")) + sorted(run_dir.glob("*.log"))
    for path in log_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            values = [float(match.group(1)) for match in pattern.finditer(text)]
            if values:
                curves.setdefault(name, []).extend(values)
    return curves


def load_curves(run_dir: str | Path) -> dict[str, list[float]]:
    """Load curves from an nnU-Net output folder."""
    run_dir = Path(run_dir)
    curves = _load_checkpoint_curves(run_dir)
    log_curves = _load_log_curves(run_dir)
    for name, values in log_curves.items():
        curves.setdefault(name, values)
    return curves


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return values
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(values[lo : i + 1])))
    return out


def write_csv(curves: dict[str, list[float]], out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(curves)
    n = max((len(v) for v in curves.values()), default=0)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", *names])
        writer.writeheader()
        for i in range(n):
            row = {"epoch": i + 1}
            for name in names:
                values = curves[name]
                row[name] = values[i] if i < len(values) else ""
            writer.writerow(row)
    return out_csv


def plot_curves(curves: dict[str, list[float]], out_png: str | Path, smooth: int = 5) -> Path:
    if not any(curves.values()):
        raise ValueError("no training curves found")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    loss_names = [name for name in curves if "loss" in name]
    dice_names = [name for name in curves if "dice" in name]
    fig, axes = plt.subplots(1, 2 if dice_names else 1, figsize=(12 if dice_names else 7, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for name in loss_names:
        values = curves[name]
        axes[0].plot(range(1, len(values) + 1), values, alpha=0.28, linewidth=1)
        axes[0].plot(range(1, len(values) + 1), _smooth(values, smooth), label=name, linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if dice_names:
        for name in dice_names:
            values = curves[name]
            axes[1].plot(range(1, len(values) + 1), values, alpha=0.28, linewidth=1)
            axes[1].plot(range(1, len(values) + 1), _smooth(values, smooth), label=name, linewidth=2)
        axes[1].set_title("Dice")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.25)
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def run(run_dir: str | Path, out_dir: str | Path | None = None, smooth: int = 5) -> dict[str, Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    curves = load_curves(run_dir)
    csv_path = write_csv(curves, out_dir / "training_curves.csv")
    png_path = plot_curves(curves, out_dir / "training_curves.png", smooth=smooth)
    return {"csv": csv_path, "png": png_path}
