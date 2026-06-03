"""Configuration loading for UnfavorSeg.

A thin typed wrapper around the YAML config so the rest of the codebase does not
sprinkle ``cfg["a"]["b"]`` lookups everywhere. Unknown keys are preserved on the
raw mapping (``Config.raw``) so experimental scripts can read extra fields.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "geology.yaml"


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    @property
    def channels(self) -> List[str]:
        return list(self.raw.get("channels", ["vp", "vs", "depth"]))

    @property
    def classes(self) -> List[str]:
        """Foreground geology types.

        Each type is an INDEPENDENT 0/1 binary target (types may spatially
        overlap), not a level of a single mutually-exclusive label. For a
        per-class run this is narrowed to a single ``[<type>]`` (see
        ``segment.cli.effective_config``).
        """
        return list(self.raw.get("classes", ["unfavorable"]))

    @property
    def geology_classes(self) -> List[str]:
        """All independent geology types declared for the source dataset.

        Defaults to :attr:`classes`; the source dataset.json may also declare a
        ``geology_classes`` field which takes precedence when present (set by
        :func:`segment.cli.effective_config`).
        """
        return list(self.raw.get("geology_classes", self.classes))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def half_window(self) -> List[int]:
        return list(self.raw.get("sampling", {}).get("half_window", [4, 4, 4]))

    @property
    def axial_axis(self) -> int:
        return int(self.raw.get("sampling", {}).get("axial_axis", 0))

    @property
    def statistics(self) -> List[str]:
        return list(
            self.raw.get("sampling", {}).get(
                "statistics", ["mean", "median", "mode", "max", "min"]
            )
        )

    @property
    def mode_decimals(self) -> int:
        return int(self.raw.get("sampling", {}).get("mode_decimals", 2))

    @property
    def lambda_kl(self) -> float:
        return float(self.raw.get("fine", {}).get("lambda_kl", 0.3))

    @property
    def fine_epochs(self) -> int:
        return int(self.raw.get("fine", {}).get("epochs", 100))

    @property
    def lambda_sweep(self) -> List[float]:
        return list(self.raw.get("fine", {}).get("lambda_sweep", [0.0, 0.1, 0.3, 0.5, 1.0]))

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_config(path: Optional[os.PathLike | str] = None) -> Config:
    """Load the YAML config. Defaults to ``configs/geology.yaml``."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw=raw)
