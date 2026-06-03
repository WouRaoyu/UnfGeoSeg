"""UnfavorSeg: weakly-supervised coarse-to-fine 3D segmentation of unfavorable
tunnel geology, built on nnU-Net (data/training/inference) and 3D-TransUNet
(fine-grained backbone).

Importing this package also registers the custom nnU-Net trainer
``nnUNetTrainerTransUNetCC`` so nnU-Net can discover it via ``-tr``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "load_config", "Config"]

from .config import Config, load_config  # noqa: E402
