"""Probability-constrained loss (Manuscript Eq. 5-6).

``L = L' + lambda * D_KL(P || Q)``

* ``L'`` is the standard segmentation loss (Dice + cross-entropy).
* ``P`` is the binary soft pseudo-label distribution reconstructed directly
  from the coarse foreground probability ``p_fg``:

      P[background] = 1 - p_fg,   P[foreground] = p_fg

* ``Q = softmax(net_output)`` is the network prediction.

The KL term uses the full coarse binary probability instead of first collapsing
it to ``hard label + confidence``. With ``lambda = 0`` the loss reduces exactly
to ``L'`` (used as the no-constraint baseline in the lambda-sensitivity
experiment).

Deep supervision is disabled for this trainer so a single full-resolution
output/target/probfg triple is used (keeps the probability map aligned without
multi-resolution downsampling of the soft labels).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_base_loss():
    """Prefer nnU-Net's combined Dice+CE loss; fall back to a local equivalent
    so the loss is importable/testable without nnU-Net installed."""
    try:
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

        return DC_and_CE_loss(
            soft_dice_kwargs=dict(
                batch_dice=True, smooth=1e-5, do_bg=False, ddp=False
            ),
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
    except Exception:  # pragma: no cover - exercised only without nnU-Net
        return _LocalDiceCE()


class _LocalDiceCE(nn.Module):
    """Minimal Dice+CE fallback. ``target`` is (B,1,...) integer labels."""

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == net_output.ndim:
            target = target[:, 0]
        ce = F.cross_entropy(net_output, target.long())
        probs = torch.softmax(net_output, dim=1)
        oh = F.one_hot(target.long(), net_output.shape[1])
        oh = oh.permute(0, -1, *range(1, oh.ndim - 1)).float()
        dims = tuple(range(2, net_output.ndim))
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        dice = 1 - ((2 * inter + 1e-5) / (denom + 1e-5))[:, 1:].mean()
        return ce + dice


class ConfidenceConstrainedLoss(nn.Module):
    def __init__(self, num_classes: int, lambda_kl: float = 0.3, base_loss=None):
        super().__init__()
        self.num_classes = num_classes  # includes background
        self.lambda_kl = float(lambda_kl)
        self.base_loss = base_loss if base_loss is not None else _build_base_loss()

    def soft_target(self, probfg: torch.Tensor) -> torch.Tensor:
        """Reconstruct binary P (B, 2, ...) directly from foreground probability."""
        if self.num_classes != 2:
            raise ValueError("probfg soft targets require an independent binary run")
        if probfg.ndim >= 2 and probfg.shape[1] == 1:
            probfg = probfg[:, 0]
        fg = probfg.clamp(1e-4, 1 - 1e-4)
        return torch.stack((1.0 - fg, fg), dim=1)

    def forward(
        self,
        net_output: torch.Tensor,
        hard_target: torch.Tensor,
        probfg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.base_loss(net_output, hard_target)
        if probfg is None or self.lambda_kl == 0.0:
            return base
        P = self.soft_target(probfg)
        logQ = F.log_softmax(net_output, dim=1)
        # KL(P || Q) = sum_k P_k (log P_k - log Q_k), averaged over voxels/batch
        kl = (P * (torch.log(P.clamp_min(1e-8)) - logQ)).sum(dim=1).mean()
        return base + self.lambda_kl * kl
