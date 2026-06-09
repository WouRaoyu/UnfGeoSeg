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


class ConfidenceWeightedDiceCELoss(nn.Module):
    """Dice+CE where each voxel is weighted by the coarse-label confidence.

    For a binary hard pseudo-label, the confidence of the assigned label is
    ``probfg`` for foreground voxels and ``1 - probfg`` for background voxels.
    This is the loss to use when the coarse probability should control how much
    each hard pseudo-label contributes to training, instead of adding a separate
    soft-distribution KL term.
    """

    def __init__(
        self,
        num_classes: int,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        min_confidence_weight: float = 0.05,
        confidence_power: float = 1.0,
        bg_weight: float = 1.0,
        fg_weight: float = 1.0,
        include_bg_dice: bool = True,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("confidence-weighted loss currently supports binary runs")
        self.num_classes = num_classes
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.min_confidence_weight = float(min_confidence_weight)
        self.confidence_power = float(confidence_power)
        self.bg_weight = float(bg_weight)
        self.fg_weight = float(fg_weight)
        self.include_bg_dice = bool(include_bg_dice)

    @staticmethod
    def _target_indices(target: torch.Tensor, net_output: torch.Tensor) -> torch.Tensor:
        if target.ndim == net_output.ndim:
            target = target[:, 0]
        return target.long()

    def voxel_weights(self, target: torch.Tensor, probfg: torch.Tensor) -> torch.Tensor:
        if probfg.ndim >= 2 and probfg.shape[1] == 1:
            probfg = probfg[:, 0]
        fg = probfg.clamp(0.0, 1.0)
        target_fg = target > 0
        confidence = torch.where(target_fg, fg, 1.0 - fg)
        confidence = confidence.clamp(0.0, 1.0)
        if self.confidence_power != 1.0:
            confidence = confidence.pow(self.confidence_power)
        confidence = self.min_confidence_weight + (
            1.0 - self.min_confidence_weight
        ) * confidence
        class_weight = torch.where(
            target_fg,
            torch.as_tensor(self.fg_weight, device=target.device, dtype=confidence.dtype),
            torch.as_tensor(self.bg_weight, device=target.device, dtype=confidence.dtype),
        )
        return confidence * class_weight

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        probfg: torch.Tensor,
    ) -> torch.Tensor:
        target_idx = self._target_indices(target, net_output)
        weights = self.voxel_weights(target_idx, probfg).to(net_output.dtype)

        ce_map = F.cross_entropy(net_output, target_idx, reduction="none")
        ce = (ce_map * weights).mean()

        probs = torch.softmax(net_output, dim=1)
        oh = F.one_hot(target_idx, self.num_classes)
        oh = oh.permute(0, -1, *range(1, oh.ndim - 1)).to(probs.dtype)
        w = weights[:, None]
        dims = tuple(range(2, net_output.ndim))
        inter = (w * probs * oh).sum(dims)
        pred = (w * probs).sum(dims)
        tgt = (w * oh).sum(dims)
        dice_score = (2.0 * inter + 1e-5) / (pred + tgt + 1e-5)
        class_weights = torch.as_tensor(
            [self.bg_weight, self.fg_weight], device=net_output.device, dtype=net_output.dtype
        )
        if self.include_bg_dice:
            dice_loss = 1.0 - dice_score
            selected_weights = class_weights
        else:
            dice_loss = 1.0 - dice_score[:, 1:]
            selected_weights = class_weights[1:]
        dice = (dice_loss * selected_weights).sum() / (
            dice_loss.shape[0] * selected_weights.sum().clamp_min(1e-8)
        )
        dice = dice * weights.mean()
        return self.weight_ce * ce + self.weight_dice * dice


class ConfidenceConstrainedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lambda_kl: float = 0.3,
        base_loss=None,
        confidence_weighted: bool = False,
        min_confidence_weight: float = 0.05,
        confidence_power: float = 1.0,
        bg_weight: float = 1.0,
        fg_weight: float = 1.0,
        include_bg_dice: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes  # includes background
        self.lambda_kl = float(lambda_kl)
        self.base_loss = base_loss if base_loss is not None else _build_base_loss()
        self.confidence_weighted = bool(confidence_weighted)
        self.weighted_base = (
            ConfidenceWeightedDiceCELoss(
                num_classes=num_classes,
                min_confidence_weight=min_confidence_weight,
                confidence_power=confidence_power,
                bg_weight=bg_weight,
                fg_weight=fg_weight,
                include_bg_dice=include_bg_dice,
            )
            if self.confidence_weighted
            else None
        )

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
        if self.weighted_base is not None and probfg is not None:
            base = self.weighted_base(net_output, hard_target, probfg)
        else:
            base = self.base_loss(net_output, hard_target)
        if probfg is None or self.lambda_kl == 0.0:
            return base
        P = self.soft_target(probfg)
        logQ = F.log_softmax(net_output, dim=1)
        # KL(P || Q) = sum_k P_k (log P_k - log Q_k), averaged over voxels/batch
        kl = (P * (torch.log(P.clamp_min(1e-8)) - logQ)).sum(dim=1).mean()
        return base + self.lambda_kl * kl
