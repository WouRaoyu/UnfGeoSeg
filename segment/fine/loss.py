"""Fine-stage losses.

The original probability-constrained loss follows Manuscript Eq. 5-6:

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

The weak fine-stage variant below deliberately treats ``probfg`` as uncertain
weak supervision rather than ground truth. It down-weights voxels whose
foreground probability is close to 0.5, adds only a small scheduled KL term, and
exposes consistency / edge-aware smoothness helpers used by the weak trainer.
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


def confidence_weight_from_probfg(
    probfg: torch.Tensor,
    confidence_floor: float = 0.1,
) -> torch.Tensor:
    """Map foreground probability to weak-supervision voxel weights.

    ``probfg=0.5`` is maximally uncertain and receives ``confidence_floor``;
    ``probfg`` near 0 or 1 receives weight 1.0. The returned tensor has shape
    ``(B, ...)`` and is detached because it is a supervision confidence signal,
    not a learnable path.
    """
    if probfg.ndim >= 2 and probfg.shape[1] == 1:
        probfg = probfg[:, 0]
    floor = float(confidence_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1]")
    certainty = (2.0 * (probfg.float().clamp(0.0, 1.0) - 0.5).abs()).clamp(0.0, 1.0)
    return (floor + (1.0 - floor) * certainty).detach()


def _target_without_channel(
    target: torch.Tensor,
    net_output_ndim: int | None = None,
) -> torch.Tensor:
    if (
        target.ndim >= 2
        and target.shape[1] == 1
        and (net_output_ndim is None or target.ndim == net_output_ndim)
    ):
        target = target[:, 0]
    return target.long()


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


class WeakConfidenceConstrainedLoss(nn.Module):
    """Low-pressure soft supervision for pseudo-label based fine training.

    The hard pseudo-label still contributes CE, while the coarse foreground
    probability defines a soft Dice and optional KL target. All terms are
    voxel-weighted by pseudo-label certainty so low-confidence / mixed regions
    are allowed to deviate from the coarse label.
    """

    def __init__(
        self,
        num_classes: int,
        lambda_kl: float = 0.0,
        confidence_floor: float = 0.1,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("weak confidence loss requires an independent binary run")
        self.num_classes = int(num_classes)
        self.lambda_kl = float(lambda_kl)
        self.confidence_floor = float(confidence_floor)
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)

    @staticmethod
    def soft_target(probfg: torch.Tensor) -> torch.Tensor:
        if probfg.ndim >= 2 and probfg.shape[1] == 1:
            probfg = probfg[:, 0]
        fg = probfg.float().clamp(1e-4, 1.0 - 1e-4)
        return torch.stack((1.0 - fg, fg), dim=1)

    def forward(
        self,
        net_output: torch.Tensor,
        hard_target: torch.Tensor,
        probfg: torch.Tensor,
    ) -> torch.Tensor:
        target = _target_without_channel(hard_target, net_output.ndim)
        weights = confidence_weight_from_probfg(probfg, self.confidence_floor)
        probs = torch.softmax(net_output, dim=1)
        soft = self.soft_target(probfg)

        ce = F.cross_entropy(net_output, target, reduction="none")
        ce = _weighted_mean(ce, weights)

        dims = tuple(range(2, net_output.ndim))
        w = weights[:, None]
        inter = (probs * soft * w).sum(dims)
        denom = ((probs + soft) * w).sum(dims)
        dice = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5))[:, 1:].mean()

        loss = self.weight_ce * ce + self.weight_dice * dice
        if self.lambda_kl:
            logq = F.log_softmax(net_output, dim=1)
            kl = (soft * (torch.log(soft.clamp_min(1e-8)) - logq)).sum(dim=1)
            loss = loss + self.lambda_kl * _weighted_mean(kl, weights)
        return loss


def consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    confidence_threshold: float = 0.75,
) -> torch.Tensor:
    """Masked MSE consistency on probability maps from an EMA teacher."""
    with torch.no_grad():
        teacher_prob = torch.softmax(teacher_logits, dim=1)
        teacher_conf = teacher_prob.max(dim=1).values
        mask = teacher_conf >= float(confidence_threshold)
    if not torch.any(mask):
        return student_logits.sum() * 0.0
    student_prob = torch.softmax(student_logits, dim=1)
    diff = (student_prob - teacher_prob).pow(2).sum(dim=1)
    return diff[mask].mean()


def edge_aware_total_variation(
    foreground_prob: torch.Tensor,
    image: torch.Tensor,
    edge_aware: bool = True,
    edge_sensitivity: float = 5.0,
) -> torch.Tensor:
    """Local smoothness regularizer that relaxes across strong physical edges.

    ``foreground_prob`` is ``(B, D, H, W)`` or ``(B, 1, D, H, W)``. ``image`` is
    the physical input tensor ``(B, C, D, H, W)``; if a probfg carrier channel is
    present, callers should pass only the physical channels.
    """
    if foreground_prob.ndim == image.ndim:
        foreground_prob = foreground_prob[:, 0]
    if image.ndim != 5:
        raise ValueError("image must have shape (B, C, D, H, W)")

    losses = []
    axes = (2, 3, 4)
    for axis in axes:
        f_axis = axis - 1
        if foreground_prob.shape[f_axis] <= 1:
            continue
        f_hi = foreground_prob.diff(dim=f_axis).abs()
        if edge_aware:
            grad = image.diff(dim=axis).abs().mean(dim=1)
            scale = grad.detach().mean().clamp_min(1e-6)
            edge_weight = torch.exp(-float(edge_sensitivity) * grad / scale)
            losses.append((f_hi * edge_weight).mean())
        else:
            losses.append(f_hi.mean())
    if not losses:
        return foreground_prob.sum() * 0.0
    return sum(losses) / len(losses)
