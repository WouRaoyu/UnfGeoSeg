"""Custom nnU-Net v2 trainers for the fine-grained stage.

* ``nnUNetTrainerTransUNet``   -- plain 3D-TransUNet baseline (swaps the network
  architecture, standard nnU-Net loss). Used for the uncertainty comparison.
* ``nnUNetTrainerTransUNetCC`` -- the proposed method: 3D-TransUNet trained with
  the probability-constrained loss ``L = L' + lambda * KL(P || Q)``.
* ``nnUNetTrainerTransUNetWeakCC`` -- weak pseudo-label training: low-pressure
  soft supervision, EMA consistency, and edge-aware spatial regularization.

The coarse foreground probability (``probfg``) is carried as an extra image
channel only so nnU-Net crops/augments it in lockstep with the target. The
network wrapper drops that last channel before the forward pass; ``train_step`` /
``validation_step`` read it only for the KL term.

Deep supervision is disabled so the single full-resolution output stays aligned
with the full-resolution probfg soft-target map.
"""

from __future__ import annotations

import copy
import os

import torch
from torch import autocast, nn

try:
    from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.helpers import dummy_context
except Exception:  # pragma: no cover - exercised only without nnU-Net installed
    from contextlib import nullcontext as dummy_context

    def get_tp_fp_fn_tn(*args, **kwargs):
        raise ImportError("nnU-Net is required for trainer validation metrics")

    class nnUNetTrainer(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError("nnU-Net is required to instantiate UnfavorSeg trainers")

        @staticmethod
        def build_network_architecture(*args, **kwargs):
            raise ImportError("nnU-Net is required to build the baseline trainer")

from .loss import (
    ConfidenceConstrainedLoss,
    WeakConfidenceConstrainedLoss,
    consistency_loss,
    edge_aware_total_variation,
)
from .transunet_wrapper import build_transunet


def _arch_with_segment_options(arch_init_kwargs: dict | None) -> dict:
    arch = dict(arch_init_kwargs or {})
    arch.setdefault(
        "positional_encoding",
        os.environ.get("UNFAVORSEG_POS_ENCODING", "sinusoidal_3d"),
    )
    return arch


def _linear_warmup(max_value: float, epoch: int, warmup_epochs: int) -> float:
    max_value = float(max_value)
    if max_value <= 0:
        return 0.0
    warmup_epochs = max(int(warmup_epochs), 1)
    return max_value * min(1.0, max(float(epoch + 1), 0.0) / warmup_epochs)


class _UnfavorSegEpochsMixin:
    default_num_epochs = 100

    def _set_segment_num_epochs(self) -> None:
        num_epochs = int(os.environ.get("UNFAVORSEG_EPOCHS", self.default_num_epochs))
        self.num_epochs = num_epochs
        if hasattr(self, "max_num_epochs"):
            self.max_num_epochs = num_epochs
        if "UNFAVORSEG_LR" in os.environ:
            self.initial_lr = float(os.environ["UNFAVORSEG_LR"])
        if "UNFAVORSEG_TRAIN_ITERS" in os.environ:
            self.num_iterations_per_epoch = int(os.environ["UNFAVORSEG_TRAIN_ITERS"])
        if "UNFAVORSEG_VAL_ITERS" in os.environ:
            self.num_val_iterations_per_epoch = int(os.environ["UNFAVORSEG_VAL_ITERS"])


class nnUNetTrainerUnfavorSeg(_UnfavorSegEpochsMixin, nnUNetTrainer):
    """Standard nnU-Net baseline with the UnfavorSeg training schedule."""

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._set_segment_num_epochs()

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        image_channels = num_input_channels - 1 if num_input_channels == 4 else num_input_channels
        net = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            image_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return _DropProbfgChannel(net, image_channels) if image_channels != num_input_channels else net


class _DropProbfgChannel(nn.Module):
    """Accept nnU-Net's 4-channel batch but expose only physical channels to net."""

    def __init__(self, net: nn.Module, image_channels: int):
        super().__init__()
        self.net = net
        self.image_channels = image_channels

    @property
    def decoder(self):
        """Expose the wrapped decoder for nnU-Net deep-supervision toggles."""
        return self.net.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, : self.image_channels])


class nnUNetTrainerTransUNet(_UnfavorSegEpochsMixin, nnUNetTrainer):
    """Plain 3D-TransUNet baseline (no probability constraint)."""

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._set_segment_num_epochs()
        # single full-resolution head keeps the probfg soft-target map aligned
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        image_channels = num_input_channels - 1 if num_input_channels == 4 else num_input_channels
        net = build_transunet(
            image_channels,
            num_output_channels,
            _arch_with_segment_options(arch_init_kwargs),
        )
        return _DropProbfgChannel(net, image_channels) if image_channels != num_input_channels else net


class nnUNetTrainerTransUNetCC(nnUNetTrainerTransUNet):
    """Proposed: 3D-TransUNet + probability-constrained loss."""

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.lambda_kl = float(os.environ.get("UNFAVORSEG_LAMBDA", "0.3"))

    def _build_loss(self):
        return ConfidenceConstrainedLoss(
            num_classes=self.label_manager.num_segmentation_heads,
            lambda_kl=self.lambda_kl,
        )

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if num_input_channels < 4:
            raise ValueError(
                "nnUNetTrainerTransUNetCC requires [vp, vs, depth, probfg] input channels"
            )
        image_channels = num_input_channels - 1
        net = build_transunet(
            image_channels,
            num_output_channels,
            _arch_with_segment_options(arch_init_kwargs),
        )
        return _DropProbfgChannel(net, image_channels)

    # -- training/validation with the probfg carrier channel ------------------
    @staticmethod
    def _split_probfg(data: torch.Tensor):
        """Last input channel is probfg; the network wrapper drops it."""
        if data.shape[1] < 4:
            raise ValueError("CC training batches must include a final probfg channel")
        return data, data[:, -1:]

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)
        net_in, probfg = self._split_probfg(data)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(net_in)
            l = self.loss(output, target, probfg)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)
        net_in, probfg = self._split_probfg(data)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(net_in)
            del data
            l = self.loss(output, target, probfg)

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_onehot.scatter_(1, output_seg, 1)
            del output_seg

        mask = None
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]
        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard,
                "fp_hard": fp_hard, "fn_hard": fn_hard}


class nnUNetTrainerTransUNetWeakCC(nnUNetTrainerTransUNetCC):
    """Weak fine-stage trainer for uncertain pseudo-label supervision.

    The network still sees only physical channels. ``probfg`` is used to weight
    soft supervision, while EMA consistency and edge-aware TV encourage stable
    spatial structure without forcing the model to copy uncertain pseudo-label
    boundaries.
    """

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.confidence_floor = float(os.environ.get("UNFAVORSEG_CONFIDENCE_FLOOR", "0.1"))
        self.kl_max = float(os.environ.get("UNFAVORSEG_WEAK_KL_MAX", "0.1"))
        self.weak_warmup_epochs = int(os.environ.get("UNFAVORSEG_WEAK_WARMUP_EPOCHS", "20"))
        self.consistency_enabled = os.environ.get("UNFAVORSEG_CONSISTENCY", "1") != "0"
        self.consistency_weight_max = float(os.environ.get("UNFAVORSEG_CONSISTENCY_WEIGHT", "0.2"))
        self.consistency_threshold = float(os.environ.get("UNFAVORSEG_CONSISTENCY_CONF", "0.75"))
        self.ema_decay = float(os.environ.get("UNFAVORSEG_EMA_DECAY", "0.99"))
        self.tv_weight = float(os.environ.get("UNFAVORSEG_EDGE_TV_WEIGHT", "0.02"))
        self.edge_aware_tv = os.environ.get("UNFAVORSEG_EDGE_AWARE_TV", "1") != "0"
        self.perturb_std = float(os.environ.get("UNFAVORSEG_CONSISTENCY_NOISE_STD", "0.03"))
        self.perturb_scale = float(os.environ.get("UNFAVORSEG_CONSISTENCY_SCALE_STD", "0.05"))
        self.ema_teacher = None

    def _build_loss(self):
        return WeakConfidenceConstrainedLoss(
            num_classes=self.label_manager.num_segmentation_heads,
            lambda_kl=0.0,
            confidence_floor=self.confidence_floor,
        )

    def _scheduled_kl(self) -> float:
        return _linear_warmup(
            self.kl_max,
            int(getattr(self, "current_epoch", 0)),
            self.weak_warmup_epochs,
        )

    def _scheduled_consistency_weight(self) -> float:
        return _linear_warmup(
            self.consistency_weight_max,
            int(getattr(self, "current_epoch", 0)),
            self.weak_warmup_epochs,
        )

    def _jitter_physical(self, data: torch.Tensor) -> torch.Tensor:
        if self.perturb_std <= 0.0 and self.perturb_scale <= 0.0:
            return data
        out = data.clone()
        phys = out[:, :-1]
        if self.perturb_scale > 0.0:
            scale_shape = (phys.shape[0], phys.shape[1], 1, 1, 1)
            scale = 1.0 + torch.randn(
                scale_shape, device=phys.device, dtype=phys.dtype
            ) * self.perturb_scale
            phys.mul_(scale)
        if self.perturb_std > 0.0:
            phys.add_(torch.randn_like(phys) * self.perturb_std)
        return out

    def _source_network(self) -> nn.Module:
        return self.network.module if hasattr(self.network, "module") else self.network

    def _ensure_ema_teacher(self) -> nn.Module:
        if self.ema_teacher is None:
            self.ema_teacher = copy.deepcopy(self._source_network())
            self.ema_teacher.to(self.device)
            self.ema_teacher.eval()
            for p in self.ema_teacher.parameters():
                p.requires_grad_(False)
        return self.ema_teacher

    @torch.no_grad()
    def _update_ema_teacher(self) -> None:
        teacher = self._ensure_ema_teacher()
        source = self._source_network()
        decay = float(self.ema_decay)
        for t_param, s_param in zip(teacher.parameters(), source.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)
        for t_buf, s_buf in zip(teacher.buffers(), source.buffers()):
            if torch.is_floating_point(t_buf):
                t_buf.data.mul_(decay).add_(s_buf.data, alpha=1.0 - decay)
            else:
                t_buf.data.copy_(s_buf.data)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)
        _, probfg = self._split_probfg(data)
        self.loss.lambda_kl = self._scheduled_kl()

        student_data = self._jitter_physical(data)
        teacher_data = self._jitter_physical(data)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(student_data)
            supervised = self.loss(output, target, probfg)

            consistency = output.sum() * 0.0
            cons_weight = self._scheduled_consistency_weight() if self.consistency_enabled else 0.0
            if cons_weight > 0.0:
                teacher = self._ensure_ema_teacher()
                with torch.no_grad():
                    teacher_output = teacher(teacher_data)
                consistency = consistency_loss(
                    output,
                    teacher_output,
                    confidence_threshold=self.consistency_threshold,
                )

            tv = output.sum() * 0.0
            if self.tv_weight > 0.0:
                fg = torch.softmax(output, dim=1)[:, 1]
                tv = edge_aware_total_variation(
                    fg,
                    student_data[:, :-1],
                    edge_aware=self.edge_aware_tv,
                )
            l = supervised + cons_weight * consistency + self.tv_weight * tv

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        self._update_ema_teacher()
        return {
            "loss": l.detach().cpu().numpy(),
            "loss_supervised": supervised.detach().cpu().numpy(),
            "loss_consistency": consistency.detach().cpu().numpy(),
            "loss_tv": tv.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        self.loss.lambda_kl = self._scheduled_kl()
        return super().validation_step(batch)
