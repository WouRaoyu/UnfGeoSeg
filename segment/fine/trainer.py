"""Custom nnU-Net v2 trainers for the fine-grained stage.

* ``nnUNetTrainerTransUNet``   -- plain 3D-TransUNet baseline (swaps the network
  architecture, standard nnU-Net loss). Used for the uncertainty comparison.
* ``nnUNetTrainerTransUNetCC`` -- the proposed method: 3D-TransUNet trained with
  the probability-constrained loss ``L = L' + lambda * KL(P || Q)``.

The coarse foreground probability (``probfg``) is carried as an extra image
channel only so nnU-Net crops/augments it in lockstep with the target. The
network wrapper drops that last channel before the forward pass; ``train_step`` /
``validation_step`` read it only for the KL term.

Deep supervision is disabled so the single full-resolution output stays aligned
with the full-resolution probfg soft-target map.
"""

from __future__ import annotations

import os

import torch
from torch import autocast, nn

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from .loss import ConfidenceConstrainedLoss
from .transunet_wrapper import build_transunet


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
        net = build_transunet(image_channels, num_output_channels, arch_init_kwargs)
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
        net = build_transunet(image_channels, num_output_channels, arch_init_kwargs)
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
