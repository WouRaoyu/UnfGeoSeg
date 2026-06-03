"""3D-TransUNet backbone adapted to nnU-Net v2's ``build_network_architecture``
contract.

The fine-grained backbone follows the 3D-TransUNet design (Beckschen et al.):
a convolutional U-Net encoder/decoder with a Transformer operating on the
bottleneck feature map for global context. To guarantee that any nnU-Net patch
size / plan works, :class:`TransUNet3D` mirrors the stage/stride/feature layout
of nnU-Net's ``PlainConvUNet`` (read from the plan's ``arch_kwargs``) and simply
inserts a Transformer encoder at the bottleneck.

Deep supervision is intentionally disabled (single full-resolution head) so the
confidence-constrained loss stays aligned with the pseudo-label confidence map
(see ``segment.fine.loss``).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tuple3(s) -> Tuple[int, int, int]:
    if isinstance(s, int):
        return (s, s, s)
    return tuple(int(v) for v in s)  # type: ignore[return-value]


def _pick_heads(channels: int, preferred: int = 8) -> int:
    for h in (preferred, 8, 5, 4, 2, 1):
        if channels % h == 0:
            return h
    return 1


class StageBlock(nn.Module):
    """``n_conv`` Conv-IN-LReLU convs; the first conv carries the stride
    (down-sampling within the block, as in nnU-Net's PlainConvUNet)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: Tuple[int, int, int],
        n_conv: int = 2,
        kernel: int = 3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(max(1, n_conv)):
            s = stride if i == 0 else (1, 1, 1)
            layers += [
                nn.Conv3d(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    kernel,
                    stride=s,
                    padding=kernel // 2,
                    bias=False,
                ),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.LeakyReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransformerBottleneck(nn.Module):
    """Transformer encoder over the flattened bottleneck voxels."""

    def __init__(self, channels: int, depth: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        heads = _pick_heads(channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, 1, channels))  # size-agnostic bias
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


class TransUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features_per_stage: Sequence[int],
        strides: Sequence[Sequence[int]],
        n_conv_per_stage: Sequence[int] | int = 2,
        kernel_sizes: Sequence[int] | int = 3,
        transformer_depth: int = 4,
    ):
        super().__init__()
        n_stages = len(features_per_stage)
        strides = [_as_tuple3(s) for s in strides]
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        kern = kernel_sizes if isinstance(kernel_sizes, int) else int(kernel_sizes[0][0])

        # encoder
        self.encoder = nn.ModuleList()
        prev = in_channels
        for i, f in enumerate(features_per_stage):
            self.encoder.append(
                StageBlock(prev, f, strides[i], n_conv_per_stage[i], kern)
            )
            prev = f

        self.transformer = TransformerBottleneck(
            features_per_stage[-1], depth=transformer_depth
        )

        # decoder (n_stages - 1 up steps)
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(
                    features_per_stage[i],
                    features_per_stage[i - 1],
                    kernel_size=strides[i],
                    stride=strides[i],
                )
            )
            self.decoder.append(
                StageBlock(
                    features_per_stage[i - 1] * 2,
                    features_per_stage[i - 1],
                    (1, 1, 1),
                    2,
                    kern,
                )
            )
        self.head = nn.Conv3d(features_per_stage[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        x = self.transformer(skips[-1])
        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips[:-1])):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# nnU-Net entry point
# ---------------------------------------------------------------------------
def build_transunet(
    num_input_channels: int,
    num_output_channels: int,
    arch_init_kwargs: dict | None = None,
) -> nn.Module:
    """Construct the 3D-TransUNet sized from the plan's ``arch_init_kwargs``."""
    ak = arch_init_kwargs or {}
    features = ak.get("features_per_stage", [32, 64, 128, 256, 320, 320])
    strides = ak.get(
        "strides", [[1, 1, 1]] + [[2, 2, 2]] * (len(features) - 1)
    )
    n_conv = ak.get("n_conv_per_stage", 2)
    kernels = ak.get("kernel_sizes", 3)
    return TransUNet3D(
        in_channels=num_input_channels,
        num_classes=num_output_channels,
        features_per_stage=features,
        strides=strides,
        n_conv_per_stage=n_conv,
        kernel_sizes=kernels,
    )
