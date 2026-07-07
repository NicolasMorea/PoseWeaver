from __future__ import annotations

from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiteResidualBlock(nn.Module):
    """Depthwise residual refinement, cheap enough for full-resolution features."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        groups = _groups(out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, 3, padding=1,
                groups=out_channels, bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.project(x)
        return F.gelu(projected + self.residual(projected))


def create_backbone(
    name: str,
    pretrained: bool,
    out_indices: tuple[int, ...],
    drop_path_rate: float,
) -> nn.Module:
    candidates = (
        name,
        "convnext_base.fb_in22k_ft_in1k",
        "convnext_small.in12k_ft_in1k",
        "convnext_small",
        "resnet50",
    )
    attempted: set[str] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in attempted:
            continue
        attempted.add(candidate)
        try:
            backbone = timm.create_model(
                candidate,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                drop_path_rate=drop_path_rate,
            )
            backbone.sketch2rig_backbone_name = candidate
            return backbone
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not create a feature backbone: {last_error}")


class DepthNet(nn.Module):
    """High-resolution sketch-to-depth FPN.

    The decoder keeps semantic multi-scale features while two shallow image
    guidance paths reintroduce the original half/full-resolution ink.  A
    bounded detail residual refines a coarse depth logit, and an explicit edge
    head is supervised on silhouettes and depth discontinuities.  This avoids
    transposed convolutions and their checkerboard artifacts.

    ``depth`` is normalized inverse camera depth (near=1, far=0).  ``render``
    is masked for direct visualization; use ``depth`` with ``mask`` for losses.
    """

    model_name = "depth"

    def __init__(
        self,
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k",
        pretrained: bool = True,
        decoder_channels: int = 256,
        dropout: float = 0.15,
        drop_path_rate: float = 0.15,
        max_detail_residual: float = 1.0,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.backbone = create_backbone(
            backbone_name, pretrained, out_indices, drop_path_rate
        )
        self.backbone_name = getattr(
            self.backbone, "sketch2rig_backbone_name", backbone_name
        )
        self.encoder_channels = list(self.backbone.feature_info.channels())
        self.max_detail_residual = max_detail_residual

        self.lateral = nn.ModuleList([
            nn.Conv2d(channels, decoder_channels, 1)
            for channels in self.encoder_channels
        ])
        self.smooth = nn.ModuleList([
            ConvBlock(decoder_channels, decoder_channels, dropout)
            for _ in self.encoder_channels
        ])

        half_channels = max(decoder_channels // 2, 64)
        full_channels = max(decoder_channels // 4, 48)
        self.image_half = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.refine_half = LiteResidualBlock(
            decoder_channels + 32, half_channels, dropout
        )
        self.image_full = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.GELU(),
        )
        self.refine_full = LiteResidualBlock(
            half_channels + 16, full_channels, dropout
        )
        self.shared = LiteResidualBlock(full_channels, full_channels, dropout)

        self.mask_head = nn.Conv2d(full_channels, 1, 1)
        self.boundary_head = nn.Conv2d(full_channels, 1, 1)
        self.coarse_depth_head = nn.Sequential(
            ConvBlock(full_channels, full_channels, dropout),
            nn.Conv2d(full_channels, 1, 1),
        )
        # Boundary probability is supplied to the detail head as a learned
        # high-frequency gate. It remains differentiable during training.
        self.detail_head = nn.Sequential(
            LiteResidualBlock(full_channels + 1, full_channels, dropout),
            nn.Conv2d(full_channels, 1, 3, padding=1),
        )

    @staticmethod
    def _ensure_nchw(feature: torch.Tensor, channels: int) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got {tuple(feature.shape)}")
        if feature.shape[1] == channels:
            return feature
        if feature.shape[-1] == channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"Feature map channel mismatch: expected {channels}, got {tuple(feature.shape)}"
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        features = [
            self._ensure_nchw(feature, channels)
            for feature, channels in zip(features, self.encoder_channels)
        ]
        pyramid: torch.Tensor | None = None
        for index in reversed(range(len(features))):
            lateral = self.lateral[index](features[index])
            if pyramid is not None:
                pyramid = F.interpolate(
                    pyramid, size=lateral.shape[-2:], mode="bilinear", align_corners=False
                )
                lateral = lateral + pyramid
            pyramid = self.smooth[index](lateral)
        if pyramid is None:
            raise RuntimeError("Backbone returned no feature maps")

        half_guide = self.image_half(image)
        pyramid = F.interpolate(
            pyramid, size=half_guide.shape[-2:], mode="bilinear", align_corners=False
        )
        half = self.refine_half(torch.cat((pyramid, half_guide), dim=1))
        # Nearest preserves already-localized boundaries; the following
        # learned convolution removes blockiness without pre-blurring edges.
        half = F.interpolate(half, size=image.shape[-2:], mode="nearest")
        full = self.refine_full(torch.cat((half, self.image_full(image)), dim=1))
        shared = self.shared(full)

        mask_logits = self.mask_head(shared)
        boundary_logits = self.boundary_head(shared)
        coarse_logits = self.coarse_depth_head(shared)
        boundary = torch.sigmoid(boundary_logits)
        detail_logits = torch.tanh(
            self.detail_head(torch.cat((shared, boundary), dim=1))
        ) * self.max_detail_residual
        depth_logits = coarse_logits + detail_logits
        depth = torch.sigmoid(depth_logits)
        mask = torch.sigmoid(mask_logits)
        return {
            "mask_logits": mask_logits,
            "mask": mask,
            "boundary_logits": boundary_logits,
            "boundary": boundary,
            "coarse_depth": torch.sigmoid(coarse_logits),
            "detail_logits": detail_logits,
            "depth_logits": depth_logits,
            "depth": depth,
            "render": depth * mask,
        }

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        modules = (
            self.lateral,
            self.smooth,
            self.image_half,
            self.refine_half,
            self.image_full,
            self.refine_full,
            self.shared,
            self.mask_head,
            self.boundary_head,
            self.coarse_depth_head,
            self.detail_head,
        )
        for module in modules:
            yield from module.parameters()
