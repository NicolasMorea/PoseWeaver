from __future__ import annotations

from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_feature_backbone(
    backbone_name: str,
    pretrained: bool,
    out_indices: tuple[int, ...],
    drop_path_rate: float,
) -> nn.Module:
    candidates = [
        backbone_name,
        "convnext_base.fb_in22k_ft_in1k",
        "convnext_small.in12k_ft_in1k",
        "convnext_small",
        "resnet50",
    ]
    seen: set[str] = set()
    last_error: Exception | None = None
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            model = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                drop_path_rate=drop_path_rate,
            )
            model.sketch2rig_backbone_name = name
            return model
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not create a feature backbone. Last error: {last_error}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        groups = min(32, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ISONet(nn.Module):
    """FPN sketch encoder that predicts a mask and a 64D descriptor per pixel."""

    model_name = "iso"

    def __init__(
        self,
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k",
        pretrained: bool = True,
        decoder_channels: int = 256,
        dropout: float = 0.15,
        drop_path_rate: float = 0.15,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.backbone = create_feature_backbone(
            backbone_name,
            pretrained,
            out_indices,
            drop_path_rate,
        )
        self.backbone_name = getattr(self.backbone, "sketch2rig_backbone_name", backbone_name)
        self.channels = list(self.backbone.feature_info.channels())
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, decoder_channels, kernel_size=1) for ch in self.channels]
        )
        self.smooth = nn.ModuleList(
            [ConvBlock(decoder_channels, decoder_channels, dropout) for _ in self.channels]
        )
        head_channels = decoder_channels // 2
        self.shared_head = ConvBlock(decoder_channels, head_channels, dropout)
        self.mask_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.iso_head = nn.Conv2d(head_channels, 64, kernel_size=1)

    @staticmethod
    def _ensure_nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected a 4D feature map, got {tuple(feature.shape)}")
        if feature.shape[1] == expected_channels:
            return feature
        if feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"Feature map channel mismatch: expected {expected_channels}, got {tuple(feature.shape)}"
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = [
            self._ensure_nchw(feature, expected_channels)
            for feature, expected_channels in zip(features, self.channels)
        ]

        pyramid: torch.Tensor | None = None
        for idx in reversed(range(len(features))):
            lateral = self.lateral[idx](features[idx])
            if pyramid is not None:
                pyramid = F.interpolate(
                    pyramid,
                    size=lateral.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                lateral = lateral + pyramid
            pyramid = self.smooth[idx](lateral)
        if pyramid is None:
            raise RuntimeError("Backbone returned no feature maps")

        shared = self.shared_head(pyramid)
        mask_logits = self.mask_head(shared)
        iso_logits = self.iso_head(shared)
        mask_logits = F.interpolate(
            mask_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        iso_logits = F.interpolate(
            iso_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.sigmoid(mask_logits)
        iso = torch.sigmoid(iso_logits)
        return {
            "mask_logits": mask_logits,
            "mask": mask,
            "iso": iso,
            "render": iso * mask,
        }

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.lateral.parameters()
        yield from self.smooth.parameters()
        yield from self.shared_head.parameters()
        yield from self.mask_head.parameters()
        yield from self.iso_head.parameters()
