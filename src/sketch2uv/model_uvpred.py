from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def create_feature_backbone(backbone_name: str, pretrained: bool, out_indices: tuple[int, ...]) -> nn.Module:
    candidates = [
        backbone_name,
        "convnext_base.fb_in22k_ft_in1k",
        "convnext_base.in22k_ft_in1k",
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
            )
            model.sketch2rig_backbone_name = name
            return model
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not create a feature backbone. Last error: {last_error}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UVPredNet(nn.Module):
    """
    Dense sketch -> UV render predictor.

    Output convention:
      - mask_logits: [B, 1, H, W], foreground probability before sigmoid
      - uv:          [B, 2, H, W], normalized U/V coordinates in [0, 1]
      - render:      [B, 3, H, W], UV-render RGB as [U * mask_prob, V * mask_prob, 0]
    """

    model_name = "uvpred"

    def __init__(
        self,
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k",
        pretrained: bool = True,
        decoder_channels: int = 256,
        dropout: float = 0.1,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.backbone = create_feature_backbone(backbone_name, pretrained, out_indices)
        self.backbone_name = getattr(self.backbone, "sketch2rig_backbone_name", backbone_name)
        self.channels = list(self.backbone.feature_info.channels())

        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, decoder_channels, kernel_size=1) for ch in self.channels]
        )
        self.smooth = nn.ModuleList(
            [ConvBlock(decoder_channels, decoder_channels, dropout=dropout) for _ in self.channels]
        )
        self.head = nn.Sequential(
            ConvBlock(decoder_channels, decoder_channels // 2, dropout=dropout),
            nn.Conv2d(decoder_channels // 2, 3, kernel_size=1),
        )

    def _ensure_nchw(self, feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
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

        logits = self.head(pyramid)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        mask_logits = logits[:, :1]
        uv = torch.sigmoid(logits[:, 1:3])
        mask_prob = torch.sigmoid(mask_logits)
        render = torch.cat([uv * mask_prob, torch.zeros_like(mask_prob)], dim=1)
        return {
            "mask_logits": mask_logits,
            "mask": mask_prob,
            "uv": uv,
            "render": render,
        }

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.lateral.parameters()
        yield from self.smooth.parameters()
        yield from self.head.parameters()
