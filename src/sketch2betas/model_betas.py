from __future__ import annotations

import torch
import torch.nn as nn
import timm

NUM_BETAS = 16


def create_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
    candidates = [
        backbone_name,
        "convnext_base.fb_in22k_ft_in1k",
        "convnext_base.in22k_ft_in1k",
        "convnext_base_384_in22ft1k",
        "convnext_small.in12k_ft_in1k",
        "convnext_small_in22ft1k",
        "convnext_small",
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
                num_classes=0,
                global_pool="",
            )
            model.sketch2rig_backbone_name = name
            return model
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not create any ConvNeXt backbone. Last error: {last_error}")


class SpatialAttentionPool(nn.Module):
    """Learned attention pooling over a backbone spatial feature map."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim == 4:
            tokens = feature_map.flatten(2).permute(0, 2, 1)
        elif feature_map.ndim == 3:
            tokens = feature_map
        else:
            raise ValueError(f"Expected [B,C,H,W] or [B,T,C] features, got {tuple(feature_map.shape)}")

        tokens = self.input_proj(tokens)
        tokens = tokens + self.pos[:, : tokens.size(1)]

        query = self.query.expand(tokens.size(0), -1, -1)
        pooled, _ = self.attn(query=query, key=tokens, value=tokens, need_weights=False)
        pooled = self.norm1(query + pooled)
        pooled = self.norm2(pooled + self.mlp(pooled))
        return pooled.squeeze(1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SketchBetasGenderNet(nn.Module):
    """
    Sketch -> SMPL shape and gender predictor.

    A pretrained ConvNeXt backbone extracts high-resolution sketch features.
    A learned attention pool keeps spatial evidence available for global shape
    cues, then two task-specific heads predict normalized betas and gender.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k",
        num_betas: int = NUM_BETAS,
        num_genders: int = 2,
        pretrained: bool = True,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.15,
        mlp_blocks: int = 2,
        img_size: int = 384,
    ) -> None:
        super().__init__()

        self.backbone = create_backbone(backbone_name, pretrained)
        self.backbone_name = getattr(self.backbone, "sketch2rig_backbone_name", backbone_name)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = self.backbone(dummy)
        if feat.ndim == 4:
            in_channels = int(feat.shape[1])
        elif feat.ndim == 3:
            in_channels = int(feat.shape[-1])
        else:
            raise ValueError(f"Unsupported backbone feature shape: {tuple(feat.shape)}")

        self.pool = SpatialAttentionPool(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            *[ResidualMLPBlock(d_model, d_model * 2, dropout) for _ in range(mlp_blocks)],
            nn.LayerNorm(d_model),
        )

        self.beta_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_betas),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_genders),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        pooled = self.pool(features)
        shared = self.shared(pooled)
        return {
            "betas": self.beta_head(shared),
            "gender_logits": self.gender_head(shared),
        }
