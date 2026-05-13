from __future__ import annotations

import torch
import torch.nn as nn
import timm

NUM_JOINTS = 22

# SMPL-H 22-joint parent array (index i → parent of joint i, -1 = root)
SMPL_PARENTS: list[int] = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


class LearnedPositionalEncoding(nn.Module):
    """Learnable 2-D positional encoding for a flattened spatial feature map."""

    def __init__(self, d_model: int, max_tokens: int = 256) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return x + self.pe[:, : x.size(1)]


class PoseTransformerHead(nn.Module):
    """
    Cross-attention decoder: NUM_JOINTS learnable query tokens attend to
    backbone spatial features → per-joint 3-D coordinate regression.

    Architecture inspired by DETR / TransPose.
    """

    def __init__(
        self,
        in_channels: int,
        num_joints: int = NUM_JOINTS,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = LearnedPositionalEncoding(d_model)

        # One learnable embedding per joint (jointly updated with the decoder)
        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # feature_map: [B, C, H, W]
        B, C, H, W = feature_map.shape
        tokens = feature_map.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        tokens = self.input_proj(tokens)                   # [B, H*W, d_model]
        tokens = self.pos_enc(tokens)

        queries = self.joint_queries.expand(B, -1, -1)     # [B, J, d_model]
        out = self.decoder(tgt=queries, memory=tokens)     # [B, J, d_model]
        out = self.norm(out)
        return self.output_proj(out)                       # [B, J, 3]


class Sketch3DPoseNet(nn.Module):
    """
    Sketch → 3D skeleton predictor.

    Backbone : ConvNeXt-Small pretrained on ImageNet-12k (via timm).
               Returns a spatial feature map [B, 768, H/32, W/32].
    Head     : PoseTransformerHead — 22 joint queries cross-attend to the
               spatial feature map and regress (u_norm, v_norm, z_norm).
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small.in12k_ft_in1k",
        num_joints: int = NUM_JOINTS,
        pretrained: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # No global pooling — we want the spatial feature map
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Probe output channels with a dummy pass (works for any timm backbone)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            feat = self.backbone(dummy)
        in_channels: int = feat.shape[1]

        self.pose_head = PoseTransformerHead(
            in_channels=in_channels,
            num_joints=num_joints,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        features = self.backbone(x)        # [B, C, H', W']
        return self.pose_head(features)    # [B, J, 3]
