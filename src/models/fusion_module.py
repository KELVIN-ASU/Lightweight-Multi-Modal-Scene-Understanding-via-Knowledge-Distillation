import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Small building blocks
# ----------------------------
class Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)


class DWSeparableConv(nn.Module):
    """Depthwise separable conv: DW(3x3) + PW(1x1). Cheap and effective."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


# ----------------------------
# Camera multi-scale squeeze (FPN-lite)
# ----------------------------
class CameraFPNLite(nn.Module):
    """
    Fuses multi-scale camera features {stage: tensor} into a single map
    using 1x1 lateral projections + upsample + sum (+ DWSepConv).
    """
    def __init__(self, in_channels_by_stage: Dict[str, int],
                 target_channels: int = 128,
                 stages_to_use: Optional[List[str]] = None,
                 target_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.stages_to_use = stages_to_use or list(in_channels_by_stage.keys())
        self.laterals = nn.ModuleDict()
        for s in self.stages_to_use:
            self.laterals[s] = Conv1x1(in_channels_by_stage[s], target_channels)
        self.post = DWSeparableConv(target_channels, target_channels)
        self.target_size = target_size  # (H, W) or None = infer at runtime

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.target_size is None:
            hws = [feats[s].shape[-2:] for s in self.stages_to_use]
            H, W = max(hws, key=lambda x: x[0] * x[1])
        else:
            H, W = self.target_size

        fused = None
        for s in self.stages_to_use:
            x = self.laterals[s](feats[s])
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            fused = x if fused is None else fused + x
        return self.post(fused)


# ----------------------------
# Fusion blocks
# ----------------------------
class ConcatenationFusion(nn.Module):
    """Concat camera+LiDAR (with small projections) → DWSep fusion."""
    def __init__(self, camera_channels=128, lidar_channels=128, out_channels=256):
        super().__init__()
        self.camera_proj = Conv1x1(camera_channels, camera_channels)
        self.lidar_proj  = Conv1x1(lidar_channels,  lidar_channels)
        in_cat = camera_channels + lidar_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_cat, in_cat, kernel_size=3, padding=1, groups=in_cat, bias=False),
            nn.BatchNorm2d(in_cat),
            nn.ReLU(),
            nn.Conv2d(in_cat, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, cam_feat: torch.Tensor, lidar_feat: torch.Tensor) -> torch.Tensor:
        if cam_feat.shape[-2:] != lidar_feat.shape[-2:]:
            lidar_feat = F.interpolate(lidar_feat, size=cam_feat.shape[-2:], mode="bilinear", align_corners=False)
        cam_feat = self.camera_proj(cam_feat)
        lidar_feat = self.lidar_proj(lidar_feat)
        return self.fuse(torch.cat([cam_feat, lidar_feat], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MinimalFusion(nn.Module):
    """Ultra-light add fusion with 1x1 channel alignment on both branches."""
    def __init__(self, cam_ch=128, lidar_ch=128, out_ch=128):
        super().__init__()
        self.cam_proj   = Conv1x1(cam_ch, out_ch)
        self.lidar_proj = Conv1x1(lidar_ch, out_ch)

    def forward(self, cam_feat: torch.Tensor, lidar_feat: torch.Tensor) -> torch.Tensor:
        if cam_feat.shape[-2:] != lidar_feat.shape[-2:]:
            lidar_feat = F.interpolate(lidar_feat, size=cam_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.cam_proj(cam_feat) + self.lidar_proj(lidar_feat)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------
# Lightweight decoder / head (×4 upsample)
# ----------------------------
class LightweightSegmentationHead(nn.Module):
    """
    Two-stage upsampling head (×4 overall). Assumes encoder features at 1/4 res.
    64×64 → 128×128 → 256×256 for 256×256 input images.
    """
    def __init__(self, in_channels=256, num_classes=3):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.cls = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return self.cls(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------
# Complete model
# ----------------------------
class CompleteSegmentationModel(nn.Module):
    """
    End-to-end multi-modal model with:
      - camera encoder (single tensor or multiscale dict)
      - lidar encoder (BEV feature)
      - optional camera FPN-lite (if multiscale)
      - fusion (concat or minimal)
      - decoder (×4)
    Provides intermediates for KD.
    """
    def __init__(self,
                 camera_encoder: nn.Module,
                 lidar_encoder: nn.Module,
                 num_classes: int = 3,
                 fusion_type: str = "concat",
                 fusion_out_channels: int = 256,
                 camera_fpn_stages: Optional[List[str]] = None,
                 camera_fpn_channels: int = 128):
        super().__init__()
        self.camera_encoder = camera_encoder
        self.lidar_encoder  = lidar_encoder
        self.fusion_type    = fusion_type

        self.use_multiscale = getattr(camera_encoder, "return_multiscale", False)
        self.camera_fpn = None
        if self.use_multiscale:
            feat_info = camera_encoder.get_feature_info()
            self.camera_fpn = CameraFPNLite(
                in_channels_by_stage=feat_info,
                target_channels=camera_fpn_channels,
                stages_to_use=camera_fpn_stages  # e.g. ["stage3","stage4","stage5"]
            )
            cam_feat_channels = camera_fpn_channels
        else:
            cam_feat_channels = getattr(camera_encoder, "out_channels", 128)

        # LiDAR channels (SpatialLiDAREncoder exposes feature_dim)
        lidar_feat_channels = getattr(getattr(lidar_encoder, "encoder", lidar_encoder), "feature_dim", 128)

        if fusion_type == "concat":
            self.fusion = ConcatenationFusion(cam_feat_channels, lidar_feat_channels, fusion_out_channels)
            head_in = fusion_out_channels
        elif fusion_type == "minimal":
            self.fusion = MinimalFusion(cam_ch=cam_feat_channels, lidar_ch=lidar_feat_channels, out_ch=cam_feat_channels)
            head_in = cam_feat_channels
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.head = LightweightSegmentationHead(in_channels=head_in, num_classes=num_classes)

    def forward(self, images: torch.Tensor, points: torch.Tensor, return_intermediates: bool = False):
        cam_raw = self.camera_encoder(images)
        cam_feat = self.camera_fpn(cam_raw) if isinstance(cam_raw, dict) else cam_raw

        lidar_feat = self.lidar_encoder(points)
        lidar_feat_resized = (
            F.interpolate(lidar_feat, size=cam_feat.shape[-2:], mode="bilinear", align_corners=False)
            if cam_feat.shape[-2:] != lidar_feat.shape[-2:] else lidar_feat
        )

        if isinstance(self.fusion, ConcatenationFusion):
            cam_proj   = self.fusion.camera_proj(cam_feat)
            lidar_proj = self.fusion.lidar_proj(lidar_feat_resized)
            pre_fusion = torch.cat([cam_proj, lidar_proj], dim=1)
            fused = self.fusion.fuse(pre_fusion)
        else:
            pre_fusion = self.fusion.cam_proj(cam_feat) + self.fusion.lidar_proj(lidar_feat_resized)
            fused = pre_fusion

        logits = self.head(fused)

        if return_intermediates:
            return logits, {
                "camera_feat": cam_feat,
                "lidar_feat": lidar_feat_resized,
                "pre_fusion": pre_fusion,
                "post_fusion": fused,
                "logits": logits,
            }
        return logits

    def count_parameters(self):
        return {
            'camera_encoder': sum(p.numel() for p in self.camera_encoder.parameters() if p.requires_grad),
            'lidar_encoder':  sum(p.numel() for p in self.lidar_encoder.parameters()  if p.requires_grad),
            'fusion':         sum(p.numel() for p in self.fusion.parameters()         if p.requires_grad),
            'segmentation_head': sum(p.numel() for p in self.head.parameters()        if p.requires_grad),
            'total':          sum(p.numel() for p in self.parameters()                if p.requires_grad),
        }

    def get_architecture_summary(self):
        counts = self.count_parameters()
        total = counts['total']
        return {
            "fusion_type": self.fusion_type,
            "use_multiscale_camera": self.use_multiscale,
            "params": counts,
            "under_1M": total < 1_000_000,
            "budget_usage_pct": f"{100.0 * total / 1_000_000:.1f}%"
        }
