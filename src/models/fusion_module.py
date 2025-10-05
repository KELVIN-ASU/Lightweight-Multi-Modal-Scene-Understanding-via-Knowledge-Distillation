# fusion_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

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
    """Depthwise separable conv: DW(3x3) + PW(1x1)."""
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


class CameraFPNLite(nn.Module):
    """Fuses multi-scale camera features."""
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
        self.target_size = target_size

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
    """Concat camera+LiDAR -> depthwise + pointwise conv fusion."""
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


class MinimalFusion(nn.Module):
    """Ultra-light add fusion with 1x1 projections."""
    def __init__(self, cam_ch=128, lidar_ch=128, out_ch=128):
        super().__init__()
        self.cam_proj   = Conv1x1(cam_ch, out_ch)
        self.lidar_proj = Conv1x1(lidar_ch, out_ch)

    def forward(self, cam_feat: torch.Tensor, lidar_feat: torch.Tensor) -> torch.Tensor:
        if cam_feat.shape[-2:] != lidar_feat.shape[-2:]:
            lidar_feat = F.interpolate(lidar_feat, size=cam_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.cam_proj(cam_feat) + self.lidar_proj(lidar_feat)


class WeightedFusion(nn.Module):
    """Learnable weighted fusion with spatial attention."""
    def __init__(self, cam_ch=128, lidar_ch=128, out_ch=128):
        super().__init__()
        self.cam_proj = Conv1x1(cam_ch, out_ch)
        self.lidar_proj = Conv1x1(lidar_ch, out_ch)
        
        # Spatial attention to learn per-location weights
        self.attention = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, cam_feat: torch.Tensor, lidar_feat: torch.Tensor) -> torch.Tensor:
        if cam_feat.shape[-2:] != lidar_feat.shape[-2:]:
            lidar_feat = F.interpolate(lidar_feat, size=cam_feat.shape[-2:], mode="bilinear", align_corners=False)
        
        cam_proj = self.cam_proj(cam_feat)
        lidar_proj = self.lidar_proj(lidar_feat)
        
        # Compute spatial attention weights
        concat = torch.cat([cam_proj, lidar_proj], dim=1)
        weights = self.attention(concat)  # [B, 2, H, W]
        
        cam_weight = weights[:, 0:1, :, :]
        lidar_weight = weights[:, 1:2, :, :]
        
        return cam_proj * cam_weight + lidar_proj * lidar_weight


# ----------------------------
# Decoder heads
# ----------------------------
class LightweightSegmentationHead(nn.Module):
    """Two-stage upsampling head."""
    def __init__(self, in_channels=256, num_classes=2):
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
        return self.cls(self.up2(self.up1(x)))


class SameResolutionSegmentationHead(nn.Module):
    """Keeps feature resolution (64x64 for BEV)."""
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.block = nn.Sequential(
            DWSeparableConv(in_channels, 64),
            DWSeparableConv(64, 32),
        )
        self.cls = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x): 
        return self.cls(self.block(x))


# ----------------------------
# Complete model
# ----------------------------
class CompleteSegmentationModel(nn.Module):
    def __init__(self,
                 camera_encoder: nn.Module,
                 lidar_encoder: nn.Module,
                 num_classes: int = 2,
                 fusion_type: str = "concat",
                 fusion_out_channels: int = 256,
                 camera_fpn_stages: Optional[List[str]] = None,
                 camera_fpn_channels: int = 128,
                 output_mode: str = "same"
                 ):
        super().__init__()
        self.camera_encoder = camera_encoder
        self.lidar_encoder  = lidar_encoder
        self.fusion_type    = fusion_type
        self.output_mode    = output_mode

        # Camera feature processing
        self.use_multiscale = getattr(camera_encoder, "return_multiscale", False)
        self.camera_fpn = None
        if self.use_multiscale:
            feat_info = camera_encoder.get_feature_info()
            self.camera_fpn = CameraFPNLite(
                in_channels_by_stage=feat_info,
                target_channels=camera_fpn_channels,
                stages_to_use=camera_fpn_stages
            )
            cam_feat_channels = camera_fpn_channels
        else:
            cam_feat_channels = getattr(camera_encoder, "out_channels", 128)

        # LiDAR feature channels
        lidar_feat_channels = getattr(getattr(lidar_encoder, "encoder", lidar_encoder), "feature_dim", 128)

        # Fusion
        if fusion_type == "concat":
            self.fusion = ConcatenationFusion(cam_feat_channels, lidar_feat_channels, fusion_out_channels)
            head_in = fusion_out_channels
        elif fusion_type == "minimal":
            self.fusion = MinimalFusion(cam_ch=cam_feat_channels, lidar_ch=lidar_feat_channels, out_ch=cam_feat_channels)
            head_in = cam_feat_channels
        elif fusion_type == "weighted":
            self.fusion = WeightedFusion(cam_ch=cam_feat_channels, lidar_ch=lidar_feat_channels, out_ch=cam_feat_channels)
            head_in = cam_feat_channels
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Decoder head
        if output_mode == "x4":
            self.head = LightweightSegmentationHead(in_channels=head_in, num_classes=num_classes)
        elif output_mode == "same":
            self.head = SameResolutionSegmentationHead(in_channels=head_in, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown output_mode: {output_mode}")

    def forward(self, images: torch.Tensor, points: torch.Tensor, return_intermediates: bool = False):
        cam_raw = self.camera_encoder(images)
        cam_feat = self.camera_fpn(cam_raw) if isinstance(cam_raw, dict) else cam_raw

        lidar_feat = self.lidar_encoder(points)
        if cam_feat.shape[-2:] != lidar_feat.shape[-2:]:
            lidar_feat = F.interpolate(lidar_feat, size=cam_feat.shape[-2:], mode="bilinear", align_corners=False)

        if isinstance(self.fusion, ConcatenationFusion):
            cam_proj   = self.fusion.camera_proj(cam_feat)
            lidar_proj = self.fusion.lidar_proj(lidar_feat)
            pre_fusion = torch.cat([cam_proj, lidar_proj], dim=1)
            fused = self.fusion.fuse(pre_fusion)
        else:
            cam_proj = self.fusion.cam_proj(cam_feat)
            lidar_proj = self.fusion.lidar_proj(lidar_feat)
            if isinstance(self.fusion, WeightedFusion):
                concat = torch.cat([cam_proj, lidar_proj], dim=1)
                weights = self.fusion.attention(concat)
                pre_fusion = cam_proj * weights[:, 0:1] + lidar_proj * weights[:, 1:2]
            else:
                pre_fusion = cam_proj + lidar_proj
            fused = pre_fusion

        logits = self.head(fused)

        if return_intermediates:
            return logits, {"camera_feat": cam_feat, "lidar_feat": lidar_feat,
                            "pre_fusion": pre_fusion, "post_fusion": fused, "logits": logits}
        return logits

    def get_architecture_summary(self):
        """Return a summary of the model architecture"""
        cam_params = sum(p.numel() for p in self.camera_encoder.parameters())
        lidar_params = sum(p.numel() for p in self.lidar_encoder.parameters())
        
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        if self.camera_fpn is not None:
            fusion_params += sum(p.numel() for p in self.camera_fpn.parameters())
        
        head_params = sum(p.numel() for p in self.head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "camera_params": f"{cam_params:,}",
            "lidar_params": f"{lidar_params:,}",
            "fusion_params": f"{fusion_params:,}",
            "head_params": f"{head_params:,}",
            "total_params": f"{total_params:,}",
            "fusion_type": self.fusion_type,
            "output_mode": self.output_mode,
            "use_multiscale": self.use_multiscale
        }