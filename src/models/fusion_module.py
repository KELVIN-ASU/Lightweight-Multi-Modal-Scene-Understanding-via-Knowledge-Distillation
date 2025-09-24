import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatenationFusion(nn.Module):
    """
    Simple concatenation-based fusion for camera and LiDAR features.
    Optimized for parameter efficiency and gradient stability.
    """
    def __init__(self, camera_channels=128, lidar_channels=128, output_channels=256):
        super().__init__()
        self.camera_channels = camera_channels
        self.lidar_channels = lidar_channels
        self.output_channels = output_channels
        
        # 1x1 convolution to fuse concatenated features
        # Removed inplace=True to fix gradient flow issues
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(camera_channels + lidar_channels, output_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()  # Fixed: removed inplace=True
        )
    
    def forward(self, camera_features, lidar_features):
        """
        Args:
            camera_features: [B, camera_channels, H, W]
            lidar_features: [B, lidar_channels, H, W]
        Returns:
            fused_features: [B, output_channels, H, W]
        """
        # Ensure spatial dimensions match
        if camera_features.shape[-2:] != lidar_features.shape[-2:]:
            lidar_features = F.interpolate(
                lidar_features, 
                size=camera_features.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Concatenate along channel dimension
        concatenated = torch.cat([camera_features, lidar_features], dim=1)
        
        # Apply fusion convolution
        fused = self.fusion_conv(concatenated)
        
        return fused
    
    def count_parameters(self):
        """Count total parameters in the fusion module"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightSegmentationHead(nn.Module):
    """
    Lightweight segmentation head optimized for parameter budget.
    Aggressively reduces channels to stay under 1M parameter limit.
    """
    def __init__(self, in_channels=256, num_classes=3, upsample_factor=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.upsample_factor = upsample_factor
        
        # Progressive upsampling with aggressive channel reduction
        # Target: 256 -> 64 -> 16 -> 16 channels
        layers = []
        current_channels = in_channels
        
        # Stage 1: 256 -> 64 channels, 32x32 -> 64x64
        layers.extend([
            nn.ConvTranspose2d(current_channels, 64, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()  # Fixed: removed inplace=True
        ])
        
        # Stage 2: 64 -> 16 channels, 64x64 -> 128x128
        layers.extend([
            nn.ConvTranspose2d(64, 16, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()  # Fixed: removed inplace=True
        ])
        
        # Stage 3: 16 -> 16 channels, 128x128 -> 256x256
        layers.extend([
            nn.ConvTranspose2d(16, 16, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()  # Fixed: removed inplace=True
        ])
        
        # Final classification layer: 16 -> num_classes
        layers.append(nn.Conv2d(16, num_classes, kernel_size=3, padding=1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W] fused features
        Returns:
            segmentation_logits: [B, num_classes, H*upsample_factor, W*upsample_factor]
        """
        return self.decoder(x)
    
    def count_parameters(self):
        """Count total parameters in the segmentation head"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CompleteSegmentationModel(nn.Module):
    """
    Complete multi-modal segmentation model combining camera, LiDAR, fusion, and head.
    Optimized for <1M parameter constraint and stable training.
    """
    def __init__(self, camera_encoder, lidar_encoder, fusion_channels=256, num_classes=3):
        super().__init__()
        self.camera_encoder = camera_encoder
        self.lidar_encoder = lidar_encoder
        
        # Get channel dimensions from encoders
        camera_channels = camera_encoder.out_channels
        lidar_channels = lidar_encoder.encoder.feature_dim
        
        # Initialize fusion module
        self.fusion = ConcatenationFusion(
            camera_channels=camera_channels,
            lidar_channels=lidar_channels,
            output_channels=fusion_channels
        )
        
        # Initialize lightweight segmentation head
        self.segmentation_head = LightweightSegmentationHead(
            in_channels=fusion_channels,
            num_classes=num_classes,
            upsample_factor=8  # Match camera encoder's spatial reduction
        )
    
    def forward(self, images, points):
        """
        Args:
            images: [B, 3, H, W] input images
            points: [B, N, 4] input point clouds
        Returns:
            segmentation_logits: [B, num_classes, H, W] segmentation predictions
        """
        # Extract features from both modalities
        camera_features = self.camera_encoder(images)    # [B, 128, H/8, W/8]
        lidar_features = self.lidar_encoder(points)      # [B, 128, H/8, W/8]
        
        # Fuse the features
        fused_features = self.fusion(camera_features, lidar_features)  # [B, 256, H/8, W/8]
        
        # Generate segmentation predictions
        segmentation_logits = self.segmentation_head(fused_features)   # [B, 3, H, W]
        
        return segmentation_logits
    
    def count_parameters(self):
        """Count parameters for each component and total"""
        return {
            'camera_encoder': self.camera_encoder.count_parameters(),
            'lidar_encoder': self.lidar_encoder.count_parameters(),
            'fusion': self.fusion.count_parameters(),
            'segmentation_head': self.segmentation_head.count_parameters(),
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_architecture_summary(self):
        """Get a summary of the model architecture"""
        param_counts = self.count_parameters()
        total_params = param_counts['total']
        budget_usage = total_params / 1_000_000 * 100
        
        summary = {
            'architecture': 'Multi-Modal Concatenation Fusion',
            'fusion_type': 'concatenation',
            'parameter_breakdown': param_counts,
            'budget_usage': f"{budget_usage:.1f}%",
            'under_budget': total_params < 1_000_000,
            'components': {
                'camera_encoder': 'TwinLiteNet (MobileNetV2-based)',
                'lidar_encoder': 'Spatial BEV Encoder (PointNet-style)',
                'fusion': 'Concatenation + 1x1 Conv',
                'segmentation_head': 'Lightweight Upsampling Decoder'
            }
        }
        return summary


# Alternative minimal fusion for extreme parameter constraints
class MinimalFusion(nn.Module):
    """
    Ultra-lightweight fusion using element-wise addition.
    Use this if concatenation fusion still exceeds parameter budget.
    """
    def __init__(self, channels=128):
        super().__init__()
        self.channels = channels
        # No additional parameters - just element-wise addition
        
    def forward(self, camera_features, lidar_features):
        # Ensure spatial compatibility
        if camera_features.shape[-2:] != lidar_features.shape[-2:]:
            lidar_features = F.interpolate(
                lidar_features, size=camera_features.shape[-2:], 
                mode='bilinear', align_corners=False
            )
        
        # Simple addition (requires same number of channels)
        return camera_features + lidar_features
    
    def count_parameters(self):
        return 0  # No learnable parameters


class MinimalSegmentationModel(nn.Module):
    """
    Ultra-minimal model for extreme parameter constraints.
    Uses addition fusion and minimal segmentation head.
    """
    def __init__(self, camera_encoder, lidar_encoder, num_classes=3):
        super().__init__()
        self.camera_encoder = camera_encoder
        self.lidar_encoder = lidar_encoder
        
        # Minimal fusion with no additional parameters
        self.fusion = MinimalFusion(channels=128)
        
        # Ultra-lightweight segmentation head
        self.segmentation_head = nn.Sequential(
            # Direct upsampling without intermediate layers
            nn.ConvTranspose2d(128, 32, 8, stride=8, padding=0, bias=False),  # 8x upsampling in one step
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )
    
    def forward(self, images, points):
        camera_features = self.camera_encoder(images)
        lidar_features = self.lidar_encoder(points)
        fused_features = self.fusion(camera_features, lidar_features)
        segmentation_logits = self.segmentation_head(fused_features)
        return segmentation_logits
    
    def count_parameters(self):
        return {
            'camera_encoder': self.camera_encoder.count_parameters(),
            'lidar_encoder': self.lidar_encoder.count_parameters(),
            'fusion': self.fusion.count_parameters(),
            'segmentation_head': sum(p.numel() for p in self.segmentation_head.parameters()),
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }