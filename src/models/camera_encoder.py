# models/camera_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------
# InvertedResidual Block (from MobileNetV2 logic)
# -----------------------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_ratio=6):
        super().__init__()

        # Use residual connection only when input and output sizes match
        self.use_residual = stride == 1 and in_channels == out_channels

        # Hidden dim is how many channels to expand to temporarily
        hidden_dim = int(round(in_channels * expansion_ratio))

        layers = []

        # 1. Pointwise convolution (1x1) for expansion (only if expanding)
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6()  # Fixed: removed inplace=True
            ])

        # 2. Depthwise convolution (3x3)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),  # Depthwise: one conv per channel
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6()  # Fixed: removed inplace=True
        ])

        # 3. Pointwise projection back to out_channels
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        # Wrap all layers in a Sequential block
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # If residual connection is allowed, add input to output
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

# -----------------------------------------------
# TwinLiteEncoder: Lightweight CNN for images
# -----------------------------------------------
class TwinLiteEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, return_multiscale=False):
        super().__init__()
        
        self.return_multiscale = return_multiscale

        # Initial stem conv: reduce spatial size and extract low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6()  # Fixed: removed inplace=True
        )

        # Stage 1: Keep size the same, minimal expansion for efficiency
        self.stage1 = InvertedResidual(base_channels, base_channels, stride=1, expansion_ratio=1)
        
        # Stage 2: Downsample and expand channels
        self.stage2 = InvertedResidual(base_channels, base_channels * 2, stride=2, expansion_ratio=6)
        
        # Stage 3: Keep same size, add more capacity
        self.stage3 = InvertedResidual(base_channels * 2, base_channels * 2, stride=1, expansion_ratio=6)
        
        # Stage 4: Final downsample and channel expansion
        self.stage4 = InvertedResidual(base_channels * 2, base_channels * 4, stride=2, expansion_ratio=6)
        
        # Optional: Add one more stage for deeper features if needed
        self.stage5 = InvertedResidual(base_channels * 4, base_channels * 4, stride=1, expansion_ratio=6)

        # Store channel dimensions for each scale (useful for fusion module)
        self.feature_channels = {
            'stage2': base_channels * 2,  # 1/4 resolution
            'stage3': base_channels * 2,  # 1/4 resolution  
            'stage4': base_channels * 4,  # 1/8 resolution
            'stage5': base_channels * 4   # 1/8 resolution
        }
        
        # Final output channels (for single-scale output)
        self.out_channels = base_channels * 4

    def forward(self, x):
        # Input: [B, 3, H, W]
        x = self.stem(x)      # [B, 32, H/2, W/2]
        
        x1 = self.stage1(x)   # [B, 32, H/2, W/2]
        x2 = self.stage2(x1)  # [B, 64, H/4, W/4]
        x3 = self.stage3(x2)  # [B, 64, H/4, W/4]
        x4 = self.stage4(x3)  # [B, 128, H/8, W/8]
        x5 = self.stage5(x4)  # [B, 128, H/8, W/8]

        if self.return_multiscale:
            # Return features at multiple scales for richer fusion
            return {
                'stage2': x2,  # Early features, higher resolution
                'stage3': x3,  # Mid-level features
                'stage4': x4,  # Deep features, lower resolution
                'stage5': x5   # Final features
            }
        else:
            # Return only final features (your original approach)
            return x5

    def get_feature_info(self):
        """Helper method to get channel dimensions at each scale"""
        return self.feature_channels
    
    def count_parameters(self):
        """Count total parameters in the encoder"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)