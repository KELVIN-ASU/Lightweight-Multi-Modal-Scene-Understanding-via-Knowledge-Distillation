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
                nn.ReLU6(inplace=True)
            ])

        # 2. Depthwise convolution (3x3)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),  # Depthwise: one conv per channel
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
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
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # Initial stem conv: reduce spatial size and extract low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True)
        )

        # Stack of inverted residual blocks
        self.blocks = nn.Sequential(
            # Block 1: keep size the same
            InvertedResidual(base_channels, base_channels, stride=1, expansion_ratio=1),

            # Block 2: downsample and expand channels
            InvertedResidual(base_channels, base_channels * 2, stride=2, expansion_ratio=6),

            # Block 3: keep same size, more capacity
            InvertedResidual(base_channels * 2, base_channels * 2, stride=1, expansion_ratio=6),

            # Block 4: downsample again, increase channels
            InvertedResidual(base_channels * 2, base_channels * 4, stride=2, expansion_ratio=6),
        )

        # Final number of output channels (needed by decoder or fusion module)
        self.out_channels = base_channels * 4

    def forward(self, x):
        # Pass through stem conv
        x = self.stem(x)

        # Pass through inverted residual blocks
        x = self.blocks(x)

        # Output is the feature map to be fused or decoded
        return x
