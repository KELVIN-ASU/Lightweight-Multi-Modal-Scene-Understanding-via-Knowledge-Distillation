import torch
import torch.nn as nn
from typing import Tuple, List

# ================================================================
# Lightweight Spatial LiDAR Encoder (Vectorized Implementation)
# ================================================================
class SpatialLiDAREncoder(nn.Module):
    def __init__(self, input_dim: int = 4, feature_dim: int = 128, 
                 grid_size: Tuple[int, int] = (128, 128), 
                 point_cloud_range: List[float] = [-50, -50, -5, 50, 50, 3]):
        """
        Efficient PointNet-style encoder with BEV grid creation.

        Args:
            input_dim: Number of input point features (x,y,z,intensity = 4)
            feature_dim: Number of output feature channels
            grid_size: Size of the 2D BEV feature map (H, W)
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] for normalization
        """
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.point_cloud_range = point_cloud_range
        H, W = grid_size

        # Calculate voxel size for coordinate normalization
        self.voxel_size = [
            (point_cloud_range[3] - point_cloud_range[0]) / W,
            (point_cloud_range[4] - point_cloud_range[1]) / H
        ]

        # Enhanced point-wise feature extractor (PointNet-style MLP)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def points_to_bev_coords(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert 3D points to normalized BEV grid coordinates.

        Args:
            points: [B, N, 4] where last dim is (x, y, z, intensity)
        Returns:
            coords: [B, N, 2] normalized coordinates in [0, 1] range
            valid_mask: [B, N] boolean mask of points within range
        """
        x, y = points[..., 0], points[..., 1]

        # Normalize to [0, 1]
        x_norm = (x - self.point_cloud_range[0]) / (self.point_cloud_range[3] - self.point_cloud_range[0])
        y_norm = (y - self.point_cloud_range[1]) / (self.point_cloud_range[4] - self.point_cloud_range[1])

        coords = torch.stack([x_norm, y_norm], dim=-1)

        # Valid points are those that fall inside the defined BEV range
        valid_mask = (x_norm >= 0) & (x_norm <= 1) & (y_norm >= 0) & (y_norm <= 1)

        return coords, valid_mask

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert point cloud to BEV feature map.

        Args:
            points: [B, N, 4] point cloud tensor (x, y, z, intensity)
        Returns:
            feature_map: [B, feature_dim, H, W] BEV feature map
        """
        B, N, _ = points.shape
        H, W = self.grid_size

        # Get BEV coordinates and mask
        coords, valid_mask = self.points_to_bev_coords(points)

        # Extract per-point features [B, C, N]
        point_features = self.point_mlp(points.transpose(1, 2))  # [B, C, N]

        # Initialize feature map
        feature_map = torch.zeros(B, self.feature_dim, H, W, device=points.device, dtype=points.dtype)

        # Convert normalized coords to integer grid indices
        grid_coords = (coords * torch.tensor([W - 1, H - 1], device=points.device)).long()

        # ✅ Clamp indices to valid range
        grid_coords[..., 0] = grid_coords[..., 0].clamp(0, W - 1)
        grid_coords[..., 1] = grid_coords[..., 1].clamp(0, H - 1)

        # Batch indices
        batch_indices = torch.arange(B, device=points.device)[:, None].expand(B, N)

        # Select valid points
        valid_batch = batch_indices[valid_mask]
        valid_features = point_features.permute(0, 2, 1)[valid_mask]  # [valid_points, C]
        valid_coords = grid_coords[valid_mask]

        if len(valid_batch) > 0:
            # Linear indices for scatter
            linear_indices = valid_batch * (H * W) + valid_coords[:, 1] * W + valid_coords[:, 0]

            # Flatten feature map
            feature_map_flat = feature_map.view(B, self.feature_dim, H * W)

            # Expand indices to match feature dimensions
            expanded_indices = linear_indices.unsqueeze(1).expand(-1, self.feature_dim)

            # Scatter with max pooling (safe on CPU + GPU)
            feature_map_flat.scatter_reduce_(
                2,
                expanded_indices,
                valid_features.t(),  # [C, valid_points]
                reduce="amax",
                include_self=False
            )

            feature_map = feature_map_flat.view(B, self.feature_dim, H, W)

        return feature_map


# ================================================================
# PointPillars-based LiDAR Encoder (using MMDetection3D if available)
# ================================================================
try:
    from mmdet3d.models import PointPillarsEncoder
    MMDet3D_AVAILABLE = True
except ImportError:
    PointPillarsEncoder = None
    MMDet3D_AVAILABLE = False

class PointPillarsLiDAREncoder(nn.Module):
    def __init__(self, in_channels: int = 4, feat_channels: List[int] = [64, 128],
                 with_distance: bool = False, voxel_size: List[float] = [0.16, 0.16, 4],
                 point_cloud_range: List[float] = [-50, -50, -5, 50, 50, 3]):
        """
        LiDAR encoder using PointPillars from mmdet3d.
        """
        super().__init__()
        if not MMDet3D_AVAILABLE:
            raise ImportError("PointPillarsEncoder requires mmdet3d. Install with: pip install mmdet3d")

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.pillar_encoder = PointPillarsEncoder(
            in_channels=in_channels,
            feat_channels=feat_channels,
            with_distance=with_distance
        )

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Forward pass for PointPillars encoder."""
        bev_features = self.pillar_encoder(voxel_features, voxel_coords, batch_size)
        return bev_features


# ================================================================
# Unified LiDAR Encoder Factory
# ================================================================
class LiDAREncoder(nn.Module):
    def __init__(self, encoder_type: str = "spatial", **kwargs):
        """
        Unified LiDAR encoder that can switch between implementations.
        """
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "spatial":
            self.encoder = SpatialLiDAREncoder(**kwargs)
        elif encoder_type == "pointpillars":
            if not MMDet3D_AVAILABLE:
                print("⚠ mmdet3d not available → Falling back to SpatialLiDAREncoder")
                self.encoder = SpatialLiDAREncoder(**kwargs)
                self.encoder_type = "spatial"
            else:
                self.encoder = PointPillarsLiDAREncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.encoder(*args, **kwargs)

    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if self.encoder_type == "spatial":
            return (self.encoder.feature_dim, self.encoder.grid_size[0], self.encoder.grid_size[1])
        else:
            return (128, 32, 32)  # Example placeholder for PointPillars
