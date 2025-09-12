import torch
import time
from src.models.lidar_encoder import SpatialLiDAREncoder, LiDAREncoder, MMDet3D_AVAILABLE


def test_spatial_encoder(grid_size=(128, 128)):
    """Test the spatial LiDAR encoder with different input sizes and grid resolution."""
    print("=" * 60)
    print(f"TESTING SPATIAL LIDAR ENCODER (grid={grid_size})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = SpatialLiDAREncoder(grid_size=grid_size).to(device)
    encoder.eval()

    test_cases = [
        (1, 1000, 4),   # Single point cloud
        (2, 5000, 4),   # Batch of 2
        (4, 2000, 4),   # Batch of 4
    ]

    for batch_size, num_points, num_features in test_cases:
        print(f"\n--- Testing: batch_size={batch_size}, points={num_points} ---")

        # Random test points
        points = torch.randn(batch_size, num_points, num_features).to(device)
        points[..., 0] = points[..., 0] * 20 + 10   # x ~ [-30, 50]
        points[..., 1] = points[..., 1] * 15 - 5    # y ~ [-35, 40]
        points[..., 2] = points[..., 2] * 2         # z (height)
        points[..., 3] = torch.sigmoid(points[..., 3])  # intensity [0, 1]

        with torch.no_grad():
            output = encoder(points)

        print(f"Input shape:  {tuple(points.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

        assert output.shape[0] == batch_size, "Wrong batch size"
        assert output.shape[1] == 128, "Feature dim should be 128"
        assert output.shape[2:] == grid_size, f"Grid shape mismatch {output.shape[2:]}"


def test_unified_encoder(grid_size=(128, 128)):
    """Test the unified LiDAR encoder interface."""
    print("\n" + "=" * 60)
    print(f"TESTING UNIFIED LIDAR ENCODER (grid={grid_size})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unified_encoder = LiDAREncoder(encoder_type='spatial', grid_size=grid_size).to(device)
    unified_encoder.eval()

    points = torch.randn(2, 3000, 4).to(device)
    points[..., 0] = points[..., 0] * 25
    points[..., 1] = points[..., 1] * 20

    with torch.no_grad():
        output = unified_encoder(points)

    print(f"Unified encoder output shape: {tuple(output.shape)}")
    assert output.shape == (2, 128, *grid_size)

    output_shape = unified_encoder.get_output_shape((2, 3000, 4))
    print(f"Predicted output shape: {output_shape}")


def test_pointpillars_availability():
    """Check if PointPillars (mmdet3d) is available."""
    print("\n" + "=" * 60)
    print("TESTING POINTPILLARS AVAILABILITY")
    print("=" * 60)

    if MMDet3D_AVAILABLE:
        print("mmdet3d is available → PointPillars encoder can be used")
    else:
        print("mmdet3d is NOT available → Spatial encoder will be used as fallback")

    try:
        encoder = LiDAREncoder(encoder_type='pointpillars', grid_size=(128, 128))
        print("✓ Unified encoder handles missing mmdet3d gracefully")
    except Exception as e:
        print(f"Error: {e}")


def test_performance(grid_size=(128, 128)):
    """Test performance with a larger input."""
    print("\n" + "=" * 60)
    print(f"PERFORMANCE TEST (grid={grid_size})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SpatialLiDAREncoder(grid_size=grid_size).to(device)
    encoder.eval()

    points = torch.randn(4, 10000, 4).to(device)
    points[..., 0] = points[..., 0] * 30
    points[..., 1] = points[..., 1] * 25

    # Warmup
    with torch.no_grad():
        _ = encoder(points)

    # Time forward pass
    start_time = time.time()
    with torch.no_grad():
        output = encoder(points)
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000  # ms
    print(f"Input: {tuple(points.shape)}")
    print(f"Output: {tuple(output.shape)}")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Points processed per second: {points.numel() / (end_time - start_time) / 1e6:.2f} M points/s")


if __name__ == "__main__":
    # Run tests for both low- and high-resolution grids
    for grid in [(32, 32), (128, 128)]:
        test_spatial_encoder(grid)
        test_unified_encoder(grid)
        test_performance(grid)

    test_pointpillars_availability()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
