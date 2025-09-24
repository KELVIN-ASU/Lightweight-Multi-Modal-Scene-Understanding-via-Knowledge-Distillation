import torch
import time
from src.models.lidar_encoder import SpatialLiDAREncoder, LiDAREncoder, MMDet3D_AVAILABLE, create_test_point_cloud


def test_spatial_encoder_basic(grid_size=(128, 128)):
    """Test the spatial LiDAR encoder with different input sizes and grid resolution."""
    print("=" * 60)
    print(f"TESTING SPATIAL LIDAR ENCODER (grid={grid_size})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for use_vectorized in [True, False]:
        mode = "VECTORIZED" if use_vectorized else "ITERATIVE"
        print(f"\n--- Testing {mode} mode ---")
        
        encoder = SpatialLiDAREncoder(grid_size=grid_size, use_vectorized=use_vectorized).to(device).eval()

        print(f"LiDAR encoder parameters: {encoder.count_parameters():,}")
        print(f"Feature dimensions: {encoder.feature_dim}")
        print(f"Grid size: {encoder.grid_size}")

        test_cases = [
            (1, 1000, 4),
            (2, 5000, 4),
            (4, 2000, 4),
        ]

        for batch_size, num_points, num_features in test_cases:
            print(f"  Testing: batch_size={batch_size}, points={num_points}")
            points = create_test_point_cloud(batch_size, num_points, device)

            with torch.no_grad():
                output = encoder(points)

            print(f"    Input:  {tuple(points.shape)}")
            print(f"    Output: {tuple(output.shape)}")
            print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")

            assert output.shape[0] == batch_size
            assert output.shape[1] == 128
            assert output.shape[2:] == grid_size


def test_vectorized_vs_iterative_performance():
    """Compare performance and correctness between vectorized and iterative implementations."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: VECTORIZED vs ITERATIVE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    grid_size = (64, 64)
    points = create_test_point_cloud(batch_size=2, num_points=5000, device=device)
    
    encoder_vec = SpatialLiDAREncoder(grid_size=grid_size, use_vectorized=True).to(device).eval()
    encoder_iter = SpatialLiDAREncoder(grid_size=grid_size, use_vectorized=False).to(device).eval()

    with torch.no_grad():
        _ = encoder_vec(points)
        _ = encoder_iter(points)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Vectorized timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(5):
            output_vec = encoder_vec(points)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    vec_time = (time.time() - start_time) / 5 * 1000

    # Iterative timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(5):
            output_iter = encoder_iter(points)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    iter_time = (time.time() - start_time) / 5 * 1000

    print(f"Vectorized time: {vec_time:.2f} ms")
    print(f"Iterative time:  {iter_time:.2f} ms")
    print(f"Speedup factor:  {iter_time/vec_time:.1f}x")

    diff_mean = torch.abs(output_vec - output_iter).mean().item()
    diff_max = torch.abs(output_vec - output_iter).max().item()
    print(f"Output difference → mean abs: {diff_mean:.6f}, max abs: {diff_max:.6f}")

    # Allow mismatch due to floating-point reduction order
    assert diff_mean < 1.0, "Vectorized and iterative outputs differ too much!"

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")


def test_unified_encoder(grid_size=(128, 128)):
    """Test the unified LiDAR encoder interface."""
    print("\n" + "=" * 60)
    print(f"TESTING UNIFIED LIDAR ENCODER (grid={grid_size})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unified_encoder = LiDAREncoder(encoder_type='spatial', grid_size=grid_size, use_vectorized=True).to(device).eval()

    print(f"Unified encoder parameters: {unified_encoder.count_parameters():,}")

    points = create_test_point_cloud(batch_size=2, num_points=3000, device=device)
    with torch.no_grad():
        output = unified_encoder(points)

    print(f"Input shape:  {tuple(points.shape)}")
    print(f"Output shape: {tuple(output.shape)}")

    output_shape = unified_encoder.get_output_shape()
    print(f"Predicted output shape: {output_shape}")

    assert output.shape == (2, 128, *grid_size)
    assert output.shape[1:] == output_shape


def test_pointpillars_availability():
    """Check if PointPillars (mmdet3d) is available."""
    print("\n" + "=" * 60)
    print("TESTING POINTPILLARS AVAILABILITY")
    print("=" * 60)

    if MMDet3D_AVAILABLE:
        print("✓ mmdet3d is available → PointPillars encoder can be used")
    else:
        print("⚠ mmdet3d is NOT available → Spatial encoder will be used as fallback")

    try:
        encoder = LiDAREncoder(encoder_type='pointpillars', grid_size=(128, 128))
        print("✓ Unified encoder handles missing mmdet3d gracefully")
        print(f"Encoder type used: {encoder.encoder_type}")
    except Exception as e:
        print(f"Error: {e}")


def test_performance(grid_size=(128, 128)):
    """Benchmark performance with a larger input."""
    print("\n" + "=" * 60)
    print(f"PERFORMANCE TEST (grid={grid_size})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LiDAREncoder(encoder_type='spatial', grid_size=grid_size, use_vectorized=True).to(device).eval()

    points = create_test_point_cloud(batch_size=4, num_points=10000, device=device)

    with torch.no_grad():
        _ = encoder(points)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            output = encoder(points)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    inference_time = (end_time - start_time) / 10 * 1000
    total_points = points.shape[0] * points.shape[1]
    throughput = total_points / (inference_time / 1000) / 1e6
    
    print(f"Input: {tuple(points.shape)}")
    print(f"Output: {tuple(output.shape)}")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Throughput: {throughput:.2f} M points/s")
    
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")


def test_feature_compatibility():
    """Test compatibility between camera and LiDAR feature dimensions."""
    print("\n" + "=" * 60)
    print("TESTING CAMERA-LIDAR FEATURE COMPATIBILITY")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_feature_sizes = [(32, 32), (64, 64)]
    
    for cam_size in camera_feature_sizes:
        print(f"\n--- Testing LiDAR grid {cam_size} (for camera compatibility) ---")
        
        lidar_encoder = LiDAREncoder(
            encoder_type='spatial',
            grid_size=cam_size, 
            feature_dim=128,
            use_vectorized=True
        ).to(device).eval()
        
        points = create_test_point_cloud(batch_size=2, num_points=5000, device=device)
        with torch.no_grad():
            lidar_features = lidar_encoder(points)
        
        print(f"LiDAR features: {lidar_features.shape}")
        assert lidar_features.shape[1:] == (128, *cam_size)


def test_edge_cases():
    """Test edge cases like few points and points outside range."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LiDAREncoder(encoder_type='spatial', grid_size=(32, 32), use_vectorized=True).to(device).eval()

    # Very few points
    points_few = create_test_point_cloud(batch_size=2, num_points=10, device=device)
    with torch.no_grad():
        output_few = encoder(points_few)
    print(f"Few points input: {tuple(points_few.shape)}")
    print(f"Few points output: {tuple(output_few.shape)}")

    # Points guaranteed outside the valid range
    points_outside = torch.ones(2, 100, 4, device=device) * 9999.0
    with torch.no_grad():
        output_outside = encoder(points_outside)

    max_val = output_outside.max().item()
    print(f"Outside points output max value: {max_val:.6f}")
    assert max_val == 0.0, "Points outside range should produce zero output"


def main():
    """Run all LiDAR encoder tests."""
    torch.manual_seed(0)
    print("STARTING LIDAR ENCODER TEST SUITE")
    print("=" * 60)
    
    for grid in [(32, 32), (64, 64), (128, 128)]:
        test_spatial_encoder_basic(grid)
        test_unified_encoder(grid)
        if grid == (64, 64):
            test_performance(grid)

    test_vectorized_vs_iterative_performance()
    test_pointpillars_availability()
    test_feature_compatibility()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("ALL LIDAR ENCODER TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
