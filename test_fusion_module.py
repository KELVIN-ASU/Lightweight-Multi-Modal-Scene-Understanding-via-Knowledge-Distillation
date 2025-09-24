import torch
from src.models.camera_encoder import TwinLiteEncoder
from src.models.lidar_encoder import LiDAREncoder
from src.models.fusion_module import ConcatenationFusion, LightweightSegmentationHead, CompleteSegmentationModel

def test_fusion_module():
    """Test the fusion module independently"""
    print("=" * 60)
    print("TESTING FUSION MODULE")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create fusion module
    fusion = ConcatenationFusion(
        camera_channels=128,
        lidar_channels=128, 
        output_channels=256
    ).to(device)
    
    fusion.eval()
    
    # Test inputs (matching encoder outputs)
    camera_features = torch.randn(2, 128, 32, 32).to(device)
    lidar_features = torch.randn(2, 128, 32, 32).to(device)
    
    with torch.no_grad():
        fused_output = fusion(camera_features, lidar_features)
    
    print(f"Camera features: {camera_features.shape}")
    print(f"LiDAR features: {lidar_features.shape}")
    print(f"Fused output: {fused_output.shape}")
    print(f"Fusion parameters: {fusion.count_parameters():,}")
    
    assert fused_output.shape == (2, 256, 32, 32), "Fusion output shape mismatch"
    print("✅ Fusion module test passed")


def test_segmentation_head():
    """Test the lightweight segmentation head independently"""
    print("\n" + "=" * 60)
    print("TESTING LIGHTWEIGHT SEGMENTATION HEAD")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create lightweight segmentation head
    seg_head = LightweightSegmentationHead(
        in_channels=256,
        num_classes=3,
        upsample_factor=8
    ).to(device)
    
    seg_head.eval()
    
    # Test input (output from fusion)
    fused_features = torch.randn(2, 256, 32, 32).to(device)
    
    with torch.no_grad():
        segmentation_output = seg_head(fused_features)
    
    print(f"Fused features: {fused_features.shape}")
    print(f"Segmentation output: {segmentation_output.shape}")
    print(f"Segmentation head parameters: {seg_head.count_parameters():,}")
    
    # Should upsample 32x32 -> 256x256 (8x upsampling)
    expected_shape = (2, 3, 256, 256)
    assert segmentation_output.shape == expected_shape, f"Expected {expected_shape}, got {segmentation_output.shape}"
    print("✅ Lightweight segmentation head test passed")


def test_complete_model():
    """Test the complete end-to-end model"""
    print("\n" + "=" * 60) 
    print("TESTING COMPLETE SEGMENTATION MODEL")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize encoders
    camera_encoder = TwinLiteEncoder().to(device)
    lidar_encoder = LiDAREncoder(encoder_type="spatial", grid_size=(32, 32)).to(device)
    
    # Create complete model
    model = CompleteSegmentationModel(
        camera_encoder=camera_encoder,
        lidar_encoder=lidar_encoder,
        fusion_channels=256,
        num_classes=3
    ).to(device)
    
    model.eval()
    
    # Test with realistic inputs
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_points = torch.randn(batch_size, 5000, 4).to(device)
    
    print(f"Input images: {dummy_images.shape}")
    print(f"Input points: {dummy_points.shape}")
    
    with torch.no_grad():
        segmentation_output = model(dummy_images, dummy_points)
    
    print(f"Segmentation output: {segmentation_output.shape}")
    
    # Check output shape
    expected_shape = (batch_size, 3, 256, 256)
    assert segmentation_output.shape == expected_shape, f"Expected {expected_shape}, got {segmentation_output.shape}"
    
    # Parameter analysis
    param_breakdown = model.count_parameters()
    print("\nParameter Breakdown:")
    for component, count in param_breakdown.items():
        if component == 'total':
            budget_usage = count / 1_000_000 * 100
            print(f"  {component}: {count:,} ({budget_usage:.1f}% of 1M budget)")
        else:
            print(f"  {component}: {count:,}")
    
    # Architecture summary
    summary = model.get_architecture_summary()
    print(f"\nArchitecture: {summary['architecture']}")
    print(f"Under budget: {'✅' if summary['under_budget'] else '❌'}")
    
    print("✅ Complete model test passed")
    
    return param_breakdown['total']


def test_different_input_sizes():
    """Test model with different input image sizes"""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT INPUT SIZES")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize encoders
    camera_encoder = TwinLiteEncoder().to(device)
    lidar_encoder = LiDAREncoder(encoder_type="spatial", grid_size=(32, 32)).to(device)
    
    # Create complete model
    model = CompleteSegmentationModel(
        camera_encoder=camera_encoder,
        lidar_encoder=lidar_encoder,
        fusion_channels=256,
        num_classes=3
    ).to(device)
    
    model.eval()
    
    # Test different image sizes
    test_sizes = [(128, 128), (256, 256), (512, 512)]
    
    for h, w in test_sizes:
        print(f"\n--- Testing {h}x{w} input ---")
        
        dummy_images = torch.randn(1, 3, h, w).to(device)
        dummy_points = torch.randn(1, 3000, 4).to(device)
        
        with torch.no_grad():
            output = model(dummy_images, dummy_points)
        
        print(f"Input: {dummy_images.shape} -> Output: {output.shape}")
        
        # Output should match input spatial dimensions
        assert output.shape[-2:] == (h, w), f"Output size {output.shape[-2:]} doesn't match input {(h, w)}"
    
    print("✅ Different input sizes test passed")


def test_gradient_flow():
    """Test that gradients flow properly through the model"""
    print("\n" + "=" * 60)
    print("TESTING GRADIENT FLOW")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize encoders
    camera_encoder = TwinLiteEncoder().to(device)
    lidar_encoder = LiDAREncoder(encoder_type="spatial", grid_size=(32, 32)).to(device)
    
    # Create complete model in training mode
    model = CompleteSegmentationModel(
        camera_encoder=camera_encoder,
        lidar_encoder=lidar_encoder,
        fusion_channels=256,
        num_classes=3
    ).to(device)
    
    model.train()  # Training mode for gradient computation
    
    # Test inputs
    dummy_images = torch.randn(2, 3, 256, 256).to(device)
    dummy_points = torch.randn(2, 5000, 4).to(device)
    
    # Forward pass
    output = model(dummy_images, dummy_points)
    
    # Dummy loss (sum of all outputs)
    dummy_loss = output.sum()
    
    # Backward pass
    dummy_loss.backward()
    
    # Check gradients
    params_with_grad = 0
    total_params = 0
    params_without_grad = []
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
        else:
            params_without_grad.append(name)
    
    print(f"Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"Gradient coverage: {params_with_grad/total_params*100:.1f}%")
    
    if params_without_grad:
        print(f"Parameters without gradients: {params_without_grad}")
    
    assert params_with_grad == total_params, "Some parameters missing gradients"
    print("✅ Gradient flow test passed")


def test_output_ranges():
    """Test that model outputs are in reasonable ranges"""
    print("\n" + "=" * 60)
    print("TESTING OUTPUT RANGES")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize encoders
    camera_encoder = TwinLiteEncoder().to(device)
    lidar_encoder = LiDAREncoder(encoder_type="spatial", grid_size=(32, 32)).to(device)
    
    # Create complete model
    model = CompleteSegmentationModel(
        camera_encoder=camera_encoder,
        lidar_encoder=lidar_encoder,
        fusion_channels=256,
        num_classes=3
    ).to(device)
    
    model.eval()
    
    # Test inputs
    dummy_images = torch.randn(2, 3, 256, 256).to(device)
    dummy_points = torch.randn(2, 5000, 4).to(device)
    
    with torch.no_grad():
        output = model(dummy_images, dummy_points)
    
    print(f"Output shape: {output.shape}")
    print(f"Output min: {output.min().item():.3f}")
    print(f"Output max: {output.max().item():.3f}")
    print(f"Output mean: {output.mean().item():.3f}")
    print(f"Output std: {output.std().item():.3f}")
    
    # Check for reasonable ranges (logits should be roughly in [-10, 10])
    assert output.min() > -50 and output.max() < 50, "Output values seem unreasonable"
    print("✅ Output ranges test passed")


def main():
    """Run all fusion module tests"""
    print("STARTING UPDATED FUSION MODULE TESTS")
    print("=" * 60)
    
    try:
        # Test individual components
        test_fusion_module()
        test_segmentation_head()
        
        # Test complete model
        total_params = test_complete_model()
        
        # Test flexibility
        test_different_input_sizes()
        
        # Test training readiness
        test_gradient_flow()
        
        # Test output quality
        test_output_ranges()
        
        # Summary
        print("\n" + "=" * 60)
        print("FUSION MODULE TESTS SUMMARY")
        print("=" * 60)
        print(f"✅ All tests passed successfully!")
        print(f"📊 Total model parameters: {total_params:,}")
        print(f"💾 Parameter budget usage: {total_params/1_000_000*100:.1f}% of 1M")
        
        if total_params < 1_000_000:
            print(f"🎯 Model meets parameter constraint (<1M)")
            remaining_budget = 1_000_000 - total_params
            print(f"📈 Remaining parameter budget: {remaining_budget:,}")
        else:
            print(f"⚠️  Model exceeds parameter constraint (>1M)")
            
        print("\n🚀 Model is ready for training and evaluation!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()