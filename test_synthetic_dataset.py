# test_synthetic_dataset.py
"""
Test program for synthetic dataset generation.
Run this before training to verify everything works correctly.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_loading.synthetic_dataset import SyntheticMultiModalDataset, create_synthetic_dataloaders


def test_basic_dataset():
    """Test basic dataset functionality"""
    print("=" * 50)
    print("TEST 1: Basic Dataset Functionality")
    print("=" * 50)
    
    # Create small dataset
    dataset = SyntheticMultiModalDataset(num_samples=10)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test single sample
    sample = dataset[0]
    
    print(f"Sample 0 shapes:")
    print(f"  Image: {sample['image'].shape}")
    print(f"  Points: {sample['points'].shape}")
    print(f"  Segmentation: {sample['segmentation'].shape}")
    print(f"  Sample token: {sample['sample_token']}")
    
    # Check data types with debugging info
    print(f"Data types:")
    print(f"  Image: {sample['image'].dtype}")
    print(f"  Points: {sample['points'].dtype}")
    print(f"  Segmentation: {sample['segmentation'].dtype}")
    
    assert sample['image'].dtype == torch.float32, f"Image dtype should be float32, got {sample['image'].dtype}"
    assert sample['points'].dtype == torch.float32, f"Points dtype should be float32, got {sample['points'].dtype}"
    assert sample['segmentation'].dtype == torch.int64, f"Segmentation dtype should be int64, got {sample['segmentation'].dtype}"
    
    # Check shapes
    assert sample['image'].shape == (3, 256, 256), f"Expected image shape (3, 256, 256), got {sample['image'].shape}"
    assert sample['points'].shape == (5000, 4), f"Expected points shape (5000, 4), got {sample['points'].shape}"
    assert sample['segmentation'].shape == (256, 256), f"Expected segmentation shape (256, 256), got {sample['segmentation'].shape}"
    
    print("✓ Basic dataset test PASSED")


def test_data_ranges():
    """Test that data values are in expected ranges"""
    print("\n" + "=" * 50)
    print("TEST 2: Data Value Ranges")
    print("=" * 50)
    
    dataset = SyntheticMultiModalDataset(num_samples=5)
    
    for i in range(3):
        sample = dataset[i]
        image = sample['image']
        points = sample['points']
        segmentation = sample['segmentation']
        
        print(f"Sample {i}:")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Points XYZ range: [{points[:, :3].min():.1f}, {points[:, :3].max():.1f}]")
        print(f"  Points intensity range: [{points[:, 3].min():.3f}, {points[:, 3].max():.3f}]")
        print(f"  Segmentation classes: {segmentation.unique().tolist()}")
        
        # Validate ranges
        assert 0 <= image.min() <= image.max() <= 1, "Image values should be [0,1]"
        assert 0 <= points[:, 3].min() <= points[:, 3].max() <= 1, "Intensity should be [0,1]"
        assert set(segmentation.unique().tolist()).issubset({0, 1, 2}), "Segmentation should have classes 0,1,2"
    
    print("✓ Data ranges test PASSED")


def test_dataloader():
    """Test dataloader batch creation"""
    print("\n" + "=" * 50)
    print("TEST 3: DataLoader Functionality")
    print("=" * 50)
    
    train_loader, val_loader = create_synthetic_dataloaders(
        num_train=20,
        num_val=10,
        batch_size=4,
        num_workers=0  # Use 0 to avoid multiprocessing issues in tests
    )
    
    print(f"Created dataloaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Test batch loading
    batch = next(iter(train_loader))
    
    print(f"Batch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Points: {batch['points'].shape}")
    print(f"  Segmentations: {batch['segmentation'].shape}")
    
    batch_size = batch['image'].shape[0]
    assert batch['image'].shape == (batch_size, 3, 256, 256)
    assert batch['points'].shape == (batch_size, 5000, 4)
    assert batch['segmentation'].shape == (batch_size, 256, 256)
    
    print("✓ DataLoader test PASSED")


def test_model_compatibility():
    """Test compatibility with your existing model"""
    print("\n" + "=" * 50)
    print("TEST 4: Model Compatibility")
    print("=" * 50)
    
    try:
        from src.models.camera_encoder import TwinLiteEncoder
        from src.models.lidar_encoder import LiDAREncoder
        from src.models.fusion_module import CompleteSegmentationModel
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create model
        camera_encoder = TwinLiteEncoder(return_multiscale=True)
        lidar_encoder = LiDAREncoder(encoder_type='spatial', grid_size=(64, 64), use_vectorized=True)
        model = CompleteSegmentationModel(
            camera_encoder=camera_encoder,
            lidar_encoder=lidar_encoder,
            num_classes=3,
            fusion_type='concat'
        ).to(device)
        
        print(f"Model parameter count: {model.count_parameters()['total']:,}")
        
        # Get synthetic data
        dataset = SyntheticMultiModalDataset(num_samples=1)
        sample = dataset[0]
        
        # Prepare batch
        image = sample['image'].unsqueeze(0).to(device)  # Add batch dim
        points = sample['points'].unsqueeze(0).to(device)
        target = sample['segmentation'].unsqueeze(0).to(device)
        
        print(f"Input shapes:")
        print(f"  Image: {image.shape}")
        print(f"  Points: {points.shape}")
        print(f"  Target: {target.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(image, points)
        
        print(f"Output shape: {output.shape}")
        
        # Validate output
        assert output.shape[0] == target.shape[0], "Batch size mismatch"
        assert output.shape[1] == 3, "Should have 3 classes"
        assert output.shape[2:] == target.shape[1:], "Spatial dimensions should match"
        
        print("✓ Model compatibility test PASSED")
        
    except ImportError as e:
        print(f"⚠ Model compatibility test SKIPPED: {e}")
        print("Make sure your model files are created first")


def test_data_variety():
    """Test that synthetic data has sufficient variety"""
    print("\n" + "=" * 50)
    print("TEST 5: Data Variety")
    print("=" * 50)
    
    dataset = SyntheticMultiModalDataset(num_samples=20)
    
    # Check segmentation pattern variety
    patterns = []
    for i in range(10):
        seg = dataset[i]['segmentation']
        pattern_hash = hash(seg.numpy().tobytes())
        patterns.append(pattern_hash)
    
    unique_patterns = len(set(patterns))
    print(f"Unique segmentation patterns in 10 samples: {unique_patterns}")
    
    # Check point cloud variety
    point_counts = []
    for i in range(5):
        points = dataset[i]['points']
        non_zero_points = (points.abs().sum(dim=1) > 0).sum().item()
        point_counts.append(non_zero_points)
    
    print(f"Non-zero points per sample: {point_counts}")
    
    assert unique_patterns >= 3, "Should have at least 3 different segmentation patterns"
    assert len(set(point_counts)) >= 2, "Point counts should vary between samples"
    
    print("✓ Data variety test PASSED")


def test_deterministic_generation():
    """Test that data generation is deterministic for same indices"""
    print("\n" + "=" * 50)
    print("TEST 6: Deterministic Generation")
    print("=" * 50)
    
    dataset1 = SyntheticMultiModalDataset(num_samples=10)
    dataset2 = SyntheticMultiModalDataset(num_samples=10)
    
    # Same index should generate same data
    sample1 = dataset1[0]
    sample2 = dataset2[0]
    
    image_diff = torch.abs(sample1['image'] - sample2['image']).max()
    points_diff = torch.abs(sample1['points'] - sample2['points']).max()
    seg_diff = torch.abs(sample1['segmentation'] - sample2['segmentation']).max()
    
    print(f"Differences between same index in different datasets:")
    print(f"  Image max diff: {image_diff:.6f}")
    print(f"  Points max diff: {points_diff:.6f}")
    print(f"  Segmentation max diff: {seg_diff:.6f}")
    
    assert image_diff < 1e-5, "Images should be identical for same index"
    assert points_diff < 1e-5, "Points should be identical for same index"
    assert seg_diff == 0, "Segmentation should be identical for same index"
    
    print("✓ Deterministic generation test PASSED")


def visualize_samples():
    """Create visualization of synthetic samples"""
    print("\n" + "=" * 50)
    print("VISUALIZATION: Creating Sample Images")
    print("=" * 50)
    
    try:
        dataset = SyntheticMultiModalDataset(num_samples=10)
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        for i in range(3):
            sample = dataset[i]
            
            # Convert tensors to numpy for visualization
            image = sample['image'].permute(1, 2, 0).numpy()
            segmentation = sample['segmentation'].numpy()
            
            # Plot image
            axes[0, i].imshow(image)
            axes[0, i].set_title(f'Camera Image {i}')
            axes[0, i].axis('off')
            
            # Plot segmentation
            seg_plot = axes[1, i].imshow(segmentation, cmap='viridis', vmin=0, vmax=2)
            axes[1, i].set_title(f'Segmentation {i}')
            axes[1, i].axis('off')
        
        # Add colorbar
        plt.tight_layout()
        cbar = plt.colorbar(seg_plot, ax=axes[1, :], orientation='horizontal', pad=0.1)
        cbar.set_label('Classes: 0=Background, 1=Drivable, 2=Lane')
        
        # Save visualization
        plt.savefig('synthetic_dataset_samples.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved as 'synthetic_dataset_samples.png'")
        
        # Try to display (will work in interactive environments)
        try:
            plt.show()
        except:
            print("(Display not available, but image file was saved)")
            
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
        print("This is not critical for functionality")


def main():
    """Run all tests"""
    print("SYNTHETIC DATASET TEST SUITE")
    print("=" * 50)
    
    try:
        test_basic_dataset()
        test_data_ranges()
        test_dataloader()
        test_model_compatibility()
        test_data_variety()
        test_deterministic_generation()
        
        # Optional visualization
        visualize_samples()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("Synthetic dataset is ready for training.")
        print("Next step: Run training program")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Fix the issue before proceeding to training")
        raise


if __name__ == "__main__":
    main()