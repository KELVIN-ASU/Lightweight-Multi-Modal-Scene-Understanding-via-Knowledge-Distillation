# train_baseline.py
"""
Training program for baseline multi-modal segmentation model.
Easily switch between synthetic and nuScenes data by changing one argument.
"""
import torch
import argparse
import os
from pathlib import Path

from src.models.camera_encoder import TwinLiteEncoder
from src.models.lidar_encoder import LiDAREncoder  
from src.models.fusion_module import CompleteSegmentationModel
from src.data_loading.data_interface import create_dataloaders
from src.training.trainer import BaselineTrainer, speed_benchmark


def create_model(args, device):
    """Create and initialize the model"""
    
    camera_encoder = TwinLiteEncoder(
        base_channels=args.camera_base_channels,
        return_multiscale=True
    )
    
    lidar_encoder = LiDAREncoder(
        encoder_type='spatial',
        grid_size=tuple(args.lidar_grid_size),
        use_vectorized=True
    )
    
    model = CompleteSegmentationModel(
        camera_encoder=camera_encoder,
        lidar_encoder=lidar_encoder,
        num_classes=3,  # background, drivable_area, lane
        fusion_type=args.fusion_type,
        fusion_out_channels=args.fusion_out_channels,
        camera_fpn_stages=['stage3', 'stage4', 'stage5'],
        camera_fpn_channels=128
    ).to(device)
    
    return model


def print_model_summary(model):
    """Print model architecture summary"""
    arch_summary = model.get_architecture_summary()
    
    print("Model Architecture Summary:")
    print("-" * 50)
    for key, value in arch_summary.items():
        if key == 'params':
            print("Parameter breakdown:")
            for component, count in value.items():
                print(f"  {component}: {count:,}")
        else:
            print(f"{key}: {value}")
    
    total_params = arch_summary['params']['total']
    budget_status = "WITHIN BUDGET" if total_params < 1_000_000 else "EXCEEDS BUDGET"
    print(f"\nParameter Budget: {budget_status} ({total_params:,} / 1,000,000)")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='Train baseline multi-modal segmentation model')
    
    # Dataset selection (KEY PARAMETER FOR SWITCHING)
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'nuscenes'],
                       help='Dataset to use: synthetic or nuscenes')
    
    # Synthetic dataset arguments
    parser.add_argument('--num_train_samples', type=int, default=800,
                       help='Number of synthetic training samples')
    parser.add_argument('--num_val_samples', type=int, default=200,
                       help='Number of synthetic validation samples')
    
    # nuScenes dataset arguments  
    parser.add_argument('--dataroot', type=str, default=None,
                       help='Path to nuScenes dataset root (required for nuscenes)')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                       choices=['v1.0-mini', 'v1.0-trainval'],
                       help='nuScenes dataset version')
    
    # Model arguments
    parser.add_argument('--fusion_type', type=str, default='concat',
                       choices=['concat', 'minimal'],
                       help='Fusion strategy: concat or minimal')
    parser.add_argument('--fusion_out_channels', type=int, default=256,
                       help='Output channels from fusion module')
    parser.add_argument('--camera_base_channels', type=int, default=32,
                       help='Base channels for camera encoder')
    parser.add_argument('--lidar_grid_size', type=int, nargs=2, default=[64, 64],
                       help='LiDAR BEV grid size [H, W]')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Hardware arguments  
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    # Save/load arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints/baseline',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Testing flags
    parser.add_argument('--test_only', action='store_true',
                       help='Only test model and data pipeline')
    parser.add_argument('--speed_test', action='store_true',
                       help='Run speed benchmark')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset == 'nuscenes' and args.dataroot is None:
        parser.error("--dataroot is required when using nuScenes dataset")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("BASELINE MULTI-MODAL SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    if args.dataset == 'synthetic':
        print(f"Training samples: {args.num_train_samples}")
        print(f"Validation samples: {args.num_val_samples}")
    else:
        print(f"Data root: {args.dataroot}")
        print(f"Version: {args.version}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Fusion type: {args.fusion_type}")
    print("=" * 60)
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("Building model...")
    model = create_model(args, device)
    print_model_summary(model)
    
    # Create data loaders (THIS IS WHERE THE MAGIC HAPPENS)
    print("Creating data loaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            dataset_type=args.dataset,
            batch_size=args.batch_size,
            num_workers=0 if args.test_only else args.num_workers,
            # Synthetic args
            num_train_samples=args.num_train_samples,
            num_val_samples=args.num_val_samples,
            # nuScenes args
            dataroot=args.dataroot,
            version=args.version
        )
    except Exception as e:
        print(f"Failed to create {args.dataset} dataloaders: {e}")
        if args.dataset == 'nuscenes':
            print("Falling back to synthetic dataset...")
            train_loader, val_loader = create_dataloaders(
                dataset_type='synthetic',
                batch_size=args.batch_size,
                num_workers=0 if args.test_only else args.num_workers,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples
            )
        else:
            raise
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test data pipeline
    print("\nTesting data pipeline...")
    sample_batch = next(iter(train_loader))
    
    print(f"Batch shapes:")
    print(f"  Images: {sample_batch['image'].shape}")
    print(f"  Points: {sample_batch['points'].shape}")
    print(f"  Segmentation: {sample_batch['segmentation'].shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        test_images = sample_batch['image'][:1].to(device)
        test_points = sample_batch['points'][:1].to(device)
        test_output = model(test_images, test_points)
        print(f"  Forward pass output: {test_output.shape}")
        print(f"  Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    
    print("Data pipeline test PASSED")
    
    # Speed benchmark
    if args.speed_test or args.test_only:
        print("\nRunning speed benchmark...")
        speed_results = speed_benchmark(model, device, input_size=(256, 256), batch_size=1)
        print(f"Inference performance:")
        print(f"  Average time: {speed_results['avg_inference_time_ms']:.1f} ms")
        print(f"  FPS: {speed_results['fps']:.1f}")
        
        if speed_results['fps'] >= 30:
            print("Speed requirement MET (>= 30 FPS)")
        else:
            print("Speed requirement NOT MET (< 30 FPS)")
    
    # Exit if test only
    if args.test_only:
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("Model and data pipeline are ready for training.")
        print("Remove --test_only flag to start training.")
        print("=" * 60)
        return
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_miou = checkpoint['best_miou']
        trainer.epoch = checkpoint['epoch']
        trainer.history = checkpoint['history']
        print(f"Resumed from epoch {trainer.epoch}, best mIoU: {trainer.best_miou:.4f}")
    
    # Start training
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    try:
        history = trainer.train(args.epochs)
        
        # Final speed benchmark
        print("\nRunning final speed benchmark...")
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best.pth'))['model_state_dict'])
        final_speed = speed_benchmark(model, device, input_size=(256, 256))
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Dataset used: {args.dataset}")
        print(f"Best validation mIoU: {trainer.best_miou:.4f}")
        print(f"Final inference speed: {final_speed['fps']:.1f} FPS")
        print(f"Total parameters: {model.count_parameters()['total']:,}")
        print(f"Checkpoints saved to: {args.save_dir}")
        
        # Week 1 milestone check
        print("\nWeek 1 Baseline Milestone Check:")
        param_check = model.count_parameters()['total'] < 1_000_000
        speed_check = final_speed['fps'] >= 30
        
        print(f"  Parameter budget (<1M): {'PASS' if param_check else 'FAIL'}")
        print(f"  Speed requirement (>=30 FPS): {'PASS' if speed_check else 'FAIL'}")
        print(f"  Training convergence: {'PASS' if trainer.best_miou > 0.3 else 'FAIL'}")
        
        if param_check and speed_check:
            print("\nBaseline implementation COMPLETE!\nYou can now proceed to knowledge distillation.")
        else:
            print("\nSome requirements not met. Consider model optimization.")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Partial results saved to: {args.save_dir}")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == '__main__':
    main()