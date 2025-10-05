# import os
# import torch
# from src.data_loading.pandaset_dataset import create_pandaset_dataloaders
# from src.models.camera_encoder import TwinLiteEncoder
# from src.models.lidar_encoder import LiDAREncoder
# from src.models.fusion_module import CompleteSegmentationModel
# from src.training.trainer import Trainer

# def main():
#     root = r"D:\kelvin\Dataset\data"
#     all_scenes = sorted([d for d in os.listdir(root) if d.isdigit()])
#     n_train = int(0.8 * len(all_scenes))
#     train_scenes, val_scenes = all_scenes[:n_train], all_scenes[n_train:]

#     train_loader, val_loader = create_pandaset_dataloaders(
#         root=root,
#         train_scenes=train_scenes,
#         val_scenes=val_scenes,
#         batch_size=4,
#         num_workers=2
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cam_enc = TwinLiteEncoder(return_multiscale=True)
#     lidar_enc = LiDAREncoder(encoder_type="spatial", grid_size=(64,64), use_vectorized=True)
#     model = CompleteSegmentationModel(
#         camera_encoder=cam_enc,
#         lidar_encoder=lidar_enc,
#         num_classes=3,
#         fusion_type="concat",
#         fusion_out_channels=256,
#         camera_fpn_stages=["stage3","stage4","stage5"],
#         camera_fpn_channels=128,
#         output_mode="same"
#     ).to(device)

#     trainer = Trainer(model, train_loader, val_loader, device, save_dir="checkpoints/pandaset")

#     epochs = 20
#     start_epoch = 0
#     ckpt_path = os.path.join("checkpoints/pandaset", "latest.pth")
#     if os.path.exists(ckpt_path):
#         start_epoch = trainer.load_checkpoint(ckpt_path)

#     for epoch in range(start_epoch, epochs):
#         print(f"\n===== Epoch {epoch+1}/{epochs} =====")
#         train_loss, train_metrics = trainer.train_epoch()
#         val_loss, val_metrics = trainer.validate()

#         train_miou = train_metrics["miou"]
#         val_miou = val_metrics["miou"]
#         lr = trainer.optimizer.param_groups[0]["lr"]

#         print(f"Train Loss: {train_loss:.4f}, mIoU: {train_miou:.3f}")
#         print(f"Val   Loss: {val_loss:.4f}, mIoU: {val_miou:.3f}")

#         # update history json
#         trainer.update_history(train_loss, train_miou, val_loss, val_miou, lr)

#         # save checkpoint
#         is_best = val_miou > trainer.best_miou
#         if is_best:
#             trainer.best_miou = val_miou
#         trainer.save_checkpoint(epoch, val_miou, is_best=is_best)

# if __name__ == "__main__":
#     main()



import os
import torch
from src.data_loading.pandaset_dataset import create_pandaset_dataloaders
from src.models.camera_encoder import TwinLiteEncoder
from src.models.lidar_encoder import LiDAREncoder
from src.models.fusion_module import CompleteSegmentationModel
from src.training.trainer import Trainer

def main():
    # Dataset path
    root = r"D:\kelvin\Dataset\data"
    
    # Scene split
    all_scenes = sorted([d for d in os.listdir(root) if d.isdigit()])
    n_train = int(0.8 * len(all_scenes))
    train_scenes, val_scenes = all_scenes[:n_train], all_scenes[n_train:]
    
    print(f"Found {len(all_scenes)} scenes")
    print(f"Train: {len(train_scenes)} scenes | Val: {len(val_scenes)} scenes")

    # Create data loaders
    train_loader, val_loader = create_pandaset_dataloaders(
        root=root,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        batch_size=4,
        num_workers=2,
        verbose=True
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Build model
    print("\nBuilding model...")
    cam_enc = TwinLiteEncoder(return_multiscale=True)
    lidar_enc = LiDAREncoder(
        encoder_type="spatial", 
        grid_size=(64, 64), 
        use_vectorized=True
    )
    
    model = CompleteSegmentationModel(
        camera_encoder=cam_enc,
        lidar_encoder=lidar_enc,
        num_classes=3,
        fusion_type="concat",
        fusion_out_channels=256,
        camera_fpn_stages=["stage3", "stage4", "stage5"],
        camera_fpn_channels=128,
        output_mode="same"  # 64x64 output to match BEV labels
    ).to(device)
    
    # Print model summary
    summary = model.get_architecture_summary()
    print(f"\nModel Architecture:")
    print(f"  Camera params:  {summary['camera_params']}")
    print(f"  LiDAR params:   {summary['lidar_params']}")
    print(f"  Fusion params:  {summary['fusion_params']}")
    print(f"  Head params:    {summary['head_params']}")
    print(f"  Total params:   {summary['total_params']}")
    
    # CRITICAL: Class weights from distribution analysis
    # Training distribution: 86.21% background, 12.78% drivable, 1.01% lane
    class_weights = [0.39, 2.61, 33.09]
    
    # Training parameters
    num_epochs = 30
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        weight_decay=1e-3,  # Increased from 1e-4 for better regularization
        save_dir="checkpoints/pandaset_weighted",
        class_weights=class_weights,
        num_epochs=num_epochs
    )
    
    # Check for existing checkpoint
    start_epoch = 0
    ckpt_path = os.path.join("checkpoints/pandaset_weighted", "latest.pth")
    if os.path.exists(ckpt_path):
        response = input(f"\nFound checkpoint at {ckpt_path}. Resume training? (y/n): ")
        if response.lower() == 'y':
            start_epoch = trainer.load_checkpoint(ckpt_path)
    
    # Train
    trainer.train(start_epoch=start_epoch)

if __name__ == "__main__":
    main()