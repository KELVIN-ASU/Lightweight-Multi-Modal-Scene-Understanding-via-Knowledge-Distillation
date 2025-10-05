import os
import torch
import json
from src.data_loading.pandaset_dataset import create_pandaset_dataloaders
from src.models.camera_encoder import TwinLiteEncoder
from src.models.lidar_encoder import LiDAREncoder
from src.models.fusion_module import CompleteSegmentationModel
from src.training.trainer import Trainer

def train_fusion_variant(fusion_type, fusion_out_channels, root, train_scenes, val_scenes, device):
    """Train a single fusion variant."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {fusion_type.upper()} FUSION")
    print(f"{'='*80}")
    
    # Data loaders
    train_loader, val_loader = create_pandaset_dataloaders(
        root=root,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        batch_size=4,
        num_workers=2,
        verbose=False
    )
    
    # Model
    cam_enc = TwinLiteEncoder(return_multiscale=True)
    lidar_enc = LiDAREncoder(encoder_type="spatial", grid_size=(64, 64), use_vectorized=True)
    
    model = CompleteSegmentationModel(
        camera_encoder=cam_enc,
        lidar_encoder=lidar_enc,
        num_classes=2,
        fusion_type=fusion_type,
        fusion_out_channels=fusion_out_channels,
        camera_fpn_stages=["stage3", "stage4", "stage5"],
        camera_fpn_channels=128,
        output_mode="same"
    ).to(device)
    
    summary = model.get_architecture_summary()
    print(f"\nModel: {fusion_type}")
    print(f"  Total params: {summary['total_params']}")
    print(f"  Fusion params: {summary['fusion_params']}")
    
    # Class weights for 2-class (based on actual distribution)
    class_weights = [0.4, 3.5]
    
    # Trainer
    save_dir = f"checkpoints/fusion_ablation_{fusion_type}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        weight_decay=1e-3,
        save_dir=save_dir,
        class_weights=class_weights,
        num_epochs= 20
    )
    
    # Train
    best_miou = trainer.train()
    
    return best_miou, summary['total_params'], summary['fusion_params']


def main():
    root = r"D:\kelvin\Dataset\data"
    all_scenes = sorted([d for d in os.listdir(root) if d.isdigit()])
    n_train = int(0.8 * len(all_scenes))
    train_scenes, val_scenes = all_scenes[:n_train], all_scenes[n_train:]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"FUSION ABLATION STUDY - 2-CLASS DRIVABLE AREA SEGMENTATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Scenes: {len(train_scenes)} train, {len(val_scenes)} val")
    print(f"{'='*80}\n")
    
    # Test all fusion strategies
    results = {}
    
    # 1. Concat fusion
    miou, total_params, fusion_params = train_fusion_variant(
        "concat", 256, root, train_scenes, val_scenes, device
    )
    results["concat"] = {
        "miou": miou, 
        "total_params": total_params,
        "fusion_params": fusion_params
    }
    
    # 2. Minimal fusion
    miou, total_params, fusion_params = train_fusion_variant(
        "minimal", 128, root, train_scenes, val_scenes, device
    )
    results["minimal"] = {
        "miou": miou,
        "total_params": total_params,
        "fusion_params": fusion_params
    }
    
    # 3. Weighted fusion
    miou, total_params, fusion_params = train_fusion_variant(
        "weighted", 128, root, train_scenes, val_scenes, device
    )
    results["weighted"] = {
        "miou": miou,
        "total_params": total_params,
        "fusion_params": fusion_params
    }
    
    # Print comparison
    print(f"\n{'='*80}")
    print("FUSION ABLATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Fusion':<12} {'mIoU':>8} {'Total Params':>15} {'Fusion Params':>15}")
    print("-"*80)
    for ftype, data in results.items():
        print(f"{ftype:<12} {data['miou']:>8.4f} {data['total_params']:>15} {data['fusion_params']:>15}")
    
    best_fusion = max(results.items(), key=lambda x: x[1]['miou'])
    print(f"\n{'='*80}")
    print(f"BEST FUSION: {best_fusion[0].upper()}")
    print(f"  mIoU: {best_fusion[1]['miou']:.4f}")
    print(f"  Total params: {best_fusion[1]['total_params']}")
    print(f"{'='*80}\n")
    
    # Save results
    with open("fusion_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to fusion_ablation_results.json")

if __name__ == "__main__":
    main()