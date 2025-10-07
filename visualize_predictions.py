import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.data_loading.pandaset_dataset import create_pandaset_dataloaders
from src.models.camera_encoder import TwinLiteEncoder
from src.models.lidar_encoder import LiDAREncoder
from src.models.fusion_module import CompleteSegmentationModel

def visualize_predictions(model, val_loader, device, num_samples=8, save_dir="visualizations"):
    """Visualize model predictions vs ground truth."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Custom colormap: black (background), green (drivable)
    colors = ['black', 'green']
    cmap = ListedColormap(colors)
    
    samples_visualized = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if samples_visualized >= num_samples:
                break
                
            imgs = batch["image"].to(device)
            pts = batch["points"].to(device)
            seg_gt = batch["segmentation"].numpy()
            tokens = batch["sample_token"]
            
            # Predictions
            logits = model(imgs, pts)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            batch_size = imgs.shape[0]
            
            for i in range(batch_size):
                if samples_visualized >= num_samples:
                    break
                
                # Create figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 1. Input image
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                axes[0].imshow(img_np)
                axes[0].set_title("Input Camera Image")
                axes[0].axis('off')
                
                # 2. Ground truth BEV
                axes[1].imshow(seg_gt[i], cmap=cmap, vmin=0, vmax=1)
                axes[1].set_title("Ground Truth BEV")
                axes[1].axis('off')
                
                # 3. Prediction BEV
                axes[2].imshow(preds[i], cmap=cmap, vmin=0, vmax=1)
                axes[2].set_title("Predicted BEV")
                axes[2].axis('off')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='black', label='Background'),
                    Patch(facecolor='green', label='Drivable')
                ]
                fig.legend(handles=legend_elements, loc='lower center', ncol=2)
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)
                
                # Save
                save_path = os.path.join(save_dir, f"sample_{samples_visualized:03d}_{tokens[i]}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                samples_visualized += 1
                print(f"Saved visualization {samples_visualized}/{num_samples}")
    
    print(f"\nAll visualizations saved to {save_dir}/")


def compute_iou_per_sample(pred, gt):
    """Compute IoU for drivable class."""
    pred_drivable = (pred == 1)
    gt_drivable = (gt == 1)
    
    intersection = (pred_drivable & gt_drivable).sum()
    union = (pred_drivable | gt_drivable).sum()
    
    if union == 0:
        return 0.0
    return intersection / union


def visualize_with_metrics(model, val_loader, device, num_samples=8, save_dir="visualizations_with_metrics"):
    """Visualize predictions with per-sample IoU scores."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    colors = ['black', 'green']
    cmap = ListedColormap(colors)
    
    samples_visualized = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if samples_visualized >= num_samples:
                break
                
            imgs = batch["image"].to(device)
            pts = batch["points"].to(device)
            seg_gt = batch["segmentation"].numpy()
            tokens = batch["sample_token"]
            
            logits = model(imgs, pts)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            batch_size = imgs.shape[0]
            
            for i in range(batch_size):
                if samples_visualized >= num_samples:
                    break
                
                # Compute IoU for this sample
                iou_drivable = compute_iou_per_sample(preds[i], seg_gt[i])
                
                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                
                # Input image
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                axes[0].imshow(img_np)
                axes[0].set_title("Input Camera Image", fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                # Ground truth
                axes[1].imshow(seg_gt[i], cmap=cmap, vmin=0, vmax=1)
                axes[1].set_title("Ground Truth BEV", fontsize=12, fontweight='bold')
                axes[1].axis('off')
                axes[1].text(0.5, -0.1, f"Drivable pixels: {(seg_gt[i] == 1).sum()}", 
                           ha='center', transform=axes[1].transAxes, fontsize=10)
                
                # Prediction
                axes[2].imshow(preds[i], cmap=cmap, vmin=0, vmax=1)
                axes[2].set_title(f"Prediction (IoU: {iou_drivable:.3f})", 
                                fontsize=12, fontweight='bold', 
                                color='green' if iou_drivable > 0.5 else 'red')
                axes[2].axis('off')
                axes[2].text(0.5, -0.1, f"Drivable pixels: {(preds[i] == 1).sum()}", 
                           ha='center', transform=axes[2].transAxes, fontsize=10)
                
                # Legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='black', label='Background'),
                    Patch(facecolor='green', label='Drivable Area')
                ]
                fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11)
                
                plt.suptitle(f"Sample: {tokens[i]}", fontsize=14, fontweight='bold', y=0.98)
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.12, top=0.95)
                
                save_path = os.path.join(save_dir, f"sample_{samples_visualized:03d}_iou{iou_drivable:.3f}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                samples_visualized += 1
                print(f"Saved: {save_path.split('/')[-1]} (IoU: {iou_drivable:.3f})")
    
    print(f"\nAll visualizations saved to {save_dir}/")


def main():
    # Load validation data
    root = r"D:\kelvin\Dataset\data"
    all_scenes = sorted([d for d in os.listdir(root) if d.isdigit()])
    n_train = int(0.8 * len(all_scenes))
    train_scenes, val_scenes = all_scenes[:n_train], all_scenes[n_train:]
    
    _, val_loader = create_pandaset_dataloaders(
        root=root,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        batch_size=4,
        num_workers=0,
        verbose=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load best concat model
    cam_enc = TwinLiteEncoder(return_multiscale=True)
    lidar_enc = LiDAREncoder(encoder_type="spatial", grid_size=(64, 64), use_vectorized=True)
    
    model = CompleteSegmentationModel(
        camera_encoder=cam_enc,
        lidar_encoder=lidar_enc,
        num_classes=2,
        fusion_type="concat",
        fusion_out_channels=256,
        camera_fpn_stages=["stage3", "stage4", "stage5"],
        camera_fpn_channels=128,
        output_mode="same"
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/fusion_ablation_concat/best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from {checkpoint_path}")
    print(f"Validation mIoU: {checkpoint['val_miou']:.4f}\n")
    
    # Generate visualizations
    print("Generating basic visualizations...")
    visualize_predictions(model, val_loader, device, num_samples=12, save_dir="visualizations")
    
    print("\nGenerating visualizations with metrics...")
    visualize_with_metrics(model, val_loader, device, num_samples=12, save_dir="visualizations_with_metrics")
    
    print("\nDone! Check the 'visualizations/' and 'visualizations_with_metrics/' folders.")


if __name__ == "__main__":
    main()