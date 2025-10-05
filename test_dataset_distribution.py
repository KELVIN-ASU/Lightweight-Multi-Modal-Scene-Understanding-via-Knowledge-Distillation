# test_dataset_distribution.py
import os
import numpy as np
from tqdm import tqdm
from src.data_loading.pandaset_dataset import create_pandaset_dataloaders

def analyze_class_distribution(loader, name=""):
    """Analyze class distribution across the entire dataset"""
    print(f"\n{'='*60}")
    print(f"Analyzing {name} Dataset")
    print(f"{'='*60}")
    
    class_pixels = np.zeros(3, dtype=np.int64)
    total_samples = 0
    
    for batch in tqdm(loader, desc=f"Scanning {name}"):
        seg = batch["segmentation"].numpy()
        for cls in range(3):
            class_pixels[cls] += (seg == cls).sum()
        total_samples += seg.shape[0]
    
    total_pixels = class_pixels.sum()
    
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Total pixels: {total_pixels:,} ({seg.shape[1]}×{seg.shape[2]} grid)")
    print(f"\nClass Distribution:")
    
    class_names = ["Background", "Drivable", "Lane"]
    for cls in range(3):
        pct = class_pixels[cls] / total_pixels * 100
        print(f"  Class {cls} ({class_names[cls]:10s}): {class_pixels[cls]:,} ({pct:5.2f}%)")
    
    # Calculate class weights for balanced loss
    weights = total_pixels / (3 * class_pixels)
    print(f"\nRecommended class weights for CrossEntropyLoss:")
    print(f"  weights = {weights.tolist()}")
    
    # Check for severe imbalance
    bg_pct = class_pixels[0] / total_pixels * 100
    if bg_pct > 95:
        print(f"\nSEVERE IMBALANCE: {bg_pct:.1f}% background!")
        print("    → Model can get high accuracy by always predicting background")
    elif bg_pct > 85:
        print(f"\nHigh background: {bg_pct:.1f}%")
        print("    → Strongly recommend using class weights")
    
    return class_pixels, total_pixels

def main():
    root = r"D:\kelvin\Dataset\data"
    all_scenes = sorted([d for d in os.listdir(root) if d.isdigit()])
    n_train = int(0.8 * len(all_scenes))
    train_scenes, val_scenes = all_scenes[:n_train], all_scenes[n_train:]
    
    train_loader, val_loader = create_pandaset_dataloaders(
        root=root,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        batch_size=4,
        num_workers=0,
        verbose=False
    )
    
    train_pixels, train_total = analyze_class_distribution(train_loader, "TRAINING")
    val_pixels, val_total = analyze_class_distribution(val_loader, "VALIDATION")
    
    # Compare distributions
    print(f"\n{'='*60}")
    print("Train vs Val Distribution Comparison")
    print(f"{'='*60}")
    train_pct = train_pixels / train_total * 100
    val_pct = val_pixels / val_total * 100
    
    class_names = ["Background", "Drivable", "Lane"]
    for cls in range(3):
        diff = abs(train_pct[cls] - val_pct[cls])
        print(f"{class_names[cls]:10s}: Train {train_pct[cls]:5.2f}% | Val {val_pct[cls]:5.2f}% | Diff {diff:.2f}%")
    
    max_diff = abs(train_pct - val_pct).max()
    if max_diff > 10:
        print(f"\nLarge distribution mismatch (max diff: {max_diff:.1f}%)")
        print("    → Train and val scenes may be from different environments")

if __name__ == "__main__":
    main()