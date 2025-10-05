import os
import numpy as np
from tqdm import tqdm
from src.data_loading.pandaset_dataset import create_pandaset_dataloaders

def main():
    root = r"D:\kelvin\Dataset\data"
    all_scenes = sorted([d for d in os.listdir(root) if d.isdigit()])
    n_train = int(0.8 * len(all_scenes))
    train_scenes, val_scenes = all_scenes[:n_train], all_scenes[n_train:]
    
    train_loader, val_loader = create_pandaset_dataloaders(
        root, train_scenes, val_scenes, batch_size=4, num_workers=0, verbose=False
    )
    
    print("\n" + "="*60)
    print("2-CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for loader, name in [(train_loader, "TRAIN"), (val_loader, "VAL")]:
        class_pixels = np.zeros(2, dtype=np.int64)
        
        for batch in tqdm(loader, desc=f"Scanning {name}"):
            seg = batch["segmentation"].numpy()
            class_pixels[0] += (seg == 0).sum()
            class_pixels[1] += (seg == 1).sum()
        
        total = class_pixels.sum()
        print(f"\n{name}:")
        print(f"  Background: {class_pixels[0]:,} ({class_pixels[0]/total*100:.2f}%)")
        print(f"  Drivable:   {class_pixels[1]:,} ({class_pixels[1]/total*100:.2f}%)")
        print(f"  Class weights: [1.0, {total/(2*class_pixels[1]):.2f}]")

if __name__ == "__main__":
    main()