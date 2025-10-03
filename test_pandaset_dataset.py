import os
from src.data_loading.pandaset_dataset import create_pandaset_dataloaders

def main():
    # Adjust this to your PandaSet data folder
    root = r"D:\kelvin\Dataset\data"

    # Find all scene folders (001...047)
    all_scenes = sorted([d for d in os.listdir(root) if d.isdigit()])
    print(f"Found {len(all_scenes)} scenes")

    # 80/20 split
    n_train = int(0.8 * len(all_scenes))
    train_scenes, val_scenes = all_scenes[:n_train], all_scenes[n_train:]

    print(f"Train: {len(train_scenes)} scenes, Val: {len(val_scenes)} scenes")

    # Create loaders
    train_loader, val_loader = create_pandaset_dataloaders(
        root=root,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        batch_size=2,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Check one batch
    batch = next(iter(train_loader))
    print("Image:", batch["image"].shape)
    print("Points:", batch["points"].shape)
    print("Segmentation:", batch["segmentation"].shape)
    print("Sample token:", batch["sample_token"])

if __name__ == "__main__":
    main()
