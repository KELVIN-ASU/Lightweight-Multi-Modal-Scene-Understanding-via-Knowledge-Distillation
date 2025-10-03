# src/data_loading/download_pandaset.py
"""
Download PandaSet dataset using kagglehub
Run once before training: python -m src.data_loading.download_pandaset
"""

import kagglehub
import os
import sys

def download_pandaset(target_dir=None):
    """
    Download PandaSet dataset from Kaggle
    
    Args:
        target_dir: Optional custom directory. If None, uses kagglehub default
    
    Returns:
        Path to downloaded dataset
    """
    
    if target_dir:
        target_dir = os.path.expanduser(target_dir)
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            print(f"Dataset already exists at {target_dir}")
            response = input("Re-download? (y/n): ")
            if response.lower() != 'y':
                return target_dir
    
    print("Downloading PandaSet dataset (33.26 GB)...")
    print("This will take 1-3 hours depending on your connection...")
    
    try:
        path = kagglehub.dataset_download("usharengaraju/pandaset-dataset")
        print(f"\nDownload complete!")
        print(f"Dataset location: {path}")
        
        # Save path for future reference
        with open('.pandaset_path.txt', 'w') as f:
            f.write(path)
        
        return path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    path = download_pandaset()
    print(f"\nPandaSet ready at: {path}")
    print("\nNext steps:")
    print("1. Verify dataset structure")
    print("2. Implement PandaSet dataloader")
    print("3. Start training")