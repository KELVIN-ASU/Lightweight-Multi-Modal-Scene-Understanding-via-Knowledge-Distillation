# src/data_loading/data_interface.py
"""
Unified data interface that can switch between synthetic and real data
"""
from torch.utils.data import DataLoader
from typing import Tuple, Optional


def create_dataloaders(dataset_type: str = 'synthetic',
                      batch_size: int = 4,
                      num_workers: int = 4,
                      # Synthetic data args
                      num_train_samples: int = 800,
                      num_val_samples: int = 200,
                      # nuScenes args  
                      dataroot: Optional[str] = None,
                      version: str = 'v1.0-mini',
                      # Common args
                      image_size: Tuple[int, int] = (256, 256),
                      max_points: int = 5000) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for either synthetic or nuScenes data.
    This is the main interface that training code uses.
    """
    
    if dataset_type == 'synthetic':
        print(f"Loading synthetic dataset: {num_train_samples} train, {num_val_samples} val samples")
        from .synthetic_dataset import create_synthetic_dataloaders
        
        return create_synthetic_dataloaders(
            num_train=num_train_samples,
            num_val=num_val_samples,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
    elif dataset_type == 'nuscenes':
        print(f"Loading nuScenes dataset from: {dataroot}")
        
        if dataroot is None:
            raise ValueError("dataroot must be specified for nuScenes dataset")
            
        try:
            from nuscenes_dataset import create_dataloaders as create_nuscenes_loaders
            return create_nuscenes_loaders(
                dataroot=dataroot,
                batch_size=batch_size,
                num_workers=num_workers,
                version=version
            )
        except ImportError:
            print("nuScenes dataset not available, falling back to synthetic")
            from .synthetic_dataset import create_synthetic_dataloaders
            return create_synthetic_dataloaders(
                num_train=num_train_samples,
                num_val=num_val_samples,
                batch_size=batch_size,
                num_workers=num_workers
            )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'synthetic' or 'nuscenes'")


# Add nuScenes dataset here when available
def create_nuscenes_dataloaders_placeholder():
    """
    Placeholder for nuScenes data loading.
    Replace this with actual nuScenes implementation when ready.
    """
    pass