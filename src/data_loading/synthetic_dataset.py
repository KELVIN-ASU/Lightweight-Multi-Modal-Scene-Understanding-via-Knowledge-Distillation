# src/data_loading/synthetic_dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


class SyntheticMultiModalDataset(Dataset):
    """
    Generates synthetic camera images, LiDAR point clouds, and segmentation masks
    for testing multi-modal segmentation and replace with nuScenes dataset when pipeline is ready.
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 image_size: Tuple[int, int] = (256, 256),
                 max_points: int = 5000,
                 point_cloud_range: list = [-50, -50, -5, 50, 50, 3]):
        
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_points = max_points
        self.point_cloud_range = point_cloud_range
        
        # Pre-create road patterns for variety
        self.road_patterns = self._create_road_patterns()
        
    def _create_road_patterns(self) -> list:
        """Create different road/lane layout patterns"""
        patterns = []
        H, W = self.image_size
        
        # Pattern 1: Straight highway
        mask1 = np.zeros((H, W), dtype=np.uint8)
        mask1[int(0.3*H):, :] = 1  # Drivable area
        # Lane markings
        for x_pos in [W//4, W//2, 3*W//4]:
            mask1[:, x_pos-2:x_pos+2] = 2
        patterns.append(mask1)
        
        # Pattern 2: City street
        mask2 = np.zeros((H, W), dtype=np.uint8)
        mask2[int(0.4*H):, :] = 1  # Smaller drivable area
        # Center lane only
        mask2[:, W//2-1:W//2+1] = 2
        patterns.append(mask2)
        
        # Pattern 3: Multi-lane highway
        mask3 = np.zeros((H, W), dtype=np.uint8)
        mask3[int(0.2*H):, :] = 1  # Large drivable area
        # Multiple lanes
        for i in range(1, 6):
            x_pos = i * W // 6
            mask3[:, x_pos-1:x_pos+1] = 2
        patterns.append(mask3)
        
        return patterns
    
    def _generate_camera_image(self, idx: int) -> np.ndarray:
        """Generate realistic camera image with road scene"""
        H, W = self.image_size
        np.random.seed(idx)  # Deterministic per index
        
        # Base scene
        image = np.ones((H, W, 3), dtype=np.float32) * 0.5  # Gray road base
        
        # Sky gradient
        for y in range(H//3):
            sky_intensity = 0.7 + 0.3 * (1 - y / (H//3))
            image[y, :] = [sky_intensity * 0.8, sky_intensity * 0.9, sky_intensity]
        
        # Add road texture
        road_noise = np.random.normal(0, 0.05, (H, W, 3))
        image = np.clip(image + road_noise, 0, 1)
        
        # Add some "vehicles" as colored rectangles
        num_objects = np.random.randint(1, 4)
        for i in range(num_objects):
            obj_x = np.random.randint(W//4, 3*W//4)
            obj_y = np.random.randint(H//2, H)
            obj_w = np.random.randint(20, 60)
            obj_h = np.random.randint(15, 40)
            
            color = np.random.uniform(0.2, 0.8, 3)
            y_end = min(obj_y + obj_h, H)
            x_end = min(obj_x + obj_w, W)
            image[obj_y:y_end, obj_x:x_end] = color
        
        return image
    
    def _generate_lidar_points(self, idx: int) -> np.ndarray:
        """Generate realistic LiDAR point cloud"""
        np.random.seed(idx)  # Deterministic per index
        
        # Ground plane points (majority of points)
        num_ground = int(self.max_points * 0.7)
        ground_points = np.random.uniform(
            [self.point_cloud_range[0], self.point_cloud_range[1], -2.0], 
            [self.point_cloud_range[3], self.point_cloud_range[4], -0.5], 
            (num_ground, 3)
        )
        ground_intensity = np.random.uniform(0.1, 0.4, (num_ground, 1))
        
        # Object points (vehicles, buildings)
        num_objects = int(self.max_points * 0.3)
        object_centers = np.random.uniform(
            [-30, -30, -1], [30, 30, 2], (np.random.randint(5, 15), 3)
        )
        
        object_points = []
        for center in object_centers:
            # Points around each object
            obj_size = np.random.uniform([1, 1, 0.5], [4, 4, 2])
            num_obj_points = np.random.randint(10, 50)
            
            obj_pts = np.random.uniform(
                center - obj_size, center + obj_size, (num_obj_points, 3)
            )
            object_points.append(obj_pts)
        
        if object_points:
            object_points = np.vstack(object_points)
            # Limit to remaining budget
            if len(object_points) > num_objects:
                indices = np.random.choice(len(object_points), num_objects, replace=False)
                object_points = object_points[indices]
            object_intensity = np.random.uniform(0.4, 0.9, (len(object_points), 1))
        else:
            object_points = np.empty((0, 3))
            object_intensity = np.empty((0, 1))
        
        # Combine points
        all_points = np.vstack([ground_points, object_points])
        all_intensity = np.vstack([ground_intensity, object_intensity])
        points_with_intensity = np.hstack([all_points, all_intensity])
        
        # Ensure exact number of points
        if len(points_with_intensity) > self.max_points:
            indices = np.random.choice(len(points_with_intensity), self.max_points, replace=False)
            points_with_intensity = points_with_intensity[indices]
        elif len(points_with_intensity) < self.max_points:
            padding = np.zeros((self.max_points - len(points_with_intensity), 4))
            points_with_intensity = np.vstack([points_with_intensity, padding])
        
        return points_with_intensity.astype(np.float32)
    
    def _generate_segmentation(self, idx: int) -> np.ndarray:
        """Generate segmentation mask based on road patterns"""
        np.random.seed(idx)  # Deterministic per index
        
        # Choose pattern
        pattern_idx = idx % len(self.road_patterns)
        mask = self.road_patterns[pattern_idx].copy()
        
        # Add random variations
        if np.random.random() > 0.5:
            mask = np.fliplr(mask)  # Flip horizontally
        
        # Add some noise to lane markings
        lane_pixels = (mask == 2)
        if np.any(lane_pixels):
            noise_prob = 0.95  # 5% of lane pixels become drivable
            noise_mask = np.random.random(mask.shape) > noise_prob
            mask[lane_pixels & noise_mask] = 1
        
        return mask.astype(np.int64)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate and return a single sample"""
        
        # Generate all components
        image = self._generate_camera_image(idx)
        points = self._generate_lidar_points(idx)
        segmentation = self._generate_segmentation(idx)
        
        # Convert to PyTorch tensors with explicit dtype
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # [C, H, W]
        points_tensor = torch.from_numpy(points).float()  # [max_points, 4]
        seg_tensor = torch.from_numpy(segmentation).long()  # [H, W]
        
        return {
            'image': image_tensor,
            'points': points_tensor,
            'segmentation': seg_tensor,
            'sample_token': f'synthetic_{idx:06d}',
            'camera_info': {'camera_intrinsic': torch.eye(3)},
            'lidar_info': {'lidar_token': f'lidar_{idx:06d}'}
        }


def create_synthetic_dataloaders(num_train: int = 800, 
                               num_val: int = 200,
                               batch_size: int = 4,
                               num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders for synthetic data"""
    
    train_dataset = SyntheticMultiModalDataset(num_samples=num_train)
    val_dataset = SyntheticMultiModalDataset(num_samples=num_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader