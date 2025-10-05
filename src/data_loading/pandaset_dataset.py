import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------
# Label remapping: PandaSet → {0=background, 1=drivable (includes lanes)}
# --------------------
_DRIVABLE = {6, 7, 8, 9, 10, 12}  # Ground, Road, Lane markings, Stop lines, Other markings, Driveway

def remap_semantic(raw_ids: np.ndarray) -> np.ndarray:
    """Map PandaSet raw class IDs into {0=background, 1=drivable}."""
    mapped = np.zeros_like(raw_ids, dtype=np.int64)
    for k in _DRIVABLE:
        mapped[raw_ids == k] = 1
    return mapped


def rasterize_bev(
    x: np.ndarray, y: np.ndarray, labels: np.ndarray,
    grid_size: Tuple[int, int] = (64, 64),
    pc_range: Tuple[float, float, float, float] = (-50, 50, -50, 50)
) -> np.ndarray:
    """Rasterize per-point labels into a BEV mask {0,1}."""
    H, W = grid_size
    x_min, x_max, y_min, y_max = pc_range

    mask = np.zeros((H, W), dtype=np.int64)
    m = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    x, y, labels = x[m], y[m], labels[m]

    if x.size == 0:
        return mask

    col = np.clip(((x - x_min) / (x_max - x_min) * (W - 1)).astype(int), 0, W - 1)
    row = np.clip(((y - y_min) / (y_max - y_min) * (H - 1)).astype(int), 0, H - 1)

    for r, c, lab in zip(row, col, labels):
        if mask[r, c] == 0:
            mask[r, c] = lab
    return mask


class PandaSetDataset(Dataset):
    """
    2-class version: background (0) and drivable (1, includes lanes).
    """
    def __init__(
        self,
        root: str,
        scene_ids: List[str],
        image_size: Tuple[int, int] = (256, 256),
        grid_size: Tuple[int, int] = (64, 64),
        max_points: int = 5000,
        verbose: bool = True
    ):
        self.root = root
        self.scene_ids = scene_ids
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_points = max_points
        self.pc_range = (-50, 50, -50, 50)

        self.samples = self._index_scenes(verbose=verbose)
        if verbose:
            print(f"Indexed {len(self.samples)} valid samples from {len(scene_ids)} scenes")

    def _index_scenes(self, verbose: bool = True):
        samples = []
        for sid in self.scene_ids:
            cam_dir   = os.path.join(self.root, sid, "camera", "front_camera")
            lidar_dir = os.path.join(self.root, sid, "lidar")
            seg_dir   = os.path.join(self.root, sid, "annotations", "semseg")
            if not (os.path.isdir(cam_dir) and os.path.isdir(lidar_dir) and os.path.isdir(seg_dir)):
                continue

            frames = sorted(f[:-4] for f in os.listdir(cam_dir) if f.endswith(".jpg"))
            count_before, count_after = len(frames), 0
            for fid in frames:
                cam_path   = os.path.join(cam_dir, f"{fid}.jpg")
                lidar_path = os.path.join(lidar_dir, f"{fid}.pkl")
                seg_path   = os.path.join(seg_dir, f"{fid}.pkl")

                if os.path.exists(cam_path) and os.path.exists(lidar_path) and os.path.exists(seg_path):
                    samples.append({
                        "scene": sid,
                        "frame": fid,
                        "image": cam_path,
                        "lidar": lidar_path,
                        "semseg": seg_path,
                    })
                    count_after += 1
            if verbose:
                print(f"Scene {sid}: {count_after}/{count_before} frames usable")
        return samples

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # --- Camera image ---
        img = Image.open(s["image"]).convert("RGB")
        img = img.resize(self.image_size, Image.BILINEAR)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2,0,1).contiguous()

        # --- LiDAR points ---
        lidar_df = pd.read_pickle(s["lidar"])
        x = lidar_df["x"].to_numpy(dtype=np.float32)
        y = lidar_df["y"].to_numpy(dtype=np.float32)
        z = lidar_df["z"].to_numpy(dtype=np.float32)
        i = lidar_df["i"].to_numpy(dtype=np.float32)
        pts = np.stack([x, y, z, i], axis=1)

        if pts.shape[0] > self.max_points:
            choice = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[choice]
        elif pts.shape[0] < self.max_points:
            pad = np.zeros((self.max_points - pts.shape[0], 4), dtype=np.float32)
            pts = np.vstack([pts, pad])
        pts_t = torch.from_numpy(pts).contiguous()

        # --- Semseg to BEV mask (2-class) ---
        sem_df = pd.read_pickle(s["semseg"])
        raw_ids = sem_df["class"].to_numpy(dtype=np.int64)
        ids2 = remap_semantic(raw_ids)
        bev = rasterize_bev(x, y, ids2, grid_size=self.grid_size, pc_range=self.pc_range)
        bev_t = torch.from_numpy(bev.astype(np.int64))

        return {
            "image": img_t,
            "points": pts_t,
            "segmentation": bev_t,
            "sample_token": f"{s['scene']}_{s['frame']}",
        }


def create_pandaset_dataloaders(
    root: str,
    train_scenes: List[str],
    val_scenes: List[str],
    batch_size: int = 4,
    num_workers: int = 0,
    verbose: bool = True
):
    train_ds = PandaSetDataset(root, train_scenes, verbose=verbose)
    val_ds   = PandaSetDataset(root, val_scenes, verbose=verbose)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader