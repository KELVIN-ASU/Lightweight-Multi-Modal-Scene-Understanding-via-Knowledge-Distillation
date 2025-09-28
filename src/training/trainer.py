# src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm
import numpy as np
import os
import json


class SegmentationMetrics:
    """Calculate mIoU and other segmentation metrics"""
    
    def __init__(self, num_classes: int = 3, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with predictions and targets
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] ground truth labels
        """
        pred = torch.argmax(pred, dim=1).cpu().numpy()  # [B, H, W]
        target = target.cpu().numpy()  # [B, H, W]
        
        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()
        
        # Remove ignore index
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute_iou(self) -> Dict[str, float]:
        """Compute IoU for each class and mIoU"""
        # IoU = TP / (TP + FP + FN)
        diag = np.diag(self.confusion_matrix)
        sum_over_row = np.sum(self.confusion_matrix, axis=1)  # TP + FN
        sum_over_col = np.sum(self.confusion_matrix, axis=0)  # TP + FP
        denominator = sum_over_row + sum_over_col - diag
        
        # Avoid division by zero
        iou = np.where(denominator > 0, diag / denominator, 0.)
        
        class_names = ['background', 'drivable_area', 'lane']
        
        results = {}
        for i, name in enumerate(class_names[:self.num_classes]):
            results[f'{name}_iou'] = float(iou[i])
        
        results['miou'] = float(np.mean(iou))
        return results
    
    def compute_accuracy(self) -> float:
        """Compute pixel accuracy"""
        total_correct = np.trace(self.confusion_matrix)
        total_pixels = np.sum(self.confusion_matrix)
        return float(total_correct / total_pixels) if total_pixels > 0 else 0.0


class BaselineTrainer:
    """Trainer for baseline student model (supervised learning only)"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 save_dir: str = 'checkpoints'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100,  # Will be adjusted based on actual epochs
            eta_min=lr * 0.01
        )
        
        # Metrics
        self.train_metrics = SegmentationMetrics(num_classes=3)
        self.val_metrics = SegmentationMetrics(num_classes=3)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_miou': [],
            'val_loss': [],
            'val_miou': [],
            'lr': []
        }
        
        self.best_miou = 0.0
        self.epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            points = batch['points'].to(self.device)
            targets = batch['segmentation'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            logits = self.model(images, points)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(logits.detach(), targets)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # Compute epoch metrics
        train_iou = self.train_metrics.compute_iou()
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'miou': train_iou['miou'],
            'drivable_iou': train_iou['drivable_area_iou'],
            'lane_iou': train_iou['lane_iou']
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {self.epoch}')
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.device)
                points = batch['points'].to(self.device)
                targets = batch['segmentation'].to(self.device)
                
                # Forward pass
                logits = self.model(images, points)
                loss = self.criterion(logits, targets)
                
                # Update metrics
                total_loss += loss.item()
                self.val_metrics.update(logits, targets)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
        
        # Compute epoch metrics
        val_iou = self.val_metrics.compute_iou()
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'miou': val_iou['miou'],
            'drivable_iou': val_iou['drivable_area_iou'],
            'lane_iou': val_iou['lane_iou']
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'history': self.history,
            'model_config': self.model.get_architecture_summary()
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best.pth'))
            
        # Save history as JSON
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self, num_epochs: int) -> Dict:
        """Train the model for specified number of epochs"""
        
        # Adjust scheduler
        self.scheduler.T_max = num_epochs
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {self.model.count_parameters()}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_results = self.train_epoch()
            
            # Validate epoch
            val_results = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_results['loss'])
            self.history['train_miou'].append(train_results['miou'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_miou'].append(val_results['miou'])
            self.history['lr'].append(current_lr)
            
            # Check if best model
            is_best = val_results['miou'] > self.best_miou
            if is_best:
                self.best_miou = val_results['miou']
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Loss: {train_results['loss']:.4f}, mIoU: {train_results['miou']:.4f}")
            print(f"  Val   - Loss: {val_results['loss']:.4f}, mIoU: {val_results['miou']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            if is_best:
                print(f"  New best mIoU: {self.best_miou:.4f}")
            
            print("-" * 50)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best validation mIoU: {self.best_miou:.4f}")
        
        return self.history


def speed_benchmark(model: nn.Module, device: torch.device, 
                   input_size: Tuple[int, int] = (256, 256),
                   batch_size: int = 1,
                   num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model inference speed"""
    
    model.eval()
    
    # Create dummy data
    dummy_image = torch.randn(batch_size, 3, *input_size, device=device)
    dummy_points = torch.randn(batch_size, 5000, 4, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_image, dummy_points)
    
    # Synchronize GPU if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_image, dummy_points)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_runs) * 1000
    fps = 1000 / avg_time_ms
    
    return {
        'avg_inference_time_ms': avg_time_ms,
        'fps': fps,
        'total_benchmark_time_s': total_time,
        'num_runs': num_runs
    }