import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes=2, ignore_index=-1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self): 
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            mask = (t != self.ignore_index)
            p, t = p[mask].flatten(), t[mask].flatten()
            for i in range(len(p)):
                if 0 <= t[i] < self.num_classes and 0 <= p[i] < self.num_classes:
                    self.confusion[t[i], p[i]] += 1
    
    def compute(self):
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion[i, i]
            fp = self.confusion[:, i].sum() - tp
            fn = self.confusion[i, :].sum() - tp
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            ious.append(iou)
        return {"class_iou": ious, "miou": float(np.mean(ious))}


class Trainer:
    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-3, weight_decay=1e-3, save_dir="checkpoints",
                 class_weights=None, num_epochs=20):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Use class weights if provided
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            print(f"Using class weights: {class_weights.tolist()}")
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-5
        )

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.best_miou = 0.0
        self.history_path = os.path.join(save_dir, "training_history.json")
        self.history = {
            "train_loss": [], 
            "train_miou": [],
            "val_loss": [], 
            "val_miou": [],
            "lr": []
        }

    def train_epoch(self):
        self.model.train()
        metrics = SegmentationMetrics(num_classes=2)
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc="Train"):
            imgs = batch["image"].to(self.device)
            pts = batch["points"].to(self.device)
            seg = batch["segmentation"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs, pts)
            loss = self.criterion(logits, seg)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            metrics.update(logits.detach(), seg)

        return total_loss / len(self.train_loader), metrics.compute()

    def validate(self):
        self.model.eval()
        metrics = SegmentationMetrics(num_classes=2)
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val"):
                imgs = batch["image"].to(self.device)
                pts = batch["points"].to(self.device)
                seg = batch["segmentation"].to(self.device)

                logits = self.model(imgs, pts)
                loss = self.criterion(logits, seg)

                total_loss += loss.item()
                metrics.update(logits, seg)

        return total_loss / len(self.val_loader), metrics.compute()

    def save_checkpoint(self, epoch, val_miou, is_best=False):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_miou": val_miou,
        }
        latest_path = os.path.join(self.save_dir, "latest.pth")
        torch.save(ckpt, latest_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best.pth")
            torch.save(ckpt, best_path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        
        self.best_miou = ckpt.get("val_miou", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {path}, starting at epoch {start_epoch}, best mIoU {self.best_miou:.4f}")
        return start_epoch

    def update_history(self, train_loss, train_miou, val_loss, val_miou, lr):
        self.history["train_loss"].append(train_loss)
        self.history["train_miou"].append(train_miou)
        self.history["val_loss"].append(val_loss)
        self.history["val_miou"].append(val_miou)
        self.history["lr"].append(lr)

        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def train(self, start_epoch=0):
        """Main training loop"""
        print(f"\nStarting training from epoch {start_epoch + 1}/{self.num_epochs}")
        print("="*60)
        
        for epoch in range(start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-"*60)
            
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            train_miou = train_metrics["miou"]
            val_miou = val_metrics["miou"]
            
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val mIoU:   {val_miou:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            print(f"\n  Per-class IoU (Val):")
            class_names = ["Background", "Drivable"]
            for i, (name, iou) in enumerate(zip(class_names, val_metrics["class_iou"])):
                print(f"    {name:12s}: {iou:.4f}")
            
            self.update_history(train_loss, train_miou, val_loss, val_miou, current_lr)
            
            is_best = val_miou > self.best_miou
            if is_best:
                self.best_miou = val_miou
                print(f"  New best mIoU: {val_miou:.4f}")
            self.save_checkpoint(epoch, val_miou, is_best=is_best)
        
        print("\n" + "="*60)
        print(f"Training completed! Best validation mIoU: {self.best_miou:.4f}")
        print("="*60)
        
        return self.best_miou