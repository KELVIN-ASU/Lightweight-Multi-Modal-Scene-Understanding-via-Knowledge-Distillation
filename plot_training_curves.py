#!/usr/bin/env python3
"""
Generate training curves visualization from fusion ablation experiments.
Reads training_history.json files from checkpoint directories and creates
publication-ready plots.

Usage:
    python plot_training_curves.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_history(checkpoint_dir):
    """Load training history from checkpoint directory."""
    history_path = Path(checkpoint_dir) / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    
    with open(history_path, 'r') as f:
        return json.load(f)

def plot_training_curves(histories, save_path='visualizations/training_curves.png'):
    """
    Create comprehensive training curves visualization.
    
    Args:
        histories: dict mapping fusion type to training history
        save_path: output path for the figure
    """
    # Define colors and markers for each fusion type
    styles = {
        'concat': {'color': '#1f77b4', 'marker': 'o', 'label': 'Concatenation'},
        'minimal': {'color': '#ff7f0e', 'marker': 's', 'label': 'Minimal'},
        'weighted': {'color': '#2ca02c', 'marker': '^', 'label': 'Weighted'}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = np.arange(1, 21)
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for fusion_type, history in histories.items():
        style = styles[fusion_type]
        ax.plot(epochs, history['train_loss'], 
                marker=style['marker'], linewidth=2, markersize=5,
                label=style['label'], color=style['color'])
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('(a) Training Loss Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1, 20)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for fusion_type, history in histories.items():
        style = styles[fusion_type]
        ax.plot(epochs, history['val_loss'],
                marker=style['marker'], linewidth=2, markersize=5,
                label=style['label'], color=style['color'])
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('(b) Validation Loss Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1, 20)
    
    # Plot 3: Training mIoU
    ax = axes[1, 0]
    for fusion_type, history in histories.items():
        style = styles[fusion_type]
        ax.plot(epochs, np.array(history['train_miou']) * 100,
                marker=style['marker'], linewidth=2, markersize=5,
                label=style['label'], color=style['color'])
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training mIoU (%)', fontsize=12, fontweight='bold')
    ax.set_title('(c) Training mIoU Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1, 20)
    ax.set_ylim(55, 80)
    
    # Plot 4: Validation mIoU with peak markers
    ax = axes[1, 1]
    peak_info = {}
    
    for fusion_type, history in histories.items():
        style = styles[fusion_type]
        val_miou_pct = np.array(history['val_miou']) * 100
        
        ax.plot(epochs, val_miou_pct,
                marker=style['marker'], linewidth=2, markersize=5,
                label=style['label'], color=style['color'])
        
        # Find and mark peak
        peak_idx = np.argmax(history['val_miou'])
        peak_value = history['val_miou'][peak_idx] * 100
        peak_epoch = peak_idx + 1
        
        ax.scatter([peak_epoch], [peak_value],
                   s=150, marker='*', color=style['color'],
                   edgecolor='black', linewidth=1.5, zorder=10)
        
        peak_info[fusion_type] = {
            'epoch': peak_epoch,
            'value': peak_value,
            'color': style['color']
        }
    
    # Add annotations for peaks
    # Concatenation
    info = peak_info['concat']
    ax.annotate(f'Peak: {info["value"]:.2f}%\n(Epoch {info["epoch"]})',
                xy=(info['epoch'], info['value']),
                xytext=(info['epoch'] + 2, info['value'] + 1),
                fontsize=9, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=info['color'], alpha=0.2))
    
    # Minimal
    info = peak_info['minimal']
    ax.annotate(f'Peak: {info["value"]:.2f}%\n(Epoch {info["epoch"]})',
                xy=(info['epoch'], info['value']),
                xytext=(info['epoch'] - 2, info['value'] - 2),
                fontsize=9, ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=info['color'], alpha=0.2))
    
    # Weighted
    info = peak_info['weighted']
    ax.annotate(f'Peak: {info["value"]:.2f}%\n(Epoch {info["epoch"]})',
                xy=(info['epoch'], info['value']),
                xytext=(info['epoch'] + 2, info['value'] - 1.5),
                fontsize=9, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=info['color'], alpha=0.2))
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation mIoU (%)', fontsize=12, fontweight='bold')
    ax.set_title('(d) Validation mIoU Curves (★ = Peak Performance)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=True, shadow=True, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1, 20)
    ax.set_ylim(60, 68)
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    
    return peak_info

def print_summary(histories, peak_info):
    """Print training summary statistics."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY STATISTICS")
    print("=" * 70)
    
    for fusion_type in ['concat', 'minimal', 'weighted']:
        history = histories[fusion_type]
        peak = peak_info[fusion_type]
        
        final_train_miou = history['train_miou'][-1] * 100
        train_val_gap = final_train_miou - peak['value']
        
        print(f"\n{fusion_type.upper()} FUSION:")
        print(f"  Peak Val mIoU:     {peak['value']:.2f}% (Epoch {peak['epoch']})")
        print(f"  Final Train mIoU:  {final_train_miou:.2f}%")
        print(f"  Train-Val Gap:     {train_val_gap:.2f}%")
        print(f"  Final Val Loss:    {history['val_loss'][-1]:.4f}")
    
    print("\n" + "=" * 70)

def main():
    """Main execution function."""
    checkpoint_base = Path("checkpoints")
    
    # Define checkpoint directories
    checkpoint_dirs = {
        'concat': checkpoint_base / "fusion_ablation_concat",
        'minimal': checkpoint_base / "fusion_ablation_minimal",
        'weighted': checkpoint_base / "fusion_ablation_weighted"
    }
    
    # Load training histories
    print("Loading training histories...")
    histories = {}
    for fusion_type, checkpoint_dir in checkpoint_dirs.items():
        try:
            histories[fusion_type] = load_training_history(checkpoint_dir)
            print(f"  ✓ Loaded {fusion_type} fusion history")
        except FileNotFoundError as e:
            print(f"  ✗ Error loading {fusion_type}: {e}")
            return
    
    # Generate plots
    print("\nGenerating training curves...")
    peak_info = plot_training_curves(histories)
    
    # Print summary
    print_summary(histories, peak_info)
    
    print("\nDone! Add the generated figure to your LaTeX report:")
    print("  \\includegraphics[width=0.95\\linewidth]{visualizations/training_curves.png}")

if __name__ == "__main__":
    main()