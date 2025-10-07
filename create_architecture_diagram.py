import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create a detailed architecture diagram of the multi-modal fusion system."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    color_input = '#E8F4F8'
    color_encoder = '#B8E6F0'
    color_fusion = '#FFE5B4'
    color_decoder = '#FFB6C1'
    color_output = '#D4F1D4'
    
    # Helper function for boxes
    def draw_box(x, y, w, h, text, color, fontsize=10, fontweight='normal'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=fontsize, fontweight=fontweight, wrap=True)
    
    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2, label=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Title
    ax.text(8, 9.5, 'Lightweight Multi-Modal BEV Segmentation Architecture', 
           ha='center', fontsize=16, fontweight='bold')
    
    # ========== INPUT LAYER ==========
    y_input = 8
    draw_box(1, y_input, 2, 0.8, 'RGB Camera\n256×256×3', color_input, fontsize=9)
    draw_box(4, y_input, 2, 0.8, 'LiDAR Points\n5000×4 (x,y,z,i)', color_input, fontsize=9)
    
    # ========== ENCODER LAYER ==========
    y_encoder = 6
    
    # Camera Encoder
    draw_box(0.5, y_encoder, 3, 1.2, 'TwinLiteEncoder\n(MobileNetV2-style)\n363K params', 
            color_encoder, fontsize=9, fontweight='bold')
    ax.text(2, y_encoder - 0.3, 'base_channels=32\nInvertedResidual blocks', 
           ha='center', fontsize=7, style='italic')
    
    # Multi-scale features
    draw_box(0.3, y_encoder - 1.2, 1.2, 0.6, 'Stage2\n64×64×64', color_encoder, fontsize=7)
    draw_box(1.6, y_encoder - 1.2, 1.2, 0.6, 'Stage3\n64×64×64', color_encoder, fontsize=7)
    draw_box(2.9, y_encoder - 1.2, 1.2, 0.6, 'Stage4\n32×32×128', color_encoder, fontsize=7)
    
    # Camera FPN
    draw_box(0.8, y_encoder - 2.2, 2.7, 0.7, 'Camera FPN-Lite\n→ 64×64×128', 
            color_encoder, fontsize=8)
    
    # LiDAR Encoder
    draw_box(4.5, y_encoder, 3, 1.2, 'SpatialLiDAREncoder\n(PointNet-style)\n26K params', 
            color_encoder, fontsize=9, fontweight='bold')
    ax.text(6, y_encoder - 0.3, 'feature_dim=128\nVectorized BEV rasterization', 
           ha='center', fontsize=7, style='italic')
    
    # LiDAR stages
    draw_box(4.3, y_encoder - 1.2, 1.5, 0.6, 'Point MLP\n4→64→128', color_encoder, fontsize=7)
    draw_box(6, y_encoder - 1.2, 1.5, 0.6, 'BEV Grid\n64×64×128', color_encoder, fontsize=7)
    
    # Arrows from input to encoders
    draw_arrow(2, y_input, 2, y_encoder + 1.2)
    draw_arrow(5, y_input, 6, y_encoder + 1.2)
    
    # Arrows within encoders
    draw_arrow(2, y_encoder, 1.5, y_encoder - 1.2, '')
    draw_arrow(2, y_encoder, 2.2, y_encoder - 1.2, '')
    draw_arrow(2, y_encoder, 3.5, y_encoder - 1.2, '')
    
    draw_arrow(1.5, y_encoder - 1.8, 2.1, y_encoder - 2.2, '')
    draw_arrow(2.2, y_encoder - 1.8, 2.1, y_encoder - 2.2, '')
    draw_arrow(3.5, y_encoder - 1.8, 2.3, y_encoder - 2.2, '')
    
    draw_arrow(6, y_encoder, 5, y_encoder - 1.2, '')
    draw_arrow(6, y_encoder, 6.7, y_encoder - 1.2, '')
    
    # ========== FUSION LAYER ==========
    y_fusion = 2.5
    
    # Concat fusion (winner)
    draw_box(1, y_fusion, 6, 1, 'Concatenation Fusion (Best: 67.4% mIoU)\n' + 
            'Camera [128] + LiDAR [128] → Concat [256]\n' +
            'Depthwise Conv → Pointwise Conv → 256 channels\n' +
            '162K params', 
            color_fusion, fontsize=9, fontweight='bold')
    
    # Alternative fusions (smaller boxes on sides)
    draw_box(8, y_fusion + 0.3, 3.5, 0.35, 'Minimal Fusion: 66.6% mIoU (93K params)', 
            '#FFE5B4', fontsize=7)
    draw_box(8, y_fusion - 0.1, 3.5, 0.35, 'Weighted Fusion: 66.4% mIoU (126K params)', 
            '#FFE5B4', fontsize=7)
    
    # Arrows to fusion
    draw_arrow(2.1, y_encoder - 2.9, 3, y_fusion + 1, '128 ch')
    draw_arrow(6.7, y_encoder - 1.8, 5, y_fusion + 1, '128 ch')
    
    # ========== DECODER LAYER ==========
    y_decoder = 1
    draw_box(2, y_decoder, 4, 0.8, 'Segmentation Head (Same Resolution)\n' +
            'DWSepConv(256→64) → DWSepConv(64→32) → Conv(32→2)\n22K params', 
            color_decoder, fontsize=8, fontweight='bold')
    
    # Arrow from fusion to decoder
    draw_arrow(4, y_fusion, 4, y_decoder + 0.8, '256 ch')
    
    # ========== OUTPUT LAYER ==========
    y_output = 0.1
    draw_box(2.5, y_output, 3, 0.6, 'BEV Segmentation\n64×64×2 classes\n[Background, Drivable]', 
            color_output, fontsize=9, fontweight='bold')
    
    # Arrow from decoder to output
    draw_arrow(4, y_decoder, 4, y_output + 0.6, '2 classes')
    
    # ========== LEGEND & STATS ==========
    # Model statistics box
    stats_text = (
        'Model Statistics:\n'
        '• Total: 573K params\n'
        '• Camera: 363K (63%)\n'
        '• LiDAR: 26K (5%)\n'
        '• Fusion: 162K (28%)\n'
        '• Head: 22K (4%)\n'
        '────────────────\n'
        'Performance:\n'
        '• Val mIoU: 67.4%\n'
        '• Background IoU: 90%\n'
        '• Drivable IoU: 45%'
    )
    
    ax.text(12.5, 6.5, stats_text, fontsize=8, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.5),
           verticalalignment='top')
    
    # Legend
    legend_y = 4.5
    ax.text(12.5, legend_y, 'Component Types:', fontsize=9, fontweight='bold')
    
    legend_items = [
        (color_input, 'Input Data'),
        (color_encoder, 'Encoders'),
        (color_fusion, 'Fusion Layer'),
        (color_decoder, 'Decoder Head'),
        (color_output, 'Output')
    ]
    
    for i, (color, label) in enumerate(legend_items):
        y_pos = legend_y - 0.4 - i * 0.4
        rect = mpatches.Rectangle((12, y_pos - 0.15), 0.3, 0.25, 
                                  facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(12.5, y_pos, label, fontsize=8, va='center')
    
    # Dataset info
    dataset_text = (
        'Dataset: PandaSet\n'
        '• Train: 1,920 samples (31 scenes)\n'
        '• Val: 480 samples (8 scenes)\n'
        '• 2 classes: background/drivable\n'
        '• BEV range: [-50m, 50m]'
    )
    ax.text(12.5, 2, dataset_text, fontsize=8, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.5),
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('architecture_diagram.pdf', bbox_inches='tight', facecolor='white')
    print("Architecture diagram saved as:")
    print("  - architecture_diagram.png (high-res)")
    print("  - architecture_diagram.pdf (vector format)")
    plt.show()


if __name__ == "__main__":
    create_architecture_diagram()