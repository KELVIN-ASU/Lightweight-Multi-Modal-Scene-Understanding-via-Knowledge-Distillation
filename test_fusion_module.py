import torch
import torch.nn.functional as F

from src.models.camera_encoder import TwinLiteEncoder
from src.models.lidar_encoder import LiDAREncoder
from src.models.fusion_module import CompleteSegmentationModel


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Camera multiscale ON to exercise FPN-lite path
    cam_enc = TwinLiteEncoder(return_multiscale=True).to(device).eval()
    # LiDAR BEV at 64x64; vectorized path for speed
    lidar_enc = LiDAREncoder(encoder_type="spatial", grid_size=(64, 64), use_vectorized=True).to(device).eval()

    model = CompleteSegmentationModel(
        camera_encoder=cam_enc,
        lidar_encoder=lidar_enc,
        num_classes=3,
        fusion_type="concat",
        fusion_out_channels=256,
        camera_fpn_stages=["stage3", "stage4", "stage5"],
        camera_fpn_channels=128
    ).to(device)

    print("Arch summary:", model.get_architecture_summary())

    B, H, W = 2, 256, 256
    images = torch.randn(B, 3, H, W, device=device)
    points = torch.randn(B, 5000, 4, device=device)

    # Forward w/ intermediates
    with torch.no_grad():
        logits, mids = model(images, points, return_intermediates=True)

    print("Output logits:", logits.shape)          # should be [B, 3, 256, 256]
    print("Camera feat:", mids["camera_feat"].shape)   # typically [B, 128, 64, 64]
    print("LiDAR feat:", mids["lidar_feat"].shape)     # resized to camera grid
    print("Pre-fusion:", mids["pre_fusion"].shape)
    print("Post-fusion:", mids["post_fusion"].shape)

    # Shape checks
    assert logits.shape == (B, 3, H, W), "Segmentation logits should match image resolution"
    assert mids["camera_feat"].ndim == 4 and mids["lidar_feat"].ndim == 4

    # Quick backward sanity
    model.train()
    images.requires_grad_(True)
    logits = model(images, points, return_intermediates=False)
    labels = torch.randint(low=0, high=3, size=(B, H, W), device=device)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    print(f"Training sanity: loss={loss.item():.4f}")

    dec_grads = sum((p.grad is not None and p.grad.abs().sum().item() > 0) for p in model.head.parameters())
    fus_grads = sum((p.grad is not None and p.grad.abs().sum().item() > 0) for p in model.fusion.parameters())
    print(f"Decoder params with grad>0: {dec_grads}, Fusion params with grad>0: {fus_grads}")
    assert dec_grads > 0 and fus_grads > 0, "Expected gradients through fusion and decoder"

    print("Fusion module + full model forward/backward sanity checks passed!")


if __name__ == "__main__":
    main()
