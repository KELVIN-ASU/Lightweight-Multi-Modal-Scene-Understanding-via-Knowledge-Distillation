
import torch
from src.models.camera_encoder import TwinLiteEncoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = TwinLiteEncoder().to(device)
    model.eval()

    # Print parameter count to monitor 
    print(f"Camera encoder parameters: {model.count_parameters():,}")
    print(f"Output channels: {model.out_channels}")
    print("-" * 50)

    # -------------------------------------------------
    # 1. Test with a 256x256 RGB image
    # -------------------------------------------------
    dummy_input_1 = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        output_1 = model(dummy_input_1)
    print(f"Input (1, 3, 256, 256) -> Output {output_1.shape}")

    # -------------------------------------------------
    # 2. Test with a 512x512 RGB image
    # -------------------------------------------------
    dummy_input_2 = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        output_2 = model(dummy_input_2)
    print(f"Input (1, 3, 512, 512) -> Output {output_2.shape}")

    # -------------------------------------------------
    # 3. Test with a batch of 4 images (128x128)
    # -------------------------------------------------
    dummy_input_3 = torch.randn(4, 3, 128, 128).to(device)
    with torch.no_grad():
        output_3 = model(dummy_input_3)
    print(f"Input (4, 3, 128, 128) -> Output {output_3.shape}")

    print("-" * 50)

    # -------------------------------------------------
    # 4. Test multi-scale output functionality
    # -------------------------------------------------
    print("Testing multi-scale output:")
    model_multiscale = TwinLiteEncoder(return_multiscale=True).to(device)
    model_multiscale.eval()

    with torch.no_grad():
        multiscale_output = model_multiscale(dummy_input_1)
    
    print(f"Multi-scale features for input {dummy_input_1.shape}:")
    for stage, feat in multiscale_output.items():
        print(f"  {stage}: {feat.shape}")

    # -------------------------------------------------
    # 5. Test feature info method
    # -------------------------------------------------
    print("-" * 50)
    print("Feature channel information:")
    feature_info = model.get_feature_info()
    for stage, channels in feature_info.items():
        print(f"  {stage}: {channels} channels")

    # -------------------------------------------------
    # 6. Memory usage test (optional)
    # -------------------------------------------------
    if torch.cuda.is_available():
        print("-" * 50)
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    print("-" * 50)
    print("Camera encoder tests completed successfully!")

if __name__ == "__main__":
    main()