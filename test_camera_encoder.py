import torch
from src.models.camera_encoder import TwinLiteEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TwinLiteEncoder().to(device)
model.eval()

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
