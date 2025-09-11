import torch

# Import TwinLiteEncoder from your src/models folder
from src.models.camera_encoder import TwinLiteEncoder

# 1. Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Initialize model and move to device
model = TwinLiteEncoder().to(device)

# 3. Create a dummy input tensor (batch size=1, 3 channels, 256x256 image)
dummy_input = torch.randn(1, 3, 256, 256).to(device)

# 4. Forward pass through the encoder
model.eval()
with torch.no_grad():
    output = model(dummy_input)

# 5. Print output shape
print(f"Output shape: {output.shape}")  # Expect: [1, 128, 32, 32]
