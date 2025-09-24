# test_training_capability.py
import torch
import torch.nn as nn
from src.models.camera_encoder import TwinLiteEncoder
from src.models.lidar_encoder import LiDAREncoder
from src.models.fusion_module import CompleteSegmentationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
camera_encoder = TwinLiteEncoder().to(device)
lidar_encoder = LiDAREncoder(encoder_type="spatial", grid_size=(32, 32)).to(device)
model = CompleteSegmentationModel(camera_encoder, lidar_encoder).to(device)

model.train()  # Training mode

# Create dummy data and labels
dummy_images = torch.randn(2, 3, 256, 256).to(device)
dummy_points = torch.randn(2, 5000, 4).to(device)
dummy_labels = torch.randint(0, 3, (2, 256, 256)).to(device)

# Forward pass
try:
    outputs = model(dummy_images, dummy_points)
    
    # Loss computation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, dummy_labels)
    
    # Backward pass - this is where it might fail
    loss.backward()
    
    print(f"Success: Loss = {loss.item():.4f}")
    print("Model can train!")
    
except RuntimeError as e:
    print(f"Training failed: {e}")
    print("Model needs gradient flow fixes")