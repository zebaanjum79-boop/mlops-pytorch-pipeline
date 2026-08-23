import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Block 1: 3 input channels (RGB) -> 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3: 64 -> 128 feature maps
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)   # halves height & width
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # After 3 pools: 32x32 -> 16x16 -> 8x8 -> 4x4, with 128 channels
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # -> [batch, 32, 16, 16]
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # -> [batch, 64, 8, 8]
        x = self.pool(self.relu(self.bn3(self.conv3(x))))   # -> [batch, 128, 4, 4]

        x = x.flatten(start_dim=1)                          # -> [batch, 128*4*4]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)                                      # -> [batch, num_classes] (raw scores)
        return x


def get_model(architecture: str = "simple_cnn", num_classes: int = 10):
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")