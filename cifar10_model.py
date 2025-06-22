import torch
import torch.nn as nn
from bit_linear_layer import BitLinear

class SimpleBitNetCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Standard Conv layers first for feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # 32x32 -> 16x16. Channels: 32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # 16x16 -> 8x8. Channels: 64
        
        self.flatten = nn.Flatten()
        
        # Calculate the flattened size after conv and pool layers
        # CIFAR-10 images are 3x32x32
        # After conv1, relu1, pool1: (batch, 32, 16, 16)
        # After conv2, relu2, pool2: (batch, 64, 8, 8)
        self.fc_in_features = 64 * 8 * 8
        
        # Use BitLinear layers
        self.bitlinear1 = BitLinear(self.fc_in_features, 512)
        self.relu3 = nn.ReLU() # Standard ReLU after BitLinear output (which is float)
        self.bitlinear2 = BitLinear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.bitlinear1(x))
        x = self.bitlinear2(x)
        return x
