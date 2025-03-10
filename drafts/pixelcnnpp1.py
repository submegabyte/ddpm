## https://chatgpt.com/share/67ce8a06-bff4-800c-adfc-c22cca45809d

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.shape
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask  # Apply the mask
        return super().forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = MaskedConv2d(channels, channels, kernel_size=3, mask_type='B', padding=1)
        self.conv2 = MaskedConv2d(channels, channels, kernel_size=3, mask_type='B', padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class PixelCNNpp(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, num_layers=7):
        super().__init__()
        layers = [MaskedConv2d(in_channels, hidden_channels, kernel_size=7, mask_type='A', padding=3)]
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_channels))
        layers.append(nn.Conv2d(hidden_channels, 10 * in_channels, kernel_size=1))  # Mixture of logistics
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).permute(0, 2, 3, 1).contiguous().view(x.shape[0], x.shape[2], x.shape[3], 10, 3)

# Example usage
pixelcnnpp = PixelCNNpp()
x = torch.randn(1, 3, 32, 32)  # Example input
out = pixelcnnpp(x)
print(out.shape)  # (1, 32, 32, 10, 3) - Mixture of logistics
