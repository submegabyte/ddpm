## https://chatgpt.com/share/67cb6b10-14cc-800c-a045-d68d6dbb4568

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
T = 1000  # Number of diffusion steps
beta = torch.linspace(1e-4, 0.02, T).to(device)  # Noise schedule
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# Simple forward diffusion process
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

class UNetDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 1024, 3, padding=1)
        self.enc4 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.enc5 = nn.Conv2d(2048, 4096, 3, padding=1)
        self.dec5 = nn.Conv2d(4096, 2048, 3, padding=1)
        self.dec4 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.dec3 = nn.Conv2d(1024, 128, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec2(x))
        x = self.dec1(x)
        return x



# Get the script's filename without extension
model_name = os.path.splitext(os.path.basename(__file__))[0]