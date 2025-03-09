import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

## model
from simple_diffusion import *

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training setup
model = SimpleDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Load dataset (MNIST for simplicity)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop
epochs = 100
for epoch in range(epochs):
    for i, (x0, _) in enumerate(dataloader):
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        xt, noise = forward_diffusion(x0, t)
        noise_pred = model(xt, t)
        loss = loss_fn(noise_pred, noise)
        print(f"x0: {x0.shape}, t: {t.shape}, xt: {xt.shape}, noise: {noise.shape}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Datapoint {i}, Loss: {loss.item():.4f}")

# Ensure the 'models' folder exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save the model with the same name as the script
model_path = os.path.join(models_dir, f"{model_name}.pth")
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
