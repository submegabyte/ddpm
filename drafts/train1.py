import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

## model
from ddpm1 import *

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# Training setup
model = DDPM(input_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Load dataset (MNIST for simplicity)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64*2, shuffle=True)

## Algorithm 1
epochs = 100
for epoch in range(epochs):
    for i, (x0, _) in enumerate(dataloader):
        x0 = x0.to(device)
        t = torch.randint(0, model.T, (x0.shape[0],), device=device)
        e = torch.randn_like(x0)
        a_t = model.alpha_bar[t][:, None, None, None]

        # print(a_t.shape, x0.shape)
        xt = a_t**0.5 * x0 + (1 - a_t)**0.5 * e
        e_theta = model(xt)

        # print(e.shape, e_theta.shape)
        loss = loss_fn(e, e_theta)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

# Ensure the 'models' folder exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save the model with the same name as the script
model_path = os.path.join(models_dir, f"{model.name}.pth")
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
