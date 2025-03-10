import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from datetime import datetime

from ddpm1 import *
from utils1 import *
from sample1 import *

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# Training setup
model = DDPM(input_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
loss_fn = nn.MSELoss()

# Load dataset (MNIST for simplicity)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64*2, shuffle=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
models_dir = "models/" + timestamp
results_dir = "results/" + timestamp

## Algorithm 1
epochs = 5
for epoch in range(epochs):
    epoch_loss = 0
    for i, (x0, _) in enumerate(train_loader):
        x0 = x0.to(device)
        t = torch.randint(0, model.T, (x0.shape[0],), device=device)
        e = torch.randn_like(x0)
        a_t = model.alpha_bar[t][:, None, None, None]

        # print(a_t.shape, x0.shape)
        xt = a_t**0.5 * x0 + (1 - a_t)**0.5 * e
        e_theta = model(xt)

        # print(e.shape, e_theta.shape)
        loss = loss_fn(e, e_theta)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
    
    # Update learning rate
    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Loss: {epoch_loss / len(train_loader):.4f}, "
          f"LR: {scheduler.get_last_lr()[0]:.6f}, "
        #   f"Momentum: {current_momentum:.3f}"
        )
    
    ## sample every sample_period steps
    sample_period = 1
    if sample_during_training and epoch % sample_period == 0:
        sample_and_save(model, results_dir=results_dir, n=int(epoch/sample_period))

save_model(model, models_dir)