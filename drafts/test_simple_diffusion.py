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

# Ensure the 'models' folder exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, f"{model_name}.pth")

# Load the model from the saved file
model = SimpleDenoiser().to(device)
if os.path.exists(model_path):  # Check if the file exists before loading
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found!")


# Sampling (reverse diffusion)
@torch.no_grad()
def sample():
    x = torch.randn((1, 1, 28, 28), device=device)
    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0
        noise_pred = model(x, torch.tensor([t], device=device))
        x = (x - beta[t] * noise_pred) / torch.sqrt(alpha[t]) + torch.sqrt(beta[t]) * z
    return x

# Save the generated image
sampled_image = sample().cpu().squeeze().numpy()
plt.imshow(sampled_image, cmap="gray")
plt.axis("off")  # Remove axes for a cleaner image
# plt.savefig("generated_image.png", bbox_inches="tight", pad_inches=0)  # Save image
# plt.show()

# Ensure the 'results' folder exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Save the generated image in the 'results' folder
save_path = os.path.join(results_dir, "generated_image.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
print(f"Image saved to {save_path}")