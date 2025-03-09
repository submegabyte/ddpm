import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

## model
from unet_2 import *

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure the 'models' folder exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, f"{model_name}.pth")

# Load the model from the saved file
model = UNetDenoiser().to(device)
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
        
        ## normalize
        # x = (x - x.min()) / (x.max() - x.min() + 1e-5)
    return x

## https://chatgpt.com/share/67cbf4db-716c-800c-97e4-57444b099efb
# DDIM Sampling Process
@torch.no_grad()
def ddim_sample():
    """DDIM sampling from pure noise"""

    shape = (1, 1, 28, 28)
    ddim_steps = 300

    device = next(model.parameters()).device
    x_t = torch.randn(shape, device=device)  # Start from pure noise
    step_size = T // ddim_steps
    ts = list(reversed(range(0, T, step_size)))

    for i in range(len(ts) - 1):
        t = torch.full((shape[0],), ts[i], device=device, dtype=torch.long)
        t_next = torch.full((shape[0],), ts[i+1], device=device, dtype=torch.long)

        alpha_t = alpha_hat[t][:, None, None, None]
        alpha_t_next = alpha_hat[t_next][:, None, None, None]
        sqrt_alpha_t = alpha_t ** 0.5
        sqrt_alpha_t_next = alpha_t_next ** 0.5
        sqrt_one_minus_alpha_t = (1 - alpha_t) ** 0.5

        noise_pred = model(x_t, t)  # Predict noise
        x0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t  # Reconstruct x0

        eta = 0 ## for ddim
        if eta == 0:
            x_t = sqrt_alpha_t_next * x0_pred + (1 - alpha_t_next) ** 0.5 * noise_pred
        else:
            noise = torch.randn_like(x_t)
            sigma_t = eta * ((1 - alpha_t_next) / (1 - alpha_t)) ** 0.5
            x_t = sqrt_alpha_t_next * x0_pred + sigma_t * noise + ((1 - alpha_t_next - sigma_t**2) ** 0.5) * noise_pred

    return x_t

# Save the generated image
sampled_image = ddim_sample().cpu().squeeze().numpy()
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