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

## Algorithm 2
# Sampling (reverse diffusion)
@torch.no_grad()
def sample(model):
    x = torch.randn((1, 1, 28, 28), device=device)
    for t in reversed(range(model.T)):
        z = torch.randn_like(x) if t > 0 else 0
        e_theta = model(x, torch.tensor([t], device=device))
        # x = (x - beta[t] * e_theta) / torch.sqrt(alpha[t]) + torch.sqrt(beta[t]) * z
        x = 1 / model.alpha[t]**0.5 * (x - (1 - model.alpha[t]) / (1 - model.alpha_bar[t])**0.5 * e_theta)\
            + model.sigma[t] * z
        
        ## normalize
        # x = (x - x.min()) / (x.max() - x.min() + 1e-5)
    return x

def sample_and_save(model, results_dir='results', n = 0):
    # Save the generated image
    sampled_image = sample(model).cpu().squeeze().numpy()
    plt.imshow(sampled_image, cmap="gray")
    plt.axis("off")  # Remove axes for a cleaner image
    # plt.savefig("generated_image.png", bbox_inches="tight", pad_inches=0)  # Save image
    # plt.show()

    # Ensure the 'results' folder exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Save the generated image in the 'results' folder
    save_path = os.path.join(results_dir, f"generated_image_{n}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    print(f"Image saved to {save_path}")

if __name__ == "__main__":
    model = load_model("ddpm1")
    sample_and_save(model, results_dir='results', n=0)