import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from datetime import datetime

from ddpm2 import *
from utils1 import *

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_and_save(model, results_dir='results', n = 0):
    # Save the generated image
    sampled_image = sample(model).cpu().squeeze().numpy()
    plt.imshow(sampled_image, cmap="gray")
    plt.axis("off")  # Remove axes for a cleaner image
    # plt.savefig("generated_image.png", bbox_inches="tight", pad_inches=0)  # Save image
    # plt.show()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir += '/' + timestamp

    # Ensure the 'results' folder exists
    # results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Save the generated image in the 'results' folder
    save_path = os.path.join(results_dir, f"generated_image_{n}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    print(f"Image saved to {save_path}")

if __name__ == "__main__":
    model = load_model("ddpm1", "models/2025-03-10_14-20-11")
    sample_and_save(model, results_dir='results', n=0)