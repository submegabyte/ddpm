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

def load_model(model_name):
    # Ensure the 'models' folder exists
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.pth")

    # Load the model from the saved file
    model = DDPM().to(device)
    if os.path.exists(model_path):  # Check if the file exists before loading
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()  # Set to evaluation mode
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    
    return model

def save_model(model, models_dir="models"):
    # Ensure the 'models' folder exists
    # models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Save the model with the same name as the script
    model_path = os.path.join(models_dir, f"{model.name}.pth")
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")