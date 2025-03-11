import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pixelcnnpp3 import PixelCNNpp
from utils1 import *
from ddpm2 import *

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# model1 = DDPM(input_channels=1).to(device)
# model1.train_MNIST()
# [model1.sample_and_save(results_dir="results/before_model_save", n=i) for i in range(5)]
# save_model(model1, models_dir="models/after_model_save/")

model2 = DDPM().to(device)
model2.load_model(models_dir="models/after_model_save/")
[model2.sample_and_save(results_dir="results/after_model_save", n=i) for i in range(5)]