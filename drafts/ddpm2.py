## https://arxiv.org/pdf/2006.11239

## imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from pixelcnnpp3 import PixelCNNpp
from utils1 import *

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

## reverse diffusion
## denoising
## p_theta(x_t-1 | xt)
def p(x, t, mean, std):
    u = mean(x, t)
    s = std(x, t)
    p = torch.normal(u, s)
    return p

## forward diffusion
## q(xt | x_t-1)
def q(x, t, beta):
    I = torch.ones_like(x)
    beta = beta[t]
    u = (1 - beta)**0.5 * x
    s = beta * I
    q = torch.normal(u, s)
    return q

## q(xt | x0)
def qt(x0, t, beta):
    I = torch.ones_like(x)
    alpha = 1 - beta
    alpha_prod = torch.prod(alpha[:t])
    u = alpha_prod**0.5
    s = 1 - alpha_prod
    q = torch.normal(u, s)


class DDPM(nn.Module):

    def __init__(self, input_channels=1):
        super(DDPM, self).__init__()
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.T = 300
        self.beta = torch.linspace(1e-4, 0.02, self.T).to(device)  # Noise schedule
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma = torch.sqrt(self.beta)
        # self.device = device

        self.pixelcnnpp = PixelCNNpp(input_channels=input_channels)

    ## u_theta(x_t, t)
    ## predicts the mean
    ## of posterior probability q(x_t-1 | xt, x0)
    ## for the forward diffusion
    ## with likelihood q(xt | x_t-1)
    def u(self, e, x, t):
        u = self.beta[t] / (1 - self.alpha_bar[t])**0.5 * e(x, t)
        u = x - u
        u = 1 / self.alpha[t]**0.5 * u
        return u
    
    ## Algorithm 1
    ## x is xt, not x0
    def forward(self, x, t=None):
        xt = x
        # e = torch.randn_like(x)
        # xt = self.alpha_bar[t]**0.5 * x0 + (1 - self.alpha_bar[t])**0.5 * e

        ## estimated noise
        e_theta = self.pixelcnnpp(xt)
        return e_theta

    ## x0 = image, xT = noise
    ## x_t -> x_t-1
    def next(self, x, t):
        model = self
        e_theta = model(x, torch.tensor([t], device=device))
        z = torch.randn_like(x) if t > 0 else 0
        x = 1 / model.alpha[t]**0.5 * (x - (1 - model.alpha[t]) / (1 - model.alpha_bar[t])**0.5 * e_theta)\
                + model.sigma[t] * z
        return x
    
    ## Algorithm 2
    # Sampling (reverse diffusion)
    def sample(self):
        x = torch.randn((1, 1, 28, 28), device=device)
        for t in reversed(range(self.T)):
            x = self.next(x, t)

            ## normalize
            # x = (x - x.min()) / (x.max() - x.min() + 1e-5)
        return x

    def sample_and_save(self, results_dir='results', n = 0):
        # Save the generated image
        sampled_image = self.sample().cpu().squeeze().detach().numpy()
        plt.imshow(sampled_image, cmap="gray")
        plt.axis("off")  # Remove axes for a cleaner image
        # plt.savefig("generated_image.png", bbox_inches="tight", pad_inches=0)  # Save image
        # plt.show()

        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # results_dir += '/' + timestamp

        # Ensure the 'results' folder exists
        # results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Save the generated image in the 'results' folder
        save_path = os.path.join(results_dir, f"{n}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Image saved to {save_path}")
    
    def train_MNIST(self, sample_during_training=True):
        # Training setup
        model = self
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        loss_fn = nn.MSELoss()

        # Load dataset (MNIST for simplicity)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64*2, shuffle=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        models_dir = "models/" + timestamp
        results_dir = "results/train/" + timestamp

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
                self.sample_and_save(results_dir=results_dir, n=int(epoch/sample_period))

    def load_model(self, models_dir="models"):
        model = self

        # Ensure the 'models' folder exists
        # models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{self.name}.pth")

        # Load the model from the saved file
        model = DDPM().to(device)
        if os.path.exists(model_path):  # Check if the file exists before loading
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model.eval()  # Set to evaluation mode
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        
        return model

    def save_model(self, models_dir="models"):
        model = self
        # Ensure the 'models' folder exists
        # models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # Save the model with the same name as the script
        model_path = os.path.join(models_dir, f"{model.name}.pth")
        torch.save(model.state_dict(), model_path)

        print(f"Model saved to {model_path}")