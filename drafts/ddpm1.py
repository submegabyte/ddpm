## https://arxiv.org/pdf/2006.11239

## imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pixelcnnpp3 import PixelCNNpp

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