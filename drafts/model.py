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

## reverse diffusion
## denoising
## p_theta(x_t-1 | xt)
def p_theta(x, t, mean, std):
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
def qt0(x0, t, beta):
    I = torch.ones_like(x)
    alpha = 1 - beta
    alpha_prod = torch.prod(alpha[:t])
    u = alpha_prod**0.5
    s = 1 - alpha_prod
    q = torch.normal(u, s)

