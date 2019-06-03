import torch

def arrivalTimePoisson(lambd, t):
    return 1 - torch.exp(lambd*t)