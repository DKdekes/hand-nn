import torch
import math


def he(inputs, units):
    return torch.randn(inputs, units) / math.sqrt(inputs)


def zeros(inputs, units):
    return torch.zeros(inputs, units)
