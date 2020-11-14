import numpy as np
from hand.module import Module


class Mse(Module):
    def forward(self, inp, target):
        return (inp.squeeze() - target).pow(2).mean()

    def bwd(self, out, inp, target):
        inp.g = 2 * (inp.squeeze() - target).unsqueeze(-1) / target.shape[0]
