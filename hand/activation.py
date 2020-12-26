import numpy as np
from hand.module import Module


class Relu(Module):
    """ Relu activation function


    """
    def forward(self, inp):
        return inp.clamp_min(0.) - 0.5

    def bwd(self, output, inp): inp.g = (inp > 0).float() * output.g

    def __str__(self):
        return 'relu'
