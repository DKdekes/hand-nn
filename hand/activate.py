import numpy as np
from hand.module import Module


class Relu(Module):
    """ Relu activation function


    """
    def forward(self, inp):
        print(f'{__name__} forward')
        return inp.clamp_min(0.) - 0.5

    def bwd(self, output, inp): inp.g = (inp > 0).float() * output.g
