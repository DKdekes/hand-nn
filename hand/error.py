import numpy as np
from hand.module import Module


class Mse(Module):
    def forward(self, inp, target):
        """

        :param inp: the input to the loss function. commonly the output, or prediction, of a network
        :param target: the target for the prediction
        :return: the mean squared error of the prediction
        """
        return (inp.squeeze() - target).pow(2).mean()

    def bwd(self, out, inp, target):
        """
        calculates the gradient for the input that will minimize the loss function
        :param out:
        :param inp:
        :param target:
        :return:
        """
        inp.g = 2 * (inp.squeeze() - target).unsqueeze(-1) / target.shape[0]
