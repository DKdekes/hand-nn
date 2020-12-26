import numpy as np
from hand.module import Module
import sys


class Mse(Module):
    def forward(self, inp, target):
        """
        :param inp: the input to the loss function. commonly the output, or prediction, of a network
        :param target: the target for the prediction
        :return: the mean squared error of the prediction. remember, mse outputs a scalar, not very useful for
        calculating gradient
        """
        return (inp.squeeze() - target).pow(2).mean()

    def bwd(self, out, inp, target):
        """
        calculates the gradient for the input that will minimize the loss function, which is the
        negative derivative of mean squared error (2 * mean error). we use inp instead of out when calculating
        this gradient because out is the mse, which is just a scalar representing the mean of all errors squared.

        :param out: the output of this layer during forward prop
        :param inp: the output of the last layer of the network during forward prop
        :param target: the expected value of the output
        :return: nothing. sets the input gradient
        """
        target_diff = inp - target
        assert target_diff.shape[1] == 1, \
            'target diff cannot have multiple columns (there can only be one difference per example'
        inp.g = 2 * target_diff / target.shape[0]


def __getattr__(item):
    for a in globals():
        if a.lower() == item.lower():
            return globals()[a]
