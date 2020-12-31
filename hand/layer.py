from hand.module import Module
from hand import initialize
import math
import logging
import torch


class Linear(Module):
    def __init__(self, units, n_in=None, n_out=None, weight_init='he', bias_init='zeros', lr=0.01):
        """

        :param w: weight tensor
        :param b: bias tensor
        :param lr: learning rate. want to experiment with multiple learning rates in future, so the learning
        rate is defined at the layer level for now
        """
        self.n_in = n_in
        self.n_out = n_out
        self.w = None
        self.b = None
        self.units = units
        self.lr = lr
        self.w_init = getattr(initialize, weight_init)()
        self.b_init = getattr(initialize, bias_init)()
        self.var_scaler = None
        self.logger = logging.getLogger(__name__)

    def setup(self, n_inputs):
        assert self.w is None
        assert self.b is None
        assert self.var_scaler is None
        self.w = self.w_init.initialize(n_inputs, self.units)
        self.b = self.b_init.initialize(1, self.units)
        self.var_scaler = self.w_init.var_scaler

    def forward(self, x):
        self.logger.debug('forward')
        assert math.isclose(self.w.var(), self.var_scaler, abs_tol=0.2)
        assert math.isclose(self.w.mean(), 0, abs_tol=0.2)
        y = x @ self.w + self.b
        # assert math.isclose(y.var(), 1), y.var()
        # assert math.isclose(y.var(), 1, abs_tol=0.2)
        self.logger.debug(y.var())
        assert not torch.any(y.isnan())
        return y

    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g
        self.b.g = out.g.sum()
        self.update()

    def update(self):
        """
        move in the opposite direction of the gradient to minimize loss (movement scaled by learning rate)
        :return:
        """
        self.w -= self.w.g * self.lr

        self.b -= self.b.g * self.lr

    def __str__(self):
        return f'linear: {self.units} units'
