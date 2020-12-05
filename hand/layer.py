from hand.module import Module
from hand import initialize


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
        self.w_init = getattr(initialize, weight_init)
        self.b_init = getattr(initialize, bias_init)

    def setup(self, inputs):
        assert self.w is None
        assert self.b is None
        self.w = self.w_init(inputs, self.units)
        self.b = self.b_init(1, self.units)

    def forward(self, inp):
        return inp @ self.w + self.b

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
