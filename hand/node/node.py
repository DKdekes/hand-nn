import numpy as np
from hand.activate.sigmoid import s, ds


class SigmoidalUnit:
    def __init__(self, n_inputs):
        self.w = np.random.random(n_inputs)
        self.val = None
        self.d_val = None
        self.bias = np.random.rand()

    def compute(self, x):
        assert x.shape == self.w.shape
        net_excitation = np.dot(self.w, x) + self.bias
        self.val = s(net_excitation)
        self.d_val = ds(net_excitation)
        return self.val


if __name__ == '__main__':
    z = np.random.random(2)
    comp_unit = SigmoidalUnit(2)
    comp_unit.compute(z)
