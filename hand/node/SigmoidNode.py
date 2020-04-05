import numpy as np
from hand.activate.sigmoid import s, ds


class SigmoidNode:
    def __init__(self, n_inputs):
        self.w = np.random.random(n_inputs)
        self.a = None
        self.da = None
        self.delta = 0
        self.bias = np.random.rand()

    def compute(self, x):
        if isinstance(x, int) or x.shape == ():
            x = np.array([x])
        assert x.shape == self.w.shape
        z = np.dot(self.w, x) + self.bias
        self.a = s(z)
        self.da = ds(z)
        return self.a


if __name__ == '__main__':
    pass
