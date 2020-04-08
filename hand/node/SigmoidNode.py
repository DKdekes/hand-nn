import numpy as np


class SigmoidNode:
    def __init__(self, n_inputs):
        self.w = np.random.rand(n_inputs)
        self.z = None
        self.a = None
        self.da = None
        self.delta = 0
        self.bias = np.random.rand()

    def compute(self, x):
        if isinstance(x, int) or x.shape == ():
            x = np.array([x])
        assert x.shape == self.w.shape
        dot = np.dot(self.w, x)
        self.z = dot + self.bias
        self.a = self.sigmoid(self.z)
        self.da = self.d_sigmoid(self.z)
        return self.a

    def sigmoid(self, x, c=1):
        return 1 / (1 + np.exp(-c * x))

    def d_sigmoid(self, x, c=1):
        return self.sigmoid(x) * (1 - self.sigmoid(x, c))


if __name__ == '__main__':
    pass
