import numpy as np


class ReluNode:
    def __init__(self, n_inputs):
        self.w = np.random.rand(n_inputs)
        self.z = None
        self.a = None
        self.da = None
        self.delta = 0
        self.bias = np.random.rand()

    def compute(self, x):
        if isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, int) or x.shape == ():
            x = np.array([x])
        assert x.shape == self.w.shape
        dot = np.dot(self.w, x)
        self.z = dot + self.bias
        self.a = self.relu(self.z)
        self.da = self.d_relu(self.z)
        return self.a

    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def d_relu(self, x):
        if x > 0:
            return 1
        else:
            return 0.001


if __name__ == '__main__':
    pass
