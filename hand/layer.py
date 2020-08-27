import numpy as np
from hand.activate import relu, d_relu


class DenseLayer:
    def __init__(self, n_nodes, n_prev_nodes=1):
        self.n_nodes = n_nodes
        self.w = np.random.rand(n_prev_nodes, n_nodes)
        self.bias = np.random.rand(1, n_nodes)
        self.a = None
        self.da = None
        self.delta = None

    def compute(self, x: np.array):
        z = np.matmul(x, self.w) + self.bias
        self.a = relu(z)
        self.da = d_relu(z)
        return self.a
