import numpy as np
from hand.node.SigmoidNode import SigmoidNode


class DenseLayer:
    def __init__(self, n_nodes, n_prev_nodes=None):
        if n_prev_nodes is None:
            n_prev_nodes = n_nodes
        self.n_nodes = n_nodes
        self.nodes = [SigmoidNode(n_prev_nodes) for x in range(n_nodes)]

    def compute(self, x):
        if isinstance(x, int) or x.shape == ():
            x = np.array([x])
        if isinstance(x, list):
            x = np.array(x)
        output = np.zeros(self.n_nodes)
        for i, node in enumerate(self.nodes):
            output[i] = node.compute(x)
        return output

    def backward_propagate(self):
        pass
