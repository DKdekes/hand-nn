import numpy as np
from hand.node.node import SigmoidalUnit


class Layer:
    def __init__(self, n_nodes, n_prev_nodes=None):
        if n_prev_nodes is None:
            n_prev_nodes = n_nodes
        self.n_nodes = n_nodes
        self.nodes = [SigmoidalUnit(n_prev_nodes) for x in range(n_nodes)]

    def compute(self, x):
        output = np.zeros(self.n_nodes)
        # double check enumerate()
        for i, node in enumerate(self.nodes):
            output[i] = node.compute(x)
        return output

