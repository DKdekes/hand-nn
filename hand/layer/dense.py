import numpy as np
from hand.node.SigmoidalUnit import SigmoidalUnit
import random

class DenseLayer:
    def __init__(self, n_nodes, n_prev_nodes=None):
        self.n_nodes = n_nodes
        self.nodes = [SigmoidalUnit(n_prev_nodes) for x in range(n_nodes)]

    def compute(self, x):
        output = np.zeros(self.n_nodes)
        for i, node in enumerate(self.nodes):
            output[i] = node.compute(x)
        return output

    def backward_propagate(self):
        pass
