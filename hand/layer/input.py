import numpy as np
from hand.node.node import SigmoidalUnit


class Input:
    def __init__(self, n_inputs):
        self.n_nodes = n_inputs
        self.nodes = [SigmoidalUnit(n_inputs) for x in range(n_inputs)]

    def compute(self, x):
        output = np.zeros(self.n_nodes)
        for i, node in enumerate(self.nodes):
            output[i] = node.compute(x)
        return output



