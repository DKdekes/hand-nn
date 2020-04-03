import numpy as np
from hand.node.SigmoidalUnit import SigmoidalUnit

class OutputLayer:
    def __init__(self, n_outputs, n_prev_nodes=None):
        if n_prev_nodes is None:
            n_prev_nodes = n_outputs
        self.n_outputs = n_outputs
        self.outputs = [SigmoidalUnit(n_prev_nodes) for x in range(n_outputs)]
        self.d = np.zeros(n_outputs)

    def compute(self, x):
        output = np.zeros(self.n_outputs)
        for i, output_node in enumerate(self.outputs):
            output[i] = output_node.compute(x)
        return output

    def backpropogate(self, err):
        for i, output_node in self.outputs:
            self.d[i] =