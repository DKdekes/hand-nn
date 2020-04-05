import pandas as pd
from hand.layer.DenseLayer import DenseLayer
import numpy as np


class Network:
    def __init__(self, n_inputs, hidden_layers, n_outputs):
        self.model = []
        self.model.append(DenseLayer(n_inputs))
        self.learning_rate = 0.1
        prev_nodes = n_inputs
        for n_nodes_layer in hidden_layers:
            self.model.append(DenseLayer(n_nodes_layer, prev_nodes))
            prev_nodes = n_nodes_layer
        self.model.append(DenseLayer(n_outputs, prev_nodes))

    def __str__(self):
        str = ''
        for layer in self.model:
            str += ' {}'.format(layer.n_nodes)
        return str

    def forward_propagate(self, x):
        if x.shape == ():
            x = np.array([x])
        res = []
        for row in x:
            for layer in self.model:
                row = layer.compute(row)
            res.append(row)
        return res

    def backward_propagate(self, expected):
        if expected.shape == ():
            expected = np.array([expected])
        # bias won't be adjusted yet?
        # add bias to backprop step
        # node.delta may always be zero
        for i in reversed(range(len(self.model))):
            layer = self.model[i]
            errors = []
            if i != len(self.model) - 1:
                for j in range(len(layer.nodes)):
                    error = 0.0
                    for node in self.model[i+1].nodes:
                        error += (node.w[j] * node.delta)
                        error += (node.bias * node.delta)
                    errors.append(error)
            else:
                for j in range(len(layer.nodes)):
                    node = layer.nodes[j]
                    errors.append(expected[j] - node.a)
            for j, node in enumerate(layer.nodes):
                node.delta = errors[j] * node.da

    def update_weights(self, x):
        if x.shape == ():
            x = np.array([x])
        for i in range(len(self.model)):
            inputs = x
            if i != 0:
                inputs = [node.a for node in self.model[i-1].nodes]
            for node in self.model[i].nodes:
                for j in range(len(inputs)):
                    node.w[j] += self.learning_rate * node.delta * inputs[j]
                node.bias += self.learning_rate * node.delta

    def train(self, x, y, epochs=10):
        # does not train in batches, only epochs
        for i in range(epochs):
            for x_, y_ in zip(x, y):
                self.forward_propagate(x_)
                self.backward_propagate(y_)
                self.update_weights(x_)
        print(self.forward_propagate(x))

    def report_weights(self):
        for i, layer in enumerate(self.model):
            print('layer {}'.format(i))
            for node in layer.nodes:
                print('weights:')
                print(node.w)
                print('bias:')
                print(node.bias)

if __name__ == '__main__':
    n_inputs_ = 1
    hidden_layers_ = [1]
    n_outputs_ = 1
    network = Network(n_inputs_, hidden_layers_, n_outputs_)
    df = pd.read_csv('../../data/linear_data.csv')
    x_train = df.loc[:, 'X'].values
    y_train = df.loc[:, 'y'].values
    network.train(x_train, y_train)
    print(network.report_weights())
