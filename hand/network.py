from hand.layer import DenseLayer
import numpy as np


class Network:
    def __init__(self, n_inputs, hidden_layers, n_outputs, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        prev_nodes = n_inputs
        for n_nodes_layer in hidden_layers:
            self.layers.append(DenseLayer(n_nodes_layer, prev_nodes))
            prev_nodes = n_nodes_layer
        self.layers.append(DenseLayer(n_outputs, prev_nodes))

    def __str__(self):
        str = ''
        for layer in self.layers:
            str += ' {}'.format(layer.n_nodes)
        return str

    def forward_propagate(self, x):
        if isinstance(x, int) or x.shape == ():
            x = np.array([x])
        res = []
        row = x
        for layer in self.layers:
            row = layer.compute(row)
        res.append(row)
        return res

    def backward_propagate(self, expected):
        if isinstance(expected, int) or expected.shape == ():
            expected = np.array([expected])
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = []
            if i == len(self.layers) - 1:
                # output layer processing
                # matrix operation candidate
                for j in range(len(layer.nodes)):
                    node = layer.nodes[j]
                    errors.append(expected[j] - node.a)
            else:
                for j in range(len(layer.nodes)):
                    error = 0.0
                    # matrix operation candidate
                    for node in self.layers[i + 1].nodes:
                        error += (node.w[j] * node.delta)
                    errors.append(error)
            # matrix operation candidate
            for j, node in enumerate(layer.nodes):
                node.delta = errors[j] * node.da

    def update_weights(self, x):
        if isinstance(x, int) or x.shape == ():
            x = np.array([x])
        for i in range(len(self.layers)):
            if i == 0:
                # input layer processing
                inputs = x
            else:
                inputs = [node.a for node in self.layers[i - 1].nodes]
            # matrix operation candidate
            for node in self.layers[i].nodes:
                for j in range(len(inputs)):
                    weight_change = self.learning_rate * node.delta * inputs[j]
                    node.w[j] += weight_change
                node.bias += self.learning_rate * node.delta

    def train(self, x, y, epochs=10):
        # does not train in batches currently. each epoch is a batch
        for i in range(epochs):
            for x_, y_ in zip(x, y):
                self.forward_propagate(x_)
                self.backward_propagate(y_)
                self.update_weights(x_)

    def report_weights(self):
        for i, layer in enumerate(self.layers):
            print('layer {}'.format(i))
            for node in layer.nodes:
                print('weights:')
                print(node.w)
                print('bias:')
                print(node.bias)
