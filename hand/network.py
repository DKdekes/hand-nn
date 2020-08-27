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
        x = np.array([x]).reshape(1, -1)
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
            if i == len(self.layers) - 1:
                # output layer processing
                error = expected - layer.a
            else:
                error = np.multiply(self.layers[i+1].w, self.layers[i+1].delta)
            layer.delta = np.multiply(error, layer.da.reshape(-1, 1))

    def update_weights(self, x):
        if isinstance(x, int) or x.shape == ():
            x = np.array([x])
        for i in range(len(self.layers)):
            if i == 0:
                # input layer processing
                inputs = x
            else:
                inputs = self.layers[i - 1].a
            # matrix operation candidate
            layer = self.layers[i]
            layer.w += self.learning_rate * np.multiply(layer.delta, inputs).T
            layer.bias += self.learning_rate * layer.delta.T

    def train(self, x, y, epochs=10):
        # does not train in batches currently. batch size = 1 training example
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
