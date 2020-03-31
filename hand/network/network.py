import pandas as pd
from hand.layer import Layer
from hand.error import error


class Network:
    def __init__(self, n_inputs, hidden_layers, n_outputs):
        self.model = []
        self.model.append(Layer(n_inputs))
        prev_nodes = n_inputs
        for n_nodes_layer in hidden_layers:
            self.model.append(Layer(n_nodes_layer, prev_nodes))
            prev_nodes = n_nodes_layer
        self.model.append(Layer(n_outputs, prev_nodes))

    def __str__(self):
        str = ''
        for layer in self.model:
            str += ' {}'.format(layer.n_nodes)
        return str

    def predict(self, x_set):
        res = []
        for x in x_set:
            for layer in self.model:
                x = layer.compute(x)
            res.append(x)
        return res

    def train(self, x, y, epochs=10):
        # does not train in batches, only epochs
        for i in range(epochs):
            predictions = self.predict(x)
            err = error(y, predictions)
            print(err)


if __name__ == '__main__':
    # currently on page: 161 of K7
    #       calculating dE / dw
    n_inputs_ = 3
    hidden_layers_ = [3, 3, 3]
    n_outputs_ = 1
    network = Network(n_inputs_, hidden_layers_, n_outputs_)
    df = pd.read_csv('../data.csv')
    x_train = df.loc[:, 'bit_3':'bit_1'].values
    y_train = df.loc[:, 'target'].values
    network.train(x_train, y_train)
