import pandas as pd
from hand import Network

if __name__ == '__main__':
    n_inputs_ = 4
    hidden_layers_ = [3]
    n_outputs_ = 3
    network = Network(n_inputs_, hidden_layers_, n_outputs_)
    df = pd.read_csv('../data/iris.data')
    df = df.sample(frac=1).reset_index(drop=True)
    x_train = df.loc[:, 'f1':'f4'].values
    y_train = df.loc[:, 'y']
    encoder = {
        'Iris-setosa': [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1],
    }
    y_train = y_train.map(encoder).values
    network.train(x_train, y_train, epochs=100)
    predictions = []
    for x, y in zip(x_train, y_train):
        print('expecting: {}'.format(y))
        print('got: {}'.format(network.forward_propagate(x)))