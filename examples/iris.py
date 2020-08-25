import pandas as pd
from hand import Network

if __name__ == '__main__':
    n_inputs_ = 4
    hidden_layers_ = [3, 3]
    n_outputs_ = 1
    network = Network(n_inputs_, hidden_layers_, n_outputs_)
    df = pd.read_csv('../data/iris.data')
    x_train = df.loc[:, 0:3].values
    y_train = df.loc[:, 4]
    encoder = {
        'Iris-setosa': [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1],
    }
    y_train = y_train.apply(encoder).values
    network.train(x_train, y_train, epochs=1000)
    predictions = []
    for x in x_train:
        predictions.append(network.forward_propagate(x))
    print('predictions: {}'.format([x[0][0] for x in predictions]))
    print('labels:      {}'.format(y_train))