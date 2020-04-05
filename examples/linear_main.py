import pandas as pd
from hand.network.Network import Network

if __name__ == '__main__':
    n_inputs_ = 1
    hidden_layers_ = [1]
    n_outputs_ = 1
    network = Network(n_inputs_, hidden_layers_, n_outputs_)
    df = pd.read_csv('../data/linear_data.csv')
    x_train = df.loc[:, 'X'].values
    y_train = df.loc[:, 'y'].values
    network.train(x_train, y_train)
