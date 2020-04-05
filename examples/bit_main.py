import pandas as pd
from hand.network.Network import Network

if __name__ == '__main__':
    n_inputs_ = 3
    hidden_layers_ = [3, 3, 3]
    n_outputs_ = 1
    network = Network(n_inputs_, hidden_layers_, n_outputs_)
    df = pd.read_csv('../data/bit_data.csv')
    x_train = df.loc[:, 'bit_3':'bit_1'].values
    y_train = df.loc[:, 'target'].values
    network.train(x_train, y_train)
