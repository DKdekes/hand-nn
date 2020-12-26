import math
import pandas as pd
from hand import Model
import torch
from torch import tensor
from hand.activation import Relu
from hand.layer import Linear

if __name__ == '__main__':
    df = pd.read_csv('../data/bit_data.csv')
    x_train = df.loc[:, 'bit_3':'bit_1'].values
    y_train = df.loc[:, 'target'].values.reshape(-1, 1)

    x_train, y_train = map(tensor, (x_train, y_train))
    x_train, y_train = map(lambda x: x.float(), (x_train, y_train))

    num_features = x_train.shape[1]
    num_classes = 1

    model = Model([
        Linear(256, n_in=num_features),
        Relu(),
        Linear(256),
        Relu(),
        Linear(num_classes)
    ])

    # train model
    for _ in range(1000):
        loss = model(x_train, y_train)
        model.backward()

    print(model(x_train, y_train))
