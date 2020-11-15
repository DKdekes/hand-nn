import math
import pandas as pd
from hand import Model
import torch
from torch import tensor
from hand.activate import Relu
from hand.layer import Linear

if __name__ == '__main__':
    df = pd.read_csv('../data/bit_data.csv')
    x_train = df.loc[:, 'bit_3':'bit_1'].values
    y_train = df.loc[:, 'target'].values

    x_train, y_train = map(tensor, (x_train, y_train))
    x_train = x_train.float()
    y_train = y_train.float()

    # initialize model
    # num nodes / layer
    nh = 50
    n, m = x_train.shape
    c = y_train.max() + 1

    # parameter init
    w1 = torch.randn(m, nh)/math.sqrt(m)
    b1 = torch.zeros(nh)
    w2 = torch.randn(nh, 1) / math.sqrt(nh)
    b2 = torch.zeros(1)

    model = Model([
        Linear(w1, b1),
        Relu(),
        Linear(w2, b2)
    ])

    # train model
    for _ in range(1000):
        loss = model(x_train, y_train)
        model.backward()

    print(model(x_train, y_train))
