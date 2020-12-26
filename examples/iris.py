import pandas as pd
from torch import tensor

from hand import Model, accuracy, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from hand.activation import Relu
from hand.layer import Linear

if __name__ == '__main__':
    # data
    df = pd.read_csv('../data/iris.data')
    df = df.sample(frac=1).reset_index(drop=True)
    x = df.loc[:, 'f1':'f4'].values
    y_temp = df.loc[:, 'y']
    encoder = {
        'Iris-setosa': np.array([1, 0, 0]),
        'Iris-versicolor': np.array([0, 1, 0]),
        'Iris-virginica': np.array([0, 0, 1]),
    }
    y = []
    for y_ in y_temp:
        y.append(encoder[y_])
    y = np.array(y)
    x_train, x_test, y_train, y_test = map(lambda _x: tensor(_x).float(), train_test_split(x, y, test_size=0.33))

    num_features = x_train.shape[1]
    num_classes = y_train.shape[1]

    # model
    model = Model([
        Linear(256, n_in=num_features),
        Relu(),
        Linear(256),
        Relu(),
        Linear(num_classes)
    ])

    # dataset
    train_dataset = Dataset(x_train, y_train)

    # train model
    # epochs
    for _ in range(5):
        # batches
        for x_batch, y_batch in train_dataset:
            loss = model(x_batch, y_batch)
            model.backward()

    model.eval()
    predictions = []
    probs = []
    for x, y in zip(x_test, y_test):
        prediction, prob = model(x, y)
        predictions.append(prediction)
        probs.append(prob)
    predictions_arr = torch.stack(predictions)
    print(accuracy(predictions_arr, y_test))
