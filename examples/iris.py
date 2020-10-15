import pandas as pd
from hand import Network, accuracy
from sklearn.model_selection import train_test_split
import numpy as np

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # model
    n_inputs_ = x.shape[1]
    hidden_layers_ = [5]
    n_outputs_ = y.shape[1]
    network = Network(n_inputs_, hidden_layers_, n_outputs_)
    network.train(x_train, y_train, epochs=100)
    predictions = []
    for x, y in zip(x_test, y_test):
        print('expecting: {}'.format(y))
        prediction = network.predict(x)
        print('got: {}'.format(network.predict(x)))
        predictions.append(prediction)
    predictions = np.array(predictions)
    print(accuracy(predictions, y_test))
    print(network.report_weights())
