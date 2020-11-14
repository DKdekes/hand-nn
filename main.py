from hand import Model
import pandas as pd

n_inputs_ = 1
hidden_layers_ = [1]
n_outputs_ = 1
network = Model(n_inputs_, hidden_layers_, n_outputs_)
df = pd.read_csv('./data/linear_data.csv')
x_train = df.loc[:, 'X'].values
y_train = df.loc[:, 'y'].values
network.train(x_train, y_train, epochs=1000)
for x in x_train:
    print(network.forward_propagate(x))
network.report_weights()
