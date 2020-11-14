from hand.error import Mse


class Model:
    def __init__(self, layers, loss='mse'):
        self.layers = layers
        if loss == 'mse':
            self.loss = Mse()
        else:
            raise Exception(f'{loss} not implemented')

    def __call__(self, x, target):
        for layer in self.layers:
            x = layer(x)
        return self.loss(x, target)

    def backward(self):
        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()
