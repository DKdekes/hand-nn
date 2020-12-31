from hand import error
import torch
import logging

class Model:
    """
    the top level module for creating networks

    usage:
        model = Model([
            Linear(),
            Relu(),
            Linear()
        ])

    it would be nice to have an inference mode
    """
    def __init__(self, layers, loss='mse', lr=None):
        """
        :param layers: a list of instantiated layers
        :param loss: kind of loss function to use
        :param lr: learning rate. defined at layer level, but can also be defined at model level
        """
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.layers = layers
        self.train = True

        # loss setup
        self.loss = getattr(error, loss)()

        # learning rate setup
        if lr:
            for layer in self.layers:
                layer.lr = lr

        # layer initialization
        weight_layers = [x for x in self.layers if hasattr(x, 'w')]
        for i, layer in enumerate(weight_layers):
            if i == 0:
                layer.setup(layer.n_in)
            else:
                layer.setup(weight_layers[i - 1].units)

    def eval(self):
        self.train = False

    def __call__(self, x, target):
        if self.train:
            self.logger.debug('---propagating forward---')
            assert len(target.shape) != 1, 'target variables cannot be stored in 1d tensor'
            for layer in self.layers:
                x = layer(x)
            return x, self.loss(x, target)
        else:
            for layer in self.layers:
                x = layer(x)
            ret = torch.zeros(x.shape)
            ret[0, torch.argmax(x)] = 1
            return ret.squeeze(), x

    def backward(self):
        self.logger.debug('---propagating backward---')
        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()

    def __str__(self):
        return str([layer.__str__() for layer in self.layers])
