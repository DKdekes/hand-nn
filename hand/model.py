from hand import error


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
        self.layers = layers

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

    def __call__(self, x, target):
        for layer in self.layers:
            x = layer(x)
        return x, self.loss(x, target)

    def backward(self):
        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()
