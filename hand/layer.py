from hand.module import Module


class Linear(Module):
    def __init__(self, w, b, lr=0.01):
        """

        :param w: weight tensor
        :param b: bias tensor
        :param lr: learning rate. want to experiment with multiple learning rates in future, so the learning
        rate is defined at the layer level for now
        """
        self.w = w
        self.b = b
        self.lr = lr

    def forward(self, inp):
        return inp @ self.w + self.b

    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g
        self.b.g = out.g.sum()
        self.update()

    def update(self):
        """
        move in the opposite direction of the gradient to minimize loss (movement scaled by learning rate)
        :return:
        """
        self.w -= self.w.g * self.lr
        self.b -= self.b.g * self.lr
