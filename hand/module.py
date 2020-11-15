

class Module:
    """
    the abstract class for any module in the network. even Model inherits Module.
    """
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self, *args):
        raise Exception('not implemented')

    def bwd(self, out, *args):
        raise Exception('not implemented')

    def backward(self):
        self.bwd(self.out, *self.args)
